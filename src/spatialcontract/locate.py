"""Oracle-guided fault localisation for bugs inside a library. Deterministic, no LLM.

The oracle already runs the library on phantoms it can score. ``locate`` runs the same driver on a bank of phantoms
under ``sys.setprofile``, restricted to the library's own files, and records which library functions each run called
and which of them returned a 4x4 affine (or an object carrying one). Phantoms the code gets right and phantoms it gets
wrong then rank the functions:

- spectrum score (Ochiai): executed by the failing phantoms and not by the passing ones;
- affine provenance: the function produced an affine in a failing run, or produced exactly the affine that ended up
  in the (wrong) output file;
- a small static prior: how much of the function's own source is about affines, orientation and spacing.

The weights below were fixed before the first measurement and have not been tuned on the real-bug corpus.
The ground truth (``truth_functions``: functions the upstream fix edited) is only used to report hit@k; it is never
shown to the agent.
"""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import oracle
from .phantom import make_phantom

EXTRA_SEEDS = (3, 4, 5, 6, 7, 8)          # added to the visible bank so that some phantoms can pass; never the held-out seeds
W_SPECTRUM, W_AFFINE, W_TERMS, W_FINAL = 1.0, 0.5, 0.3, 0.3
SPATIAL_TERMS = re.compile(r"affine|direction|spacing|origin|orient|ornt|flip|permut|transpose|axcode|qform|sform|zoom|"
                           r"\bras\b|\blps\b|world|voxel|crop|pad|shape_offset", re.I)

_TRACER = r'''
import importlib.util, json, os, sys
code, inp, outp, root, dump = sys.argv[1:6]
root = os.path.normcase(os.path.abspath(root)) + os.sep
calls, affines = {}, []
def _aff(v):
    try:
        if tuple(getattr(v, "shape", ())) == (4, 4):
            return [[float(x) for x in row] for row in v.tolist()]
    except Exception:
        pass
    return None
def prof(frame, event, arg):
    if event != "call" and event != "return":
        return
    co = frame.f_code
    if not os.path.normcase(co.co_filename).startswith(root) or co.co_name.startswith("<"):
        return
    key = os.path.relpath(co.co_filename, root).replace(os.sep, "/") + ":" + str(co.co_firstlineno) + ":" + co.co_qualname
    if event == "call":
        calls[key] = calls.get(key, 0) + 1
        return
    if len(affines) >= 4000:
        return
    a = _aff(arg)
    if a is None and isinstance(arg, (tuple, list)) and len(arg) <= 4:
        a = next((x for x in map(_aff, arg) if x is not None), None)
    if a is None and isinstance(getattr(type(arg), "affine", None), property):
        try:
            a = _aff(arg.affine)
        except Exception:
            a = None
    if a is not None:
        affines.append((key, a))
spec = importlib.util.spec_from_file_location("code_under_test", code)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.setprofile(prof)
try:
    m.process(inp, outp)
finally:
    sys.setprofile(None)
    with open(dump, "w") as f:
        json.dump({"calls": calls, "affines": affines}, f)
'''


@dataclass
class Suspect:
    file: str                  # path relative to the library root, e.g. 'torchio/transforms/.../crop.py'
    line: int
    qualname: str
    fail_runs: int = 0
    pass_runs: int = 0
    calls: int = 0
    returned_affine: bool = False
    produced_final_affine: bool = False
    terms: int = 0
    score: float = 0.0
    spectrum: float = 0.0      # the Ochiai part of the score alone (for the baseline comparison)
    first_seen: int = 0        # execution order of the first call (for the baseline comparison)
    signature: str = ""

    @property
    def key(self) -> tuple:
        return (self.file, self.qualname)


@dataclass
class Localisation:
    n_fail: int
    n_pass: int
    suspects: list = field(default_factory=list)
    statuses: dict = field(default_factory=dict)      # seed -> PASS | FAIL | ERROR

    def summary(self, k: int = 8) -> str:
        head = (f"Runtime trace of the driver on {self.n_fail + self.n_pass} phantoms ({self.n_fail} wrong, {self.n_pass} right): "
                f"library functions ranked by how strongly their execution is associated with wrong output and with producing affines.")
        rows = []
        for i, s in enumerate(self.suspects[:k], 1):
            tags = [f"ran in {s.fail_runs}/{self.n_fail} wrong and {s.pass_runs}/{self.n_pass} right runs"]
            if s.produced_final_affine:
                tags.append("returned the affine that was written to the output")
            elif s.returned_affine:
                tags.append("returned an affine")
            rows.append(f"{i}. {s.file}:{s.line} {s.qualname}  [{'; '.join(tags)}]\n     {s.signature}")
        return head + "\n" + "\n".join(rows)


@lru_cache(maxsize=256)
def _functions(path: str) -> tuple:
    """(start, end, qualname, def-line text) for every function in a file; start includes decorators."""
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + ch.name
                start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.append((start, ch.end_lineno, q, lines[ch.lineno - 1].strip()[:200], ch.lineno))
                walk(ch, q + ".<locals>.")
            elif isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
    try:
        walk(ast.parse(src), "")
    except SyntaxError:
        pass
    return tuple(out)


def _source_terms(path: Path, start: int) -> tuple[int, str]:
    for s, e, q, sig, defline in _functions(str(path)):
        if start in (s, defline):
            body = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[s - 1:e])
            return len(SPATIAL_TERMS.findall(body)), sig
    return 0, ""


def trace(code_path, contract, lib_root: Path, seeds, python=None, pythonpath=None) -> list[dict]:
    """One traced run per seed: {'seed', 'status', 'calls', 'affines', 'out_affine'}."""
    runs = []
    with tempfile.TemporaryDirectory(prefix="sc_locate_") as d:
        tmp = Path(d)
        for seed in seeds:
            ph = make_phantom(seed)
            inp, outp, dump = tmp / f"in_{seed}.nii.gz", tmp / f"out_{seed}.nii.gz", tmp / f"trace_{seed}.json"
            ph.save(inp)
            ok, err = oracle._run_code(Path(code_path).resolve(), inp, outp, tmp, python, pythonpath, runner=_TRACER,
                                       extra_args=(Path(lib_root).resolve(), dump))
            rec = json.loads(dump.read_text()) if dump.exists() else {"calls": {}, "affines": []}
            status, out_aff = "ERROR", None
            if ok:
                try:
                    status = oracle.measure(outp, contract, ph).status
                    import nibabel as nib
                    out_aff = np.asarray(nib.load(str(outp)).affine, dtype=float)
                except Exception:
                    status = "ERROR"
            runs.append({"seed": seed, "status": status, "calls": rec["calls"], "affines": rec["affines"], "out_affine": out_aff,
                         "detail": err})
    return runs


def rank(runs: list[dict], lib_root: Path) -> Localisation:
    """Score every library function that ran in at least one wrong run (ERROR counts as wrong)."""
    fail = [r for r in runs if r["status"] != "PASS"]
    ok = [r for r in runs if r["status"] == "PASS"]
    loc = Localisation(len(fail), len(ok), statuses={r["seed"]: r["status"] for r in runs})
    table: dict[str, Suspect] = {}
    for r in runs:
        wrong = r["status"] != "PASS"
        for key, n in r["calls"].items():
            f, line, q = key.split(":", 2)
            s = table.setdefault(key, Suspect(f, int(line), q, first_seen=len(table)))
            s.calls += n
            if wrong:
                s.fail_runs += 1
            else:
                s.pass_runs += 1
        if wrong:
            for key, a in r["affines"]:
                if key in table:
                    table[key].returned_affine = True
                    if r["out_affine"] is not None and np.allclose(np.asarray(a), r["out_affine"], atol=1e-4):
                        table[key].produced_final_affine = True
    for s in table.values():
        if not s.fail_runs:
            continue
        s.terms, s.signature = _source_terms(Path(lib_root) / s.file, s.line)
        s.spectrum = spectrum = s.fail_runs / math.sqrt(len(fail) * (s.fail_runs + s.pass_runs)) if fail else 0.0
        s.score = (W_SPECTRUM * spectrum + W_AFFINE * s.returned_affine + W_TERMS * min(s.terms, 5) / 5
                   + W_FINAL * s.produced_final_affine)
        loc.suspects.append(s)
    loc.suspects.sort(key=lambda s: (-s.score, s.calls, s.file, s.line))
    return loc


def locate(code_path, contract, lib_root: Path, seeds, python=None, pythonpath=None) -> Localisation:
    return rank(trace(code_path, contract, lib_root, tuple(seeds) + tuple(s for s in EXTRA_SEEDS if s not in seeds),
                      python, pythonpath), lib_root)


# ----------------------------------------------------------------------------- ground truth (evaluation only)
def truth_functions(repo_dir: Path, pre: str, post: str, files: list[str], pre_root: Path, strip: str = "") -> list[tuple]:
    """(file, qualname) of every pre-fix function containing a line the upstream fix changed. ``files`` are library-root
    relative (as in libagent.fix_files); ``strip`` is the repo-relative prefix in front of them (e.g. 'src/')."""
    out = set()
    for f in files:
        diff = subprocess.run(["git", "diff", "-U0", pre, post, "--", strip + f], cwd=repo_dir, capture_output=True,
                              text=True, check=True).stdout
        changed = []
        for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", diff, re.M):
            a, b = int(m.group(1)), int(m.group(2) or 1)
            changed += list(range(a, a + b)) if b else [a]
        funcs = _functions(str(pre_root / f))
        for ln in changed:
            inner = [x for x in funcs if x[0] <= ln <= x[1]]
            out.add((f, min(inner, key=lambda x: x[1] - x[0])[2] if inner else "<module>"))
    return sorted(out)


def hits(loc: Localisation, truth_funcs: list[tuple], ks=(1, 3, 5, 10)) -> dict:
    """Function- and file-level hit@k of the ranking against the upstream fix."""
    truth_files = {f for f, _ in truth_funcs}
    keys = [s.key for s in loc.suspects]
    first_func = next((i + 1 for i, k in enumerate(keys) if k in set(truth_funcs)), None)
    first_file = next((i + 1 for i, k in enumerate(keys) if k[0] in truth_files), None)
    return {"rank_first_true_function": first_func, "rank_first_true_file": first_file,
            **{f"func@{k}": bool(first_func and first_func <= k) for k in ks},
            **{f"file@{k}": bool(first_file and first_file <= k) for k in ks},
            "n_ranked": len(keys), "n_fail": loc.n_fail, "n_pass": loc.n_pass}
