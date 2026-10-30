"""Repair a bug that lives INSIDE a library (the real-bug corpus), not in a small driver script.

The agent works on a private copy of the library package at the pre-fix commit. It is given the driver script (read-only), the fixed
contract and the oracle's measurements, plus tools to list, read, search and edit the library, and to re-run the oracle. It is not told
which file is at fault. A patch that makes the visible oracle pass goes through the same held-out validation as everything else.
Besides REPAIRED / OVERFIT / FAILED we record whether the agent edited a file the upstream fix also edited (localisation).
"""
from __future__ import annotations

import difflib
import fnmatch
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import heldout, locate, oracle
from .agent import Log
from .contract import load_from_source
from .realbugs import ENVS, WORK, RealCase

MAX_READ_LINES = 150
MAX_HITS = 40
FORBIDDEN_ADDED = ("subprocess", "socket", "os.system", "urllib", "requests", "eval(", "exec(", "__import__", "shutil.rmtree")
PACKAGE = {"torchio": "torchio", "nilearn": "nilearn", "nibabel": "nibabel", "MONAI": "monai"}


def fix_files(case: RealCase) -> list[str]:
    """Package-relative paths of the non-test source files the upstream fix changed (ground truth, never shown to the agent)."""
    out = subprocess.run(["git", "diff", "--name-only", case.pre, case.post], cwd=WORK / "repos" / case.repo, capture_output=True, text=True, check=True).stdout
    pkg = PACKAGE[case.repo]
    files = []
    for f in out.split():
        f = f[4:] if f.startswith("src/") else f
        if f.startswith(pkg + "/") and f.endswith(".py") and "/tests/" not in f:
            files.append(f)
    return files


class Workspace:
    """A private, editable copy of the library package at the pre-fix commit."""

    def __init__(self, case: RealCase, root: Path | None = None):
        self.case = case
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix=f"sc_lib_{case.id}_"))
        self.pkg = PACKAGE[case.repo]
        src = case.worktree("pre") / case.src_subdir / self.pkg
        self.lib = self.root / "lib"
        shutil.copytree(src, self.lib / self.pkg, ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc", "*.so", "*.pyd"))
        self.originals: dict[str, str] = {}

    @property
    def pythonpath(self) -> list:
        return [self.lib, WORK / "shims"]

    def _resolve(self, rel: str) -> Path:
        p = (self.lib / rel).resolve()
        if not str(p).startswith(str((self.lib / self.pkg).resolve())):
            raise ValueError(f"path must be inside the package '{self.pkg}/'")
        return p

    def list_dir(self, rel: str = "") -> str:
        p = self._resolve(rel or self.pkg)
        if not p.is_dir():
            return f"not a directory: {rel}"
        items = sorted(p.iterdir(), key=lambda q: (q.is_file(), q.name))
        lines = []
        for q in items:
            if q.name == "__pycache__":
                continue
            name = q.relative_to(self.lib).as_posix()
            if q.is_dir():
                lines.append(name + "/")
            elif q.suffix == ".py":
                with q.open(encoding="utf-8", errors="ignore") as fh:
                    lines.append(f"{name}  ({sum(1 for _ in fh)} lines)")
            else:
                lines.append(name)
        return "\n".join(lines[:100]) + (f"\n... {len(lines) - 100} more" if len(lines) > 100 else "")

    def read_file(self, rel: str, start: int = 1, end: int | None = None) -> str:
        p = self._resolve(rel)
        if not p.is_file():
            return f"no such file: {rel}"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start))
        end = min(len(lines), int(end) if end else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        body = "\n".join(f"{i + 1:5d}|{lines[i]}" for i in range(start - 1, end))
        return f"{rel} lines {start}-{end} of {len(lines)}\n{body}"

    def search(self, pattern: str, glob: str = "*.py") -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"bad regex: {e}"
        hits = []
        for f in sorted((self.lib / self.pkg).rglob("*.py")):
            rel = f.relative_to(self.lib).as_posix()
            # accept any of: a bare name ('processing.py'), a pattern on the file name, or a path pattern relative to the library root
            if not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f.name, glob) or fnmatch.fnmatch(rel, "*" + glob.lstrip("*/"))):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(self.lib).as_posix()}:{i}: {line.strip()[:160]}")
                    if len(hits) >= MAX_HITS:
                        return "\n".join(hits) + f"\n(stopped at {MAX_HITS} matches; refine the pattern)"
        return "\n".join(hits) or "no matches"

    def replace_in_file(self, rel: str, old: str, new: str) -> str:
        try:
            p = self._resolve(rel)
        except ValueError as e:
            return f"rejected: {e}"
        if not p.is_file() or p.suffix != ".py":
            return f"rejected: {rel} is not an existing .py file"
        text = p.read_text(encoding="utf-8")
        if not old:
            return "rejected: old_string is empty"
        n = text.count(old)
        if n == 0:
            return ("rejected: old_string occurs 0 times. " + self._closest(text, old)
                    + " (copy only the text after the '|', keep the file's exact indentation, or use replace_lines)")
        if n != 1:
            return f"rejected: old_string occurs {n} times (it must occur exactly once; include more context, or use replace_lines)"
        added = [l for l in new.splitlines() if l not in old]
        for tok in FORBIDDEN_ADDED:
            if any(tok in l for l in added):
                return f"rejected: the edit adds '{tok}', which is not allowed"
        updated = text.replace(old, new)
        try:
            compile(updated, rel, "exec")
        except SyntaxError as e:
            return f"rejected: the edit makes {rel} a syntax error ({e.msg}, line {e.lineno}); nothing was changed"
        self.originals.setdefault(rel, text)
        p.write_text(updated, encoding="utf-8")
        return f"edited {rel}"

    @staticmethod
    def _closest(text: str, old: str) -> str:
        lines = text.splitlines()
        first = next((l.strip() for l in old.splitlines() if l.strip()), "")
        ratio = lambda i: difflib.SequenceMatcher(None, lines[i].strip(), first).ratio()   # noqa: E731
        best = max(range(len(lines)), key=ratio, default=None)
        if best is None or ratio(best) < 0.6:
            return "No similar line found."
        lo, hi = max(0, best - 1), min(len(lines), best + 4)
        return "Closest region in the file:\n" + "\n".join(f"{i + 1:5d}|{lines[i]}" for i in range(lo, hi))

    def replace_lines(self, rel: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace lines start_line..end_line (inclusive, 1-based, as numbered by read_file) with new_text."""
        try:
            p = self._resolve(rel)
        except ValueError as e:
            return f"rejected: {e}"
        if not p.is_file() or p.suffix != ".py":
            return f"rejected: {rel} is not an existing .py file"
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        start, end = int(start_line), int(end_line)
        if not (1 <= start <= end <= len(lines)):
            return f"rejected: line range {start}-{end} is outside the file (1-{len(lines)})"
        old = "".join(lines[start - 1:end])
        added = [l for l in new_text.splitlines() if l not in old]
        for tok in FORBIDDEN_ADDED:
            if any(tok in l for l in added):
                return f"rejected: the edit adds '{tok}', which is not allowed"
        new_block = new_text if (new_text.endswith("\n") or not new_text) else new_text + "\n"
        updated = "".join(lines[:start - 1]) + new_block + "".join(lines[end:])
        try:
            compile(updated, rel, "exec")
        except SyntaxError as e:
            return f"rejected: the edit makes {rel} a syntax error ({e.msg}, line {e.lineno}); nothing was changed"
        self.originals.setdefault(rel, text)
        p.write_text(updated, encoding="utf-8")
        new_lines = updated.splitlines()
        lo, hi = max(0, start - 3), min(len(new_lines), start - 1 + len(new_block.splitlines()) + 2)
        return f"edited {rel}; it now reads:\n" + "\n".join(f"{i + 1:5d}|{new_lines[i]}" for i in range(lo, hi))

    def revert_file(self, rel: str) -> str:
        try:
            p = self._resolve(rel)
        except ValueError as e:
            return f"rejected: {e}"
        if rel not in self.originals:
            return f"{rel} has not been edited"
        p.write_text(self.originals.pop(rel), encoding="utf-8")
        return f"reverted {rel} to the original library version"

    def touched(self) -> list[str]:
        return sorted(r for r, orig in self.originals.items() if self._resolve(r).read_text(encoding="utf-8") != orig)

    def diff(self) -> str:
        out = []
        for rel in self.touched():
            cur = self._resolve(rel).read_text(encoding="utf-8")
            out += list(difflib.unified_diff(self.originals[rel].splitlines(True), cur.splitlines(True), f"a/{rel}", f"b/{rel}", n=2))
        return "".join(out)

    def oracle(self, contract, seeds=None):
        return oracle.check(self.case.script, contract, seeds=seeds or self.case.seeds or oracle.DEFAULT_SEEDS, python=str(ENVS[self.case.env]), pythonpath=self.pythonpath)

    def locate(self, contract):
        """Runtime-trace ranking of library functions over the visible bank plus locate.EXTRA_SEEDS (never the held-out seeds)."""
        return locate.locate(self.case.script, contract, self.lib, self.case.seeds or oracle.DEFAULT_SEEDS, python=str(ENVS[self.case.env]),
                             pythonpath=self.pythonpath)

    def truth_functions(self) -> list:
        """Functions the upstream fix edited, for evaluation only."""
        c = self.case
        return locate.truth_functions(WORK / "repos" / c.repo, c.pre, c.post, fix_files(c), self.lib,
                                      strip=(c.src_subdir + "/") if c.src_subdir else "")

    def validate(self, contract, run_seed: int = 0, gates: bool = True):
        """Held-out validation (with the no-op check), then, if it passes and ``gates``, the library's own covering tests and the
        reader invariant (libverify.py): a repair counts only if none of them fails."""
        held = self._heldout(contract, run_seed)
        if gates and held["passed"]:
            from . import libverify
            for name, gate in (("own_tests", libverify.regression), ("reader", libverify.reader)):
                g = gate(self)
                held["summary"][name] = g["verdict"]
                held.setdefault("gates", {})[name] = g
            held["passed"] = all(v == "PASS" for v in held["summary"].values())
        return held

    def _heldout(self, contract, run_seed: int = 0):
        return heldout.validate(self.case.script, contract, run_seed=run_seed, python=str(ENVS[self.case.env]), pythonpath=self.pythonpath,
                                hidden_seeds=self.case.hidden_seeds or None,
                                original={"code_path": self.case.script, "python": str(ENVS[self.case.env]),
                                          "pythonpath": self.case.pythonpath("pre")})

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
I = {"type": "integer"}
TOOLS = [
    _tool("list_dir", "List a directory of the library package (paths are relative to the library root, e.g. 'torchio/transforms').", {"path": S}, []),
    _tool("read_file", f"Read up to {MAX_READ_LINES} numbered lines of a library file.", {"path": S, "start": I, "end": I}, ["path"]),
    _tool("search", "Regex search over the library's .py files; returns 'path:line: text' (max 40 hits).", {"pattern": S, "glob": S}, ["pattern"]),
    _tool("replace_in_file", "Edit a library file: replace the single exact occurrence of old_string with new_string. Rejected if old_string is not unique or the result does not compile.",
          {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
    _tool("replace_lines", "Edit a library file by line numbers (as shown by read_file): replace lines start_line..end_line (inclusive) with new_text, which must "
          "carry the file's real indentation and no line numbers. Re-read the file after an edit: line numbers shift.",
          {"path": S, "start_line": I, "end_line": I, "new_text": S}, ["path", "start_line", "end_line", "new_text"]),
    _tool("revert_file", "Undo all your edits to one library file (restores the original).", {"path": S}, ["path"]),
    _tool("run_oracle", "Run the driver script against your edited library and return the oracle's landmark measurements.", {}, []),
    _tool("give_up", "Stop without a fix.", {"reason": S}, ["reason"]),
]
TOOL_SUSPECTS = _tool("get_suspects", "Return the top-k library functions from the runtime trace ranking (default 15, max 30).", {"k": I}, [])

SYSTEM = """You fix spatial / orientation / affine bugs inside a medical-imaging LIBRARY (the code is a private copy; edit it with the tools).
A small driver script calls the library; its output is wrong with respect to the SPATIAL_CONTRACT. The driver and the contract are fixed and are NOT where the bug is: the
defect is in the library source. A deterministic oracle pushes asymmetric landmark phantoms through the driver and reports, per landmark, where it landed versus where the
contract says it must be (mm). You are not told which file is at fault: use list_dir, search and read_file to locate it, make the smallest correct edit with replace_lines (or replace_in_file),
then call run_oracle to check.

read_file shows 'N|text': everything after the first '|' is the exact file content (including its indentation), N is only a line number.

Conventions: a NIfTI affine maps voxel index (voxel centre) to world coordinates. RAS+ means +x Right, +y Anterior, +z Superior; LPS+ is the opposite in x and y.
"array_orientation", when present, means the output voxel axes (i, j, k) must run toward those patient directions (aff2axcodes of the output affine, in RAS).

Rules:
1. Fix the root cause in the library, in general. Do not special-case inputs, shapes, orientations or voxel sizes; hidden phantoms with other orientations, voxel sizes and
   obliquity, a fresh phantom and a non-phantom image are checked after the oracle passes.
2. Minimal edits. Do not edit tests, the driver or the contract. Do not add subprocess, network, eval or exec.
3. You have a limited number of tool calls and oracle runs; stop with give_up if you cannot find the cause.
"""

SYSTEM_LOCALIZE = """
The first message also contains a runtime trace: the driver was run under a profiler on phantoms the library gets right and on phantoms
it gets wrong, and the library functions that ran are ranked by how strongly they are associated with the wrong output and with producing
affines. The ranking is a statistical hint, not an answer: the defect is usually in or near a high-ranked function, but read the code
before editing. get_suspects returns more of the ranking.
"""


@dataclass
class LibResult:
    case: str
    outcome: str                       # REPAIRED | OVERFIT | FAILED | GAVE_UP | BUDGET | ERROR
    steps: int = 0
    oracle_runs: int = 0
    llm_calls: int = 0
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    touched: list = field(default_factory=list)
    upstream_files: list = field(default_factory=list)
    localized: bool = False
    localisation: dict | None = None   # hit@k of the trace ranking against the upstream fix (never shown to the agent)
    visible_passed: bool = False
    heldout: dict | None = None
    diff: str = ""
    reasons: list = field(default_factory=list)
    tool_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _prune(messages: list, keep_last: int = 6) -> None:
    """Replace the content of old read-only tool results (list_dir / read_file / search) so the context stops growing without bound.
    Edits and oracle results are always kept."""
    names = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            names[c["id"]] = c["function"]["name"]
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for i in tool_idx[:-keep_last]:
        m = messages[i]
        if names.get(m["tool_call_id"]) in ("list_dir", "read_file", "search") and not m["content"].startswith("[earlier"):
            m["content"] = f"[earlier {names[m['tool_call_id']]} output omitted to save space; call it again if you need it]"


def repair_library(case: RealCase, client, *, max_steps: int = 40, max_oracle_runs: int = 8, max_tokens: int = 8000, max_run_tokens: int = 300_000,
                   log_path: Path | None = None, run_seed: int = 0, llm_kwargs: dict | None = None, keep: bool = False,
                   localize: bool = False, diagnose: bool = True) -> LibResult:
    """``localize`` adds the runtime-trace ranking (locate.py) to the first message plus a get_suspects tool; ``diagnose`` controls
    the misplacement pattern in oracle feedback. Both off reproduces the configuration of the real3 run."""
    from .llm import BudgetExceeded

    contract = load_from_source(case.script.read_text(encoding="utf-8"))
    log = Log(log_path)
    ws = Workspace(case)
    res = LibResult(case=case.id, outcome="FAILED", upstream_files=fix_files(case))
    t0 = time.time()
    try:
        first = ws.oracle(contract)
        first.diagnose = diagnose
        log.event("start", case=case.id, initial_verdict=first.verdict, feedback=first.feedback(max_lines=12), upstream_files=res.upstream_files)
        driver = case.script.read_text(encoding="utf-8")
        tools, trace_text, loc = TOOLS, "", None
        if localize:
            loc = ws.locate(contract)
            trace_text = "\n\n" + loc.summary(8)
            tools = TOOLS + [TOOL_SUSPECTS]
            try:
                res.localisation = locate.hits(loc, ws.truth_functions())
            except Exception as e:          # ground truth needs the git clone; the run itself does not
                res.localisation = {"error": f"{type(e).__name__}: {e}"}
            log.event("localisation", n_fail=loc.n_fail, n_pass=loc.n_pass, shown=trace_text.strip(), hits=res.localisation)
        messages = [
            {"role": "system", "content": SYSTEM + (SYSTEM_LOCALIZE if localize else "")},
            {"role": "user", "content": (
                f"Library: {PACKAGE[case.repo]} (private copy). Driver script (read-only):\n```python\n{driver}```\n\n"
                f"SPATIAL_CONTRACT (fixed): {json.dumps(contract.to_dict())}\n\nOracle report with the unmodified library:\n{first.feedback(max_lines=12)}\n\n"
                f"Library top level:\n{ws.list_dir('')}{trace_text}")},
        ]
        for step in range(1, max_steps + 1):
            res.steps = step
            try:
                resp = client.chat(messages, max_tokens=max_tokens, tools=tools, tool_choice="auto", **(llm_kwargs or {}))
            except BudgetExceeded as e:
                res.outcome, res.reasons = "BUDGET", [str(e)]
                break
            except Exception as e:
                res.outcome, res.reasons = "ERROR", [f"{type(e).__name__}: {str(e)[:200]}"]
                break
            u = resp.usage
            if u:
                res.tokens["prompt"] += u.prompt_tokens
                res.tokens["completion"] += u.completion_tokens
            res.llm_calls += 1
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            log.event("llm_response", step=step, finish=resp.choices[0].finish_reason, n_tool_calls=len(calls), content=(msg.content or "")[:800],
                      tokens=dict(res.tokens))
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
            messages.append(assistant)
            if not calls:
                messages.append({"role": "user", "content": "Use the tools: search/read the library, edit it with replace_in_file, check with run_oracle, or call give_up."})
                continue
            done = False
            for c in calls:
                name = c.function.name
                res.tool_counts[name] = res.tool_counts.get(name, 0) + 1
                try:
                    a = json.loads(c.function.arguments or "{}")
                except ValueError as e:
                    out = f"tool arguments were not valid JSON: {e}"
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
                    continue
                try:
                    if name == "list_dir":
                        out = ws.list_dir(a.get("path", ""))
                    elif name == "read_file":
                        out = ws.read_file(a["path"], a.get("start", 1), a.get("end"))
                    elif name == "search":
                        out = ws.search(a["pattern"], a.get("glob", "*.py"))
                    elif name == "replace_in_file":
                        out = ws.replace_in_file(a["path"], a["old_string"], a["new_string"])
                    elif name == "replace_lines":
                        out = ws.replace_lines(a["path"], a["start_line"], a["end_line"], a["new_text"])
                    elif name == "revert_file":
                        out = ws.revert_file(a["path"])
                    elif name == "run_oracle":
                        if res.oracle_runs >= max_oracle_runs:
                            out = f"oracle run limit ({max_oracle_runs}) reached; call give_up or finish with your last edit"
                            done = True
                        else:
                            res.oracle_runs += 1
                            rep = ws.oracle(contract)
                            rep.diagnose = diagnose
                            out = rep.feedback(max_lines=12)
                            log.event("oracle", step=step, run=res.oracle_runs, verdict=rep.verdict, feedback=out, touched=ws.touched())
                            if rep.passed:
                                res.visible_passed = True
                                done = True
                    elif name == "get_suspects" and loc is not None:
                        out = loc.summary(max(1, min(30, int(a.get("k", 15)))))
                    elif name == "give_up":
                        res.outcome, res.reasons = "GAVE_UP", [a.get("reason", "")]
                        out = "ok"
                        done = True
                    else:
                        out = f"unknown tool {name}"
                except KeyError as e:
                    out = f"missing argument {e}"
                except ValueError as e:
                    out = f"rejected: {e}"
                log.event("tool", step=step, name=name, args={k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in a.items()}, result=out[:600])
                messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if done:
                break
            _prune(messages)
            if res.tokens["prompt"] + res.tokens["completion"] > max_run_tokens:
                res.reasons.append("per-run token cap reached")
                break
        else:
            res.reasons.append("step cap reached")
        res.touched = ws.touched()
        res.diff = ws.diff()
        res.localized = bool(set(res.touched) & set(res.upstream_files))
        if res.visible_passed:
            held = ws.validate(contract, run_seed=run_seed)
            res.heldout = held
            res.outcome = "REPAIRED" if held["passed"] else "OVERFIT"
            log.event("heldout", passed=held["passed"], summary=held["summary"])
        log.event("end", outcome=res.outcome, touched=res.touched, localized=res.localized, seconds=round(time.time() - t0, 1))
        return res
    finally:
        if not keep:
            ws.cleanup()
