"""Staged repair of a bug inside a library: localise, propose whole-function rewrites, let the oracle choose. An alternative to the
free-roaming tool loop in libagent.py, after its logs showed most of the budget going to reading code (Agentless-style).

1. Localise without the model: the runtime trace (locate.py) ranks the library functions that ran; the top K are shown to the model
   with their full source, so it does not have to search or read.
2. Propose: N independent calls each return one candidate fix as a tool call that replaces whole functions (no line numbers, no
   exact-match strings, so no edit can fail to apply because of indentation).
3. Verify: every candidate is applied to a fresh copy and run through the visible oracle; the first that passes goes through the
   same held-out validation as everything else. Otherwise the best two candidates and their measurements are fed back for the
   next round. At most ``rounds`` x ``n`` calls.
The model never sees the held-out checks, and the ground truth (which functions the upstream fix edited) is only used to report
whether the accepted patch touched one of them.
"""
from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import locate
from .agent import Log
from .contract import load_from_source
from .libagent import PACKAGE, Workspace, fix_files
from .realbugs import RealCase

MAX_FUNC_LINES = 220

TOOL = [{"type": "function", "function": {
    "name": "propose_fix",
    "description": "Propose one fix: replace one or more whole library functions with corrected versions.",
    "parameters": {"type": "object", "required": ["rationale", "edits"], "properties": {
        "rationale": {"type": "string", "description": "Root cause in one or two sentences, and what the edit changes."},
        "edits": {"type": "array", "items": {"type": "object", "required": ["path", "function", "new_source"], "properties": {
            "path": {"type": "string", "description": "Library-relative file path exactly as shown, e.g. 'torchio/transforms/crop.py'."},
            "function": {"type": "string", "description": "Qualified name exactly as shown, e.g. 'Crop.apply_transform'."},
            "new_source": {"type": "string", "description": "The complete new function, from its 'def' line to its end, without "
                                                            "decorators. Indentation is normalised for you."}}}}}}}}]

SYSTEM = """You fix spatial / orientation / affine bugs inside a medical-imaging LIBRARY. A small driver script calls the library; its output
is wrong with respect to the SPATIAL_CONTRACT. The driver and the contract are correct and fixed: the defect is in the library.

A deterministic oracle pushes asymmetric landmark phantoms through the driver and reports, per landmark, where it landed versus where the
contract says it must be (mm), plus a one-line description of the misplacement. You are given the library functions a runtime trace
ranked as most suspicious, with their full source. Reply with ONE call to propose_fix that replaces whole functions with corrected
versions. You may only replace functions that are shown, or other functions in the same files if you name them exactly.

Conventions: a NIfTI affine maps voxel index (voxel centre) to world coordinates. RAS+ means +x Right, +y Anterior, +z Superior; LPS+ is the
opposite in x and y (ITK/SimpleITK use LPS). "array_orientation", when present, means the output voxel axes must run toward those patient
directions (aff2axcodes of the output affine, in RAS).

Rules: fix the root cause in general (hidden phantoms with other orientations, voxel sizes and obliquity are checked after you pass); keep
the change minimal; keep every function's signature; do not add subprocess, network, eval or exec."""


@dataclass
class StagedResult:
    case: str
    outcome: str                      # REPAIRED | OVERFIT | FAILED | BUDGET | ERROR
    rounds: int = 0
    llm_calls: int = 0
    candidates: int = 0
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    touched: list = field(default_factory=list)
    upstream_files: list = field(default_factory=list)
    localized: bool = False
    visible_passed: bool = False
    heldout: dict | None = None
    diff: str = ""
    reasons: list = field(default_factory=list)
    shown: list = field(default_factory=list)
    tried_files: list = field(default_factory=list)   # every file any candidate edited (localisation, comparable to libagent)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _function_table(ws: Workspace, rel: str) -> dict:
    """qualname -> (def line, end line) in the CURRENT file (line numbers shift after an edit)."""
    p = ws._resolve(rel)
    locate._functions.cache_clear()
    return {q: (defline, end) for _s, end, q, _sig, defline in locate._functions(str(p))}


def _source(ws: Workspace, rel: str, qualname: str) -> str | None:
    span = _function_table(ws, rel).get(qualname)
    if not span:
        return None
    lines = ws._resolve(rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[span[0] - 1:span[1]])


def apply_edits(ws: Workspace, edits: list) -> list[str]:
    """Replace whole functions; returns problems (empty = applied). Re-indents to the original def's column, compile-checks the file."""
    problems = []
    if isinstance(edits, str):                   # some calls return the list JSON-encoded inside the string
        try:
            edits = json.loads(edits)
        except ValueError:
            return ["edits must be a list of {path, function, new_source} objects"]
    for e in edits if isinstance(edits, list) else [edits]:
        if not isinstance(e, dict):
            problems.append("each edit must be an object with path, function and new_source")
            continue
        rel, q, new = str(e.get("path", "")).replace("\\", "/"), str(e.get("function", "")), str(e.get("new_source", ""))
        try:
            p = ws._resolve(rel)
        except ValueError as err:
            problems.append(f"{rel}: {err}")
            continue
        if not p.is_file():
            problems.append(f"{rel}: no such file")
            continue
        span = _function_table(ws, rel).get(q)
        if not span:
            problems.append(f"{rel}: no function named {q!r}")
            continue
        if any(tok in new for tok in ("subprocess", "socket", "os.system", "eval(", "exec(", "__import__")):
            problems.append(f"{q}: forbidden call in the new source")
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        indent = lines[span[0] - 1][:len(lines[span[0] - 1]) - len(lines[span[0] - 1].lstrip())]
        body = textwrap.indent(textwrap.dedent(new.strip("\n")), indent) + "\n"
        if not body.lstrip().startswith(("def ", "async def ")):
            problems.append(f"{q}: new_source must start with its def line")
            continue
        updated = "".join(lines[:span[0] - 1]) + body + "".join(lines[span[1]:])
        try:
            compile(updated, rel, "exec")
        except SyntaxError as err:
            problems.append(f"{q}: the new source does not compile ({err.msg}, line {err.lineno})")
            continue
        ws.originals.setdefault(rel, text)
        p.write_text(updated, encoding="utf-8")
    return problems


def _normalise(edits) -> list:
    """The edit list as a list of dicts, whatever JSON shape the call used (a JSON-encoded string, a single object)."""
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except ValueError:
            return [{"path": "", "function": "", "new_source": edits}]
    edits = edits if isinstance(edits, list) else [edits] if edits else []
    return [e if isinstance(e, dict) else {"path": "", "function": "", "new_source": str(e)} for e in edits]


def _reset(ws: Workspace) -> None:
    for rel in list(ws.originals):
        ws.revert_file(rel)


def _score(rep) -> tuple:
    """Lower is better: (ERROR?, landmarks off, worst error)."""
    if rep.verdict == "ERROR":
        return (1, 99, float("inf"))
    off = sum(1 for p in rep.phantoms for lm in p.landmarks if not lm.ok) + sum(p.orientation_ok is False for p in rep.phantoms)
    w = rep.worst()
    return (0, off, w[0] if w else 0.0)


def repair_staged(case: RealCase, client, *, n: int = 3, rounds: int = 3, k: int = 6, max_tokens: int = 8000,
                  max_run_tokens: int = 150_000, temperature: float = 0.7, log_path: Path | None = None, run_seed: int = 0,
                  diverse: bool = False) -> StagedResult:
    """``diverse`` shows the best-ranked function of each of the top files first (then fills by rank), so a fixed function ranked
    below several helpers of one hot file still gets shown."""
    from .llm import BudgetExceeded

    contract = load_from_source(case.script.read_text(encoding="utf-8"))
    log = Log(log_path)
    ws = Workspace(case)
    res = StagedResult(case=case.id, outcome="FAILED", upstream_files=fix_files(case))
    t0 = time.time()
    try:
        first = ws.oracle(contract)
        loc = ws.locate(contract)
        blocks = []
        ranked = loc.suspects[:k * 4]
        if diverse:
            firsts, seen = [], set()
            for s in ranked:
                if s.file not in seen:
                    firsts.append(s)
                    seen.add(s.file)
            ranked = firsts + [s for s in ranked if s not in firsts]
        for s in ranked:
            src = _source(ws, s.file, s.qualname)
            if not src or src.count("\n") > MAX_FUNC_LINES or len(blocks) >= k:
                continue
            tag = "returned the affine written to the output" if s.produced_final_affine else "returned an affine" if s.returned_affine else ""
            blocks.append(f"### {len(blocks) + 1}. {s.file} :: {s.qualname}  (ran in {s.fail_runs}/{loc.n_fail} wrong and "
                          f"{s.pass_runs}/{loc.n_pass} right runs{'; ' + tag if tag else ''})\n```python\n{src}\n```")
            res.shown.append([s.file, s.qualname])
        driver = case.script.read_text(encoding="utf-8")
        base = (f"Library: {PACKAGE[case.repo]} (private copy). Driver script:\n```python\n{driver}```\n\nSPATIAL_CONTRACT (fixed): "
                f"{json.dumps(contract.to_dict())}\n\nOracle report with the unmodified library:\n{first.feedback(max_lines=12)}\n\n"
                f"Most suspicious library functions (runtime trace on {loc.n_fail} wrong and {loc.n_pass} right phantoms):\n\n" + "\n\n".join(blocks))
        log.event("start", case=case.id, initial_verdict=first.verdict, shown=res.shown, upstream_files=res.upstream_files)
        history = []      # (score, rationale, edits, feedback)
        for rnd in range(1, rounds + 1):
            res.rounds = rnd
            prompt = base
            if history:
                best = sorted(history, key=lambda h: h[0])[:2]
                prompt += "\n\nEarlier candidates did not pass. The two closest, with what the oracle measured after applying them:\n\n" + "\n\n".join(
                    f"Candidate (rationale: {h[1]})\n" + "\n".join(f"- replaced {e.get('path')} :: {e.get('function')}:\n```python\n"
                                                                   f"{str(e.get('new_source', ''))[:2500]}\n```" for e in h[2])
                    + f"\nOracle after this candidate:\n{h[3]}" for h in best)
            for _ in range(n):
                if res.tokens["prompt"] + res.tokens["completion"] > max_run_tokens:
                    res.outcome, res.reasons = "BUDGET", ["per-run token cap reached"]
                    raise StopIteration
                try:
                    resp = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], max_tokens=max_tokens,
                                       tools=TOOL, tool_choice="auto", temperature=temperature)
                except BudgetExceeded as e:
                    res.outcome, res.reasons = "BUDGET", [str(e)]
                    raise StopIteration
                u = resp.usage
                if u:
                    res.tokens["prompt"] += u.prompt_tokens
                    res.tokens["completion"] += u.completion_tokens
                res.llm_calls += 1
                calls = list(resp.choices[0].message.tool_calls or [])
                if not calls:
                    log.event("no_tool_call", round=rnd)
                    continue
                try:
                    args = json.loads(calls[0].function.arguments)
                except ValueError:
                    log.event("bad_json", round=rnd)
                    continue
                edits, rationale = _normalise(args.get("edits")), args.get("rationale", "")
                res.candidates += 1
                _reset(ws)
                problems = apply_edits(ws, edits)
                if problems:
                    history.append(((2, 99, float("inf")), rationale, edits, "not applied: " + "; ".join(problems)))
                    log.event("candidate", round=rnd, rationale=rationale, rejected=problems)
                    continue
                res.tried_files = sorted(set(res.tried_files) | set(ws.touched()))
                rep = ws.oracle(contract)
                fb = rep.feedback(max_lines=10)
                log.event("candidate", round=rnd, rationale=rationale, verdict=rep.verdict, touched=ws.touched(), feedback=fb)
                if rep.passed:
                    res.visible_passed = True
                    raise StopIteration
                history.append((_score(rep), rationale, edits, fb))
        _reset(ws)
    except StopIteration:
        pass
    except Exception as e:                       # recorded, not hidden
        res.outcome, res.reasons = "ERROR", [f"{type(e).__name__}: {str(e)[:200]}"]
    try:
        res.touched, res.diff = ws.touched(), ws.diff()
        res.localized = bool((set(res.touched) | set(res.tried_files)) & set(res.upstream_files))
        if res.visible_passed:
            held = ws.validate(contract, run_seed=run_seed)
            res.heldout = held
            res.outcome = "REPAIRED" if held["passed"] else "OVERFIT"
        log.event("end", outcome=res.outcome, touched=res.touched, tokens=res.tokens, seconds=round(time.time() - t0, 1))
        return res
    finally:
        ws.cleanup()
