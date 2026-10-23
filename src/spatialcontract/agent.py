"""Contract-verified repair agent.

Flow: gate (ABSTAIN unless the pipeline declares a usable contract) -> oracle on visible phantoms -> if it fails, ask
Nemotron (tool call ``submit_patch``) for a full-file patch -> guard the patch -> re-run the oracle -> feed the
measurements back -> repeat up to ``max_iters``. A patch that passes the visible oracle is then checked on the
held-out bank (``heldout.validate``) and labelled REPAIRED or OVERFIT. Every step is written to a JSONL log.

The model never sees the held-out phantoms, only visible-oracle measurements. It cannot edit the contract.
"""
from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import heldout, oracle
from .contract import SpatialContract, load_from_source

ALLOWED_IMPORTS = {"numpy", "nibabel", "scipy", "SimpleITK", "math", "itertools", "functools", "typing", "collections",
                   "__future__"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "breakpoint"}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "submit_patch",
        "description": "Submit the COMPLETE patched Python file. It is applied in a sandbox and the oracle is re-run; "
                       "the measurements come back as the tool result.",
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string", "description": "One or two sentences: root cause and what you changed."},
                "patched_source": {"type": "string", "description": "The entire patched file contents."},
            },
            "required": ["rationale", "patched_source"],
        },
    },
}]

SYSTEM_PROMPT = """You repair spatial / orientation / affine bugs in medical-imaging Python scripts (nibabel, numpy, scipy, SimpleITK).
Each script defines process(in_path, out_path): it reads a NIfTI volume and writes a NIfTI volume. A module-level SPATIAL_CONTRACT
dict declares what the physical coordinates of the OUTPUT must mean. A deterministic oracle pushes asymmetric landmark phantoms
(random input orientation, voxel size, obliquity and origin) through the script and reports, per landmark, where it landed versus
where the contract says it must land.

Conventions: a NIfTI affine maps voxel index (voxel centre) to world coordinates. RAS+ means +x Right, +y Anterior, +z Superior;
LPS+ is the opposite in x and y. The units of the affine's world coordinates must match the contract's units
(mm, or m = 1/1000 of a mm). "array_orientation", when present, means the output voxel axes (i, j, k) must run toward those
patient directions (nibabel.aff2axcodes of the output affine, expressed in RAS).

Rules:
1. Reply by calling submit_patch with the COMPLETE patched file.
2. Do not change SPATIAL_CONTRACT or the signature of process. Never edit the contract to make a failure go away.
3. Fix the root cause in general. Do not special-case particular inputs, shapes, orientations or voxel sizes: hidden phantoms
   with other orientations, voxel sizes and obliquity, and a non-phantom image, are checked after you pass.
4. Keep the change minimal. Only numpy, nibabel, scipy and SimpleITK may be imported; no file or network access besides the two paths.
"""


SYSTEM_PROMPT_TESTS = """You repair spatial / orientation / affine bugs in medical-imaging Python scripts (nibabel, numpy, scipy, SimpleITK).
Each script defines process(in_path, out_path): it reads a NIfTI volume and writes a NIfTI volume. A module-level SPATIAL_CONTRACT
dict, when present, declares what the physical coordinates of the OUTPUT must mean.

Conventions: a NIfTI affine maps voxel index (voxel centre) to world coordinates. RAS+ means +x Right, +y Anterior, +z Superior;
LPS+ is the opposite in x and y. The units of the affine's world coordinates must match the contract's units
(mm, or m = 1/1000 of a mm). "array_orientation", when present, means the output voxel axes (i, j, k) must run toward those
patient directions (nibabel.aff2axcodes of the output affine, expressed in RAS).

The repository's unit tests are run after each patch and their results are returned to you; a patch is accepted as soon as they pass.
The tests are ordinary unit tests and may not cover everything: hidden checks of the output's physical coordinates are applied afterwards.

Rules:
1. To change the script, call submit_patch with the COMPLETE patched file.
2. If you believe the script already satisfies its contract, call report_no_bug. If the script declares no usable SPATIAL_CONTRACT
   (missing, unknown or ambiguous fields) so that the intended convention cannot be determined, call abstain: never guess a convention.
3. Do not change SPATIAL_CONTRACT or the signature of process.
4. Fix the root cause in general. Do not special-case particular inputs, shapes, orientations or voxel sizes.
5. Keep the change minimal. Only numpy, nibabel, scipy and SimpleITK may be imported; no file or network access besides the two paths.
"""

TOOLS_TESTS = TOOLS + [
    {"type": "function", "function": {
        "name": "report_no_bug",
        "description": "Declare that the script already satisfies its contract; nothing is changed.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
    {"type": "function", "function": {
        "name": "abstain",
        "description": "Decline to patch because the intended spatial convention cannot be determined (no usable contract).",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


@dataclass
class RepairResult:
    case: str
    outcome: str                     # ABSTAIN | NO_CHANGE_NEEDED | REPAIRED | OVERFIT | FAILED | BUDGET | ERROR
    iterations: int = 0
    llm_calls: int = 0
    tokens: dict = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    visible_passed: bool = False
    heldout: dict | None = None
    reasons: list = field(default_factory=list)
    final_source: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("final_source")
        return d


# ----------------------------------------------------------------------------- patch guards
def check_patch(original: str, patched: str) -> list[str]:
    """Reasons to reject a patch before running it. Empty list => may run."""
    try:
        tree = ast.parse(patched)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from '{node.module}' not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"call to '{node.func.id}' not allowed")
    procs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process"]
    if not procs:
        problems.append("no top-level function 'process'")
    else:
        want = [a.arg for a in next(n for n in ast.parse(original).body
                                    if isinstance(n, ast.FunctionDef) and n.name == "process").args.args]
        if [a.arg for a in procs[0].args.args] != want:
            problems.append(f"process signature changed (expected arguments {want})")
    old_c, new_c = load_from_source(original), load_from_source(patched)
    if (old_c and old_c.to_dict()) != (new_c and new_c.to_dict()):
        problems.append("SPATIAL_CONTRACT was changed, added or removed; the contract is fixed")
    return problems


def extract_code_block(text: str | None) -> str | None:
    if not text:
        return None
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if "def process" in b]
    return blocks[-1] if blocks else None


# ----------------------------------------------------------------------------- the loop
class Log:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def event(self, kind: str, **data):
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time(), 2), "event": kind, **data}, default=str) + "\n")


def oracle_evaluator(contract: SpatialContract, seeds=oracle.DEFAULT_SEEDS) -> Callable:
    def evaluate(code_path):
        rep = oracle.check(code_path, contract, seeds=seeds)
        return rep.passed, rep.feedback(max_lines=12), rep.verdict
    return evaluate


def _usage(resp, result: RepairResult):
    u = getattr(resp, "usage", None)
    if u:
        result.tokens["prompt"] += u.prompt_tokens
        result.tokens["completion"] += u.completion_tokens
    result.llm_calls += 1


def repair(case_path, client, *, max_iters: int = 4, run_seed: int = 0, max_tokens: int = 6000, log_dir=None,
           workdir=None, evaluate_factory: Callable = oracle_evaluator, gate: bool = True, validate: bool = True,
           llm_kwargs: dict | None = None, arm: str = "oracle") -> RepairResult:
    """arm="oracle": gate + oracle feedback (arm 3). arm="tests": no gate, repo-test feedback, report_no_bug/abstain tools (arm 2).
    ``evaluate_factory(contract)`` must return ``evaluate(path) -> (done, text, verdict)`` for the chosen arm."""
    case_path = Path(case_path)
    original = case_path.read_text(encoding="utf-8")
    name = case_path.stem
    log = Log(Path(log_dir) / f"{name}.jsonl" if log_dir else None)
    res = RepairResult(case=name, outcome="ERROR", log_path=str(log.path) if log.path else None)
    contract = load_from_source(original)

    if gate and (contract is None or contract.problems()):
        res.outcome = "ABSTAIN"
        res.reasons = ["no SPATIAL_CONTRACT declared"] if contract is None else contract.problems()
        log.event("abstain", reasons=res.reasons)
        return res
    if contract is None:
        contract = SpatialContract()
    evaluate = evaluate_factory(contract)
    work = Path(workdir) if workdir else (Path(log_dir) / f"{name}_work" if log_dir else Path(case_path).parent / "_agent_work")
    work.mkdir(parents=True, exist_ok=True)
    cur_path = work / f"{name}_iter0.py"
    cur_path.write_text(original, encoding="utf-8")

    done, text, verdict = evaluate(cur_path)
    log.event("initial_oracle" if arm == "oracle" else "initial_repo_tests", verdict=verdict, feedback=text)
    if done and arm == "oracle":
        res.outcome, res.visible_passed, res.final_source = "NO_CHANGE_NEEDED", True, original
        return res

    if arm == "oracle":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "This script is reported to produce wrongly placed or oriented output. Fix it.\n\n"
                f"SPATIAL_CONTRACT (fixed, do not edit): {json.dumps(contract.to_dict())}\n\n"
                f"Script:\n```python\n{original}```\n\nOracle report on the original script:\n{text}")},
        ]
    else:
        declared = load_from_source(original)
        contract_line = ("No SPATIAL_CONTRACT is declared in this script." if declared is None
                         else f"SPATIAL_CONTRACT (fixed, do not edit): {json.dumps(declared.to_dict())}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TESTS},
            {"role": "user", "content": (
                "This script is reported to produce wrongly placed or oriented output. Fix it, or say if there is nothing to fix.\n\n"
                f"{contract_line}\n\nScript:\n```python\n{original}```\n\nRepo test results on the original script:\n{text}")},
        ]
    tools = TOOLS if arm == "oracle" else TOOLS_TESTS
    best_source = original
    from .llm import BudgetExceeded

    for it in range(1, max_iters + 1):
        res.iterations = it
        try:
            resp = client.chat(messages, max_tokens=max_tokens, tools=tools, tool_choice="auto", **(llm_kwargs or {}))
        except BudgetExceeded as e:
            res.outcome, res.reasons = "BUDGET", [str(e)]
            log.event("budget_stop", detail=str(e))
            return res
        except Exception as e:
            res.outcome, res.reasons = "ERROR", [f"{type(e).__name__}: {str(e)[:200]}"]
            log.event("llm_error", detail=res.reasons[0])
            return res
        _usage(resp, res)
        msg = resp.choices[0].message
        calls = list(getattr(msg, "tool_calls", None) or [])
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        log.event("llm_response", iteration=it, finish=resp.choices[0].finish_reason, n_tool_calls=len(calls),
                  content=(msg.content or "")[:2000], reasoning_chars=len(reasoning or ""), tokens=dict(res.tokens))

        patched, rationale, call_id = None, "", None
        feedback_text = "submit_patch needs a non-empty patched_source."
        if calls and calls[0].function.name in ("report_no_bug", "abstain"):
            try:
                reason = json.loads(calls[0].function.arguments).get("reason", "")
            except (ValueError, AttributeError):
                reason = ""
            res.outcome = "NO_CHANGE_CLAIMED" if calls[0].function.name == "report_no_bug" else "ABSTAIN"
            res.reasons, res.final_source = [reason], best_source
            log.event(calls[0].function.name, iteration=it, reason=reason)
            return res
        if calls:
            call_id = calls[0].id
            try:
                args = json.loads(calls[0].function.arguments)
                patched, rationale = args.get("patched_source"), args.get("rationale", "")
            except (ValueError, AttributeError) as e:
                feedback_text = f"Your tool call arguments were not valid JSON ({e}). Call submit_patch again."
        else:
            patched = extract_code_block(msg.content)
            feedback_text = "You must call submit_patch with the complete patched file."

        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls[:1]]
        messages.append(assistant)

        if patched:
            problems = check_patch(original, patched)
            log.event("patch", iteration=it, rationale=rationale, rejected=problems or None, chars=len(patched))
            if problems:
                feedback_text = "Patch rejected before running: " + "; ".join(problems)
            else:
                cur_path = work / f"{name}_iter{it}.py"
                cur_path.write_text(patched, encoding="utf-8")
                done, text, verdict = evaluate(cur_path)
                log.event("oracle_after_patch", iteration=it, verdict=verdict, feedback=text)
                best_source = patched
                if done and arm == "tests":
                    res.outcome, res.final_source, res.visible_passed = "ACCEPTED_BY_TESTS", patched, True
                    log.event("accepted_by_repo_tests", iteration=it)
                    return res
                if done:
                    res.visible_passed, res.final_source = True, patched
                    if not validate:
                        res.outcome = "REPAIRED"
                        return res
                    held = heldout.validate(cur_path, contract, run_seed=run_seed)
                    res.heldout = held
                    log.event("heldout", **{k: held[k] for k in ("passed", "summary", "per_run_seed")})
                    res.outcome = "REPAIRED" if held["passed"] else "OVERFIT"
                    return res
                feedback_text = (f"Oracle verdict after your patch:\n{text}" if arm == "oracle" else f"Repo test results after your patch:\n{text}")
        elif not calls:
            log.event("no_patch", iteration=it)
        if call_id:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": feedback_text})
        else:
            messages.append({"role": "user", "content": feedback_text})

    res.outcome, res.final_source = "FAILED", best_source
    res.reasons = ["iteration cap reached without a passing patch"]
    log.event("end", outcome=res.outcome)
    return res
