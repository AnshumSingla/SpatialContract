"""Turn an agent run (JSONL log + the patched files it left behind) into a structured record the demo page renders.

The same function builds the recorded replays from the evaluation logs and the live view of a running job, so what the page
shows in replay mode is exactly what a live run would show. Every number in a record is either read from the run log or
re-measured with the deterministic oracle on the saved file; nothing is typed in by hand.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from . import oracle
from .contract import UNIT_TO_MM, load_from_source

VISIBLE_SEEDS = oracle.DEFAULT_SEEDS


def structured_report(path, contract, seeds=VISIBLE_SEEDS, python=None, pythonpath=None) -> dict:
    """Oracle report reduced to what the page draws: verdict, per-phantom landmark expected/got in mm, orientation info."""
    rep = oracle.check(path, contract, seeds=seeds, python=python, pythonpath=pythonpath)
    mm = UNIT_TO_MM.get(contract.units, 1.0) if contract and contract.units else 1.0
    out = {"verdict": rep.verdict, "reasons": rep.reasons, "frame": contract.world_frame if contract else None,
           "array_orientation": contract.array_orientation if contract else None, "phantoms": []}
    for p in rep.phantoms:
        lms = []
        for lm in p.landmarks:
            lms.append({"label": lm.label,
                        "expected": [round(v * mm, 2) for v in lm.expected] if lm.expected else None,
                        "got": [round(v * mm, 2) for v in lm.got] if lm.got else None,
                        "error_mm": None if lm.error_mm is None else round(lm.error_mm, 2), "ok": lm.ok})
        out["phantoms"].append({"seed": p.seed, "status": p.status, "in_axcodes": p.in_axcodes, "out_axcodes": p.out_axcodes,
                                "orientation_ok": p.orientation_ok, "tol_mm": p.tol_mm_used, "detail": p.detail, "landmarks": lms})
    w = rep.worst()
    out["worst_error_mm"] = None if w is None else (None if w[0] == float("inf") else round(w[0], 2))
    out["any_lost"] = bool(w and w[0] == float("inf"))
    out["feedback"] = rep.feedback(max_lines=10)
    return out


def unified_diff(a: str, b: str) -> list[str]:
    return [l.rstrip("\n") for l in difflib.unified_diff(a.splitlines(True), b.splitlines(True), "before", "after", n=2)][2:]


def read_events(log_path) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass            # a line still being written by a live job
    return out


class _Cache:
    def __init__(self):
        self.d = {}

    def report(self, path: Path, contract):
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self.d:
            self.d[key] = structured_report(path, contract)
        return self.d[key]


_cache = _Cache()


def build_record(original: str, log_path, work_dir, name: str, *, title: str = "", note: str = "") -> dict:
    """Parse one agent log (arm3 style) into a record. ``work_dir`` holds ``{name}_iter{n}.py`` written by the agent."""
    contract = load_from_source(original)
    work = Path(work_dir)
    events = read_events(log_path)
    rec = {"id": name, "title": title or name, "note": note, "source": original,
           "contract": contract.to_dict() if contract else None, "steps": [], "heldout": None, "outcome": None,
           "tokens_total": 0, "status": "running", "abstain_reasons": None}
    usable = contract is not None and not contract.problems()
    prev_tokens = 0
    prev_src = original
    pending: dict = {}
    for e in events:
        k = e["event"]
        if k == "abstain":
            rec.update(outcome="ABSTAIN", status="done", abstain_reasons=e.get("reasons"))
        elif k == "initial_oracle":
            first = work / f"{name}_iter0.py"
            if first.exists() and usable:
                rec["initial"] = _cache.report(first, contract)
        elif k == "llm_response":
            tok = e["tokens"]["prompt"] + e["tokens"]["completion"]
            pending = {"iteration": e["iteration"], "tokens": tok - prev_tokens, "finish": e.get("finish"),
                       "n_tool_calls": e.get("n_tool_calls", 0)}
            prev_tokens = tok
            rec["tokens_total"] = tok
        elif k == "patch":
            pending["rationale"] = e.get("rationale", "")
            pending["rejected"] = e.get("rejected")
            f = work / f"{name}_iter{e['iteration']}.py"
            if not e.get("rejected") and f.exists():
                new = f.read_text(encoding="utf-8")
                pending["diff"] = unified_diff(prev_src, new)
                pending["patched_source"] = new
                prev_src = new
        elif k == "oracle_after_patch":
            f = work / f"{name}_iter{e['iteration']}.py"
            if usable and f.exists():
                pending["oracle"] = _cache.report(f, contract)
            pending["verdict"] = e["verdict"]
        elif k == "no_patch":
            pending["no_patch"] = True
        elif k == "heldout":
            rec["heldout"] = {"passed": e["passed"], "summary": e["summary"], "per_run_seed": e.get("per_run_seed")}
        elif k in ("end", "budget_stop", "llm_error"):
            rec["status"] = "done"
        # a step is complete once its oracle verdict (or rejection / no patch) is in
        if pending and ("verdict" in pending or pending.get("rejected") or pending.get("no_patch")):
            rec["steps"].append(pending)
            pending = {}
    if rec["heldout"] is not None:
        rec["outcome"] = "REPAIRED" if rec["heldout"]["passed"] else "OVERFIT"
        rec["status"] = "done"
    elif rec["status"] == "done" and rec["outcome"] is None:
        rec["outcome"] = "FAILED"
    if rec.get("initial") and rec["initial"]["verdict"] == "PASS" and not rec["steps"]:
        rec.update(outcome="NO_CHANGE_NEEDED", status="done")
    rec["final_source"] = prev_src
    return rec


def repo_test_block(original: str, base: str, spec: dict | None, log_path=None, work_dir=None, name: str = "") -> dict:
    """What an ordinary unit-test suite says about the original, and (if a baseline-arm log is given) about the patch it accepted."""
    from . import repo_tests
    p = Path(work_dir) / f"{name}_iter0.py" if work_dir else None
    tmp = None
    if p is None or not p.exists():
        import tempfile
        d = tempfile.mkdtemp(prefix="sc_demo_")
        p = Path(d) / "orig.py"
        p.write_text(original, encoding="utf-8")
    res = repo_tests.run_repo_tests(p, base, spec or {})
    block = {"passed": res.passed, "text": res.text()}
    if log_path:
        ev = read_events(log_path)
        accepted = [e for e in ev if e["event"] == "accepted_by_repo_tests"]
        patches = [e for e in ev if e["event"] == "patch"]
        block["baseline_accepted"] = bool(accepted)
        if accepted and patches:
            f = Path(work_dir) / f"{name}_iter{patches[-1]['iteration']}.py"
            contract = load_from_source(original)
            if f.exists() and contract:
                block["baseline_diff"] = unified_diff(original, f.read_text(encoding="utf-8"))
                block["baseline_oracle"] = _cache.report(f, contract)
                block["baseline_rationale"] = patches[-1].get("rationale", "")
    return block


def build_library_record(case, run_dir, rep: int, title: str = "") -> dict:
    """One library-repair run (results/real_repair/<run>/logs/<case>_r<rep>.*) as a record. The pre-fix library is measured at build time, and the
    agent's saved diff is applied to a fresh private copy to measure its final state; nothing is typed in by hand."""
    import subprocess
    import tempfile

    from . import libagent
    from .realbugs import ENVS

    run_dir = Path(run_dir)
    log = read_events(run_dir / "logs" / f"{case.id}_r{rep}.jsonl")
    diff = (run_dir / "logs" / f"{case.id}_r{rep}.diff").read_text(encoding="utf-8")
    row = next(r for r in map(json.loads, (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()) if r["case"] == case.id and r["rep"] == rep)
    source = case.script.read_text(encoding="utf-8")
    contract = load_from_source(source)
    seeds = case.seeds or VISIBLE_SEEDS
    py = str(ENVS[case.env])
    rec = {"kind": "library", "id": case.id, "title": title or case.id, "description": case.summary, "source": source,
           "contract": contract.to_dict(), "outcome": row["outcome"], "tokens_total": row["tokens"]["prompt"] + row["tokens"]["completion"],
           "llm_calls": row["llm_calls"], "n_steps": row["steps"], "oracle_runs": row["oracle_runs"], "localized": row["localized"],
           "touched": row["touched"], "upstream_files": row["upstream_files"], "diff": diff, "heldout": None, "status": "done",
           "source_run": f"results/real_repair/{run_dir.name}/logs/{case.id}_r{rep}.jsonl", "model": row.get("model"), "steps": [], "tool_counts": row["tool_counts"]}
    ws = libagent.Workspace(case)
    try:
        rec["initial"] = structured_report(case.script, contract, seeds, py, ws.pythonpath)
        rec["upstream_diff_lines"] = None
        events = []
        for e in log:
            if e["event"] == "tool":
                a = e["args"]
                what = a.get("path") or a.get("pattern") or a.get("reason") or ""
                if a.get("start") is not None and e["name"] == "read_file":
                    what += f" [{a.get('start')}-{a.get('end', '')}]"
                if e["name"] == "replace_lines":
                    what += f" [{a.get('start_line')}-{a.get('end_line')}]"
                events.append({"step": e["step"], "name": e["name"], "what": what[:110], "result": e["result"][:140].replace(chr(10), " | ")})
            elif e["event"] == "oracle":
                events.append({"step": e["step"], "name": "run_oracle", "what": "", "result": f"{e['verdict']}", "verdict": e["verdict"]})
        rec["events"] = events
        for e in log:
            if e["event"] == "heldout":
                rec["heldout"] = {"passed": e["passed"], "summary": e["summary"]}
        if diff.strip():
            d = Path(tempfile.mkdtemp(prefix="sc_demo_apply_")) / "x.diff"
            d.write_text(diff, encoding="utf-8")
            r = subprocess.run(["git", "apply", "-p1", "--whitespace=nowarn", str(d)], cwd=ws.lib, capture_output=True, text=True)
            if r.returncode == 0:
                rec["final"] = structured_report(case.script, contract, seeds, py, ws.pythonpath)
    finally:
        ws.cleanup()
    return rec
