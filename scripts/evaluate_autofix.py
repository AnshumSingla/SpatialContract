"""Deterministic-first repair on the seeded corpus, no tokens: which bugs does autofix.py repair without a model, and what would a
"deterministic first, model only when needed" pipeline have spent? Every autofix patch goes through the same patch guard, visible
oracle and held-out validation as a model patch. Controls and must-abstain cases are run too (the gate and the PASS check come
first, so autofix must never touch them).

  python scripts/evaluate_autofix.py      -> results/eval/autofix.json"""
import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import agent, autofix, evaluation as ev, heldout, oracle  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402


def run(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    contract = load_from_source(src)
    if contract is None or contract.problems():
        return {"outcome": "ABSTAIN"}
    rep = oracle.check(path, contract)
    if rep.passed:
        return {"outcome": "NO_CHANGE_NEEDED"}
    corr = autofix.consistent_map(rep, contract)
    if corr is None:
        return {"outcome": "NOT_APPLICABLE", "patterns": sorted({(p.pattern or {}).get("kind", "-") for p in rep.phantoms})}
    args = [a.arg for a in next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "process").args.args]
    patched = autofix.wrapper_source(src, corr, contract, args)
    problems = agent.check_patch(src, patched)
    if problems:
        return {"outcome": "GUARD_REJECTED", "problems": problems}
    with tempfile.TemporaryDirectory(prefix="sc_autofix_") as d:
        p = Path(d) / path.name
        p.write_text(patched, encoding="utf-8")
        vis = oracle.check(p, contract)
        if not vis.passed:
            return {"outcome": "VISIBLE_FAIL", "feedback": vis.feedback(max_lines=4)}
        held = heldout.validate(p, contract, run_seed=0)
    return {"outcome": "REPAIRED" if held["passed"] else "OVERFIT", "heldout": held["summary"],
            "correction": "orient" if isinstance(corr, str) else "affine"}


def main() -> int:
    manifest = ev.load_manifest()
    rows = []
    for m in manifest:
        r = {"id": m["id"], "kind": m["kind"], "bug_class": m.get("bug_class"), **run(ev.CORPUS / m["file"])}
        rows.append(r)
        print(f"{m['id']:28s} {m['kind']:8s} {r['outcome']:16s} {r.get('patterns', '')}{r.get('heldout', '')}", flush=True)
    a3 = [json.loads(l) for l in (ROOT / "results/eval/main/raw.jsonl").read_text(encoding="utf-8").splitlines()]
    a3 = [r for r in a3 if r["arm"] == "arm3" and r["kind"] == "bug"]
    auto = {r["id"] for r in rows if r["outcome"] == "REPAIRED"}
    bugs = [m["id"] for m in manifest if m["kind"] == "bug"]
    rep_rate = lambda ids_auto: sum(1.0 if i in ids_auto else ev.per_case(a3, "arm3", "repaired", {i}).get(i, 0) for i in bugs) / len(bugs)  # noqa: E731
    tok_all = sum(r["tokens"] for r in a3)
    tok_hybrid = sum(r["tokens"] for r in a3 if r["id"] not in auto)
    summary = {"bugs": len(bugs), "auto_repaired": len(auto), "auto_overfit": sum(r["outcome"] == "OVERFIT" for r in rows),
               "auto_ids": sorted(auto), "not_applicable": sum(r["outcome"] == "NOT_APPLICABLE" for r in rows if r["kind"] == "bug"),
               "controls_touched": sum(r["outcome"] not in ("NO_CHANGE_NEEDED",) for r in rows if r["kind"] == "control"),
               "abstain_touched": sum(r["outcome"] != "ABSTAIN" for r in rows if r["kind"] == "abstain"),
               "hybrid_repaired_pct": round(100 * rep_rate(auto)), "arm3_repaired_pct": round(100 * rep_rate(set())),
               "arm3_tokens_on_bugs": tok_all, "hybrid_tokens_on_bugs": tok_hybrid}
    (ROOT / "results/eval/autofix.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
