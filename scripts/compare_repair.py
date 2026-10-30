"""Compare every library-repair approach on the same real bugs, oracle and held-out validation. No tokens.

  python scripts/compare_repair.py       -> results/repair_comparison.json and a table on stdout
"Localized" counts runs that edited a file the upstream fix edited (for the staged runs: any candidate did). Cost uses
Nemotron 3 Super's listed prices ($0.30 / $0.90 per million prompt / completion tokens). Harness errors (a crash of our code,
not a verdict on the method) are listed and excluded from the run counts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("real3", "tool-calling agent, no trace"), ("real4", "tool-calling agent + trace ranking"),
        ("staged1", "staged: trace top-6 functions, 3 candidates x 3 rounds"),
        ("staged2", "staged, file-diverse suspects (top function of each file first, k=8)"),
        ("mut1", "mutation search over the trace's top suspects, no model")]
P_IN, P_OUT = 0.30e-6, 0.90e-6


def _gate_failures() -> dict:
    """run key 'run_id/case#rN' -> list of library gates it fails (results/regression_check.json, results/reader_check.json)."""
    out = {}
    for f, bad in (("regression_check.json", "REGRESSION"), ("reader_check.json", "READER_CHANGED")):
        path = ROOT / "results" / f
        if path.exists():
            for r in json.loads(path.read_text(encoding="utf-8")):
                if r["status"] == bad:
                    out.setdefault(r["run"], []).append(bad.lower())
    return out


def main() -> int:
    out, gates = {}, _gate_failures()
    for run_id, label in RUNS:
        f = ROOT / "results/real_repair" / run_id / "raw.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        if run_id.startswith("staged"):
            # staged1 recorded `localized` after resetting the workspace (always False for a failed run); recount from the logs:
            # did any applied candidate edit a file the upstream fix edited?
            for r in rows:
                log = ROOT / "results/real_repair" / run_id / "logs" / f"{r['case']}_r{r.get('rep', 0)}.jsonl"
                if log.exists():
                    ev = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
                    up = set(r.get("upstream_files") or [])
                    r["localized"] = any(set(e.get("touched") or []) & up for e in ev if e["event"] == "candidate")
        harness =[r for r in rows if r["outcome"] in ("ERROR", "HARNESS_ERROR")]
        rows = [r for r in rows if r not in harness]
        rep = [r for r in rows if r["outcome"] == "REPAIRED"]
        failed_gates = {f"{r['case']}#r{r.get('rep', 0)}": gates.get(f"{run_id}/{r['case']}#r{r.get('rep', 0)}") for r in rep}
        verified = [r for r in rep if not failed_gates[f"{r['case']}#r{r.get('rep', 0)}"]]
        tok = [r.get("tokens", {}) for r in rows]
        usd = sum(t.get("prompt", 0) * P_IN + t.get("completion", 0) * P_OUT for t in tok)
        out[run_id] = {"approach": label, "runs": len(rows), "bugs": len({r["case"] for r in rows}), "repaired_runs": len(rep),
                       "bugs_repaired": sorted({r["case"] for r in rep}), "overfit": sum(r["outcome"] == "OVERFIT" for r in rows),
                       "localized_runs": sum(bool(r.get("localized")) for r in rows),
                       "tokens": sum(t.get("prompt", 0) + t.get("completion", 0) for t in tok), "usd": round(usd, 3),
                       "usd_per_repair": round(usd / len(rep), 3) if rep else None,
                       "verified_runs": len(verified), "bugs_verified": sorted({r["case"] for r in verified}),
                       "failed_library_gates": {k: v for k, v in failed_gates.items() if v},
                       "usd_per_verified_repair": round(usd / len(verified), 3) if verified else None,
                       "harness_errors": [f"{r['case']}#r{r.get('rep')}" for r in harness]}
    (ROOT / "results/repair_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'run':8s} {'verified runs':>14s} {'bugs':>5s} {'(passed oracle)':>16s} {'localized':>10s} {'$':>6s} {'$/verified':>10s}  approach")
    for k, v in out.items():
        print(f"{k:8s} {v['verified_runs']:>6d} of {v['runs']:<6d} {len(v['bugs_verified']):>3d}/{v['bugs']:<2d} {v['repaired_runs']:>10d} runs "
              f"{v['localized_runs']:>5d}/{v['runs']:<4d} {v['usd']:>6.2f} {str(v['usd_per_verified_repair']):>10s}  {v['approach']}  "
              f"{v['bugs_verified']}" + (f"  gates failed: {v['failed_library_gates']}" if v["failed_library_gates"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
