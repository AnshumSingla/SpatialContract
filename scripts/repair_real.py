"""Repair the real library bugs with the oracle-guided agent. Spends tokens: needs SC_ALLOW_API=1 and a budget.

  SC_ALLOW_API=1 SC_TOKEN_CAP=4000000 python scripts/repair_real.py --reps 1 --budget 1500000 --run-id real1
Resumable (rows already in results/real_repair/<run-id>/raw.jsonl are skipped). Needs the worktrees and environments from
scripts/setup_real_bugs.py (see README). Each run edits a private copy of the library; nothing under _work/wt is modified."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import libagent, realbugs  # noqa: E402
from spatialcontract.llm import Client, TokenBudget, require_api_enabled  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="case ids (default: all 7)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--budget", type=int, default=1_500_000)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--max-oracle-runs", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()

    require_api_enabled()
    base = TokenBudget()
    client = Client(budget=TokenBudget(cap=min(base.cap, base.total + a.budget)), model=a.model)
    out = ROOT / "results" / "real_repair" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = {(r["case"], r["rep"]) for r in map(json.loads, raw.read_text(encoding="utf-8").splitlines())} if raw.exists() else set()
    cases = [c for c in realbugs.CASES if not a.cases or c.id in a.cases]
    tasks = [(c, r) for r in range(a.reps) for c in cases if (c.id, r) not in done]
    print("model:", client.model, "| tasks:", len(tasks), flush=True)

    def work(t):
        case, rep = t
        t0 = time.time()
        try:
            res = libagent.repair_library(case, client, max_steps=a.max_steps, max_oracle_runs=a.max_oracle_runs, max_tokens=a.max_tokens,
                                          log_path=out / "logs" / f"{case.id}_r{rep}.jsonl", run_seed=rep)
            row = res.to_dict() | {"rep": rep, "seconds": round(time.time() - t0, 1), "model": client.model}
            (out / "logs").mkdir(exist_ok=True)
            (out / "logs" / f"{case.id}_r{rep}.diff").write_text(res.diff, encoding="utf-8")
        except Exception as e:                       # recorded, not hidden
            row = {"case": case.id, "rep": rep, "outcome": "HARNESS_ERROR", "reasons": [f"{type(e).__name__}: {str(e)[:300]}"]}
        with raw.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        tok = row.get("tokens", {})
        print(f"{row['case']:24s} r{rep} {row['outcome']:14s} steps={row.get('steps')} oracle_runs={row.get('oracle_runs')} "
              f"localized={row.get('localized')} touched={row.get('touched')} tokens={tok.get('prompt', 0) + tok.get('completion', 0)} "
              f"{row.get('seconds')}s", flush=True)
        return row

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(work, tasks))
    print("ledger total:", client.budget.total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
