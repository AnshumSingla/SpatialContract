"""Staged repair of the real library bugs (staged.py: trace -> whole-function candidates -> oracle picks). Spends tokens: needs
SC_ALLOW_API=1, a token cap and preferably a dollar cap.

  SC_ALLOW_API=1 SC_TOKEN_CAP=... SC_USD_CAP=... python scripts/repair_staged.py --run-id staged1 [--cases ...] [--reps 1]
Resumable (rows already in results/real_repair/<run-id>/raw.jsonl are skipped). Same cases, oracle, held-out validation and
result fields as scripts/repair_real.py, so the two can be compared directly."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import realbugs, staged  # noqa: E402
from spatialcontract.llm import Client, TokenBudget, require_api_enabled  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--budget", type=int, default=1_500_000)
    ap.add_argument("--n", type=int, default=3, help="candidates per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=6, help="suspect functions shown")
    ap.add_argument("--max-run-tokens", type=int, default=150_000)
    ap.add_argument("--diverse", action="store_true", help="show the best function of each top file first")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--run-id", default=time.strftime("staged-%Y%m%d-%H%M%S"))
    a = ap.parse_args()

    require_api_enabled()
    base = TokenBudget()
    client = Client(budget=TokenBudget(cap=min(base.cap, base.total + a.budget)))
    out = ROOT / "results" / "real_repair" / a.run_id
    (out / "logs").mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = {(r["case"], r["rep"]) for r in map(json.loads, raw.read_text(encoding="utf-8").splitlines())} if raw.exists() else set()
    tasks = [(c, r) for r in range(a.reps) for c in realbugs.CASES if (not a.cases or c.id in a.cases) and (c.id, r) not in done]
    print("model:", client.model, "| tasks:", len(tasks), f"| n={a.n} rounds={a.rounds} k={a.k} diverse={a.diverse}", flush=True)

    def work(t):
        case, rep = t
        t0 = time.time()
        try:
            res = staged.repair_staged(case, client, n=a.n, rounds=a.rounds, k=a.k, max_run_tokens=a.max_run_tokens, diverse=a.diverse,
                                       log_path=out / "logs" / f"{case.id}_r{rep}.jsonl", run_seed=rep)
            row = res.to_dict() | {"rep": rep, "seconds": round(time.time() - t0, 1), "model": client.model}
            (out / "logs" / f"{case.id}_r{rep}.diff").write_text(res.diff, encoding="utf-8")
        except Exception as e:
            row = {"case": case.id, "rep": rep, "outcome": "HARNESS_ERROR", "reasons": [f"{type(e).__name__}: {str(e)[:300]}"]}
        with raw.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        tok = row.get("tokens", {})
        print(f"{row['case']:24s} r{rep} {row['outcome']:9s} rounds={row.get('rounds')} candidates={row.get('candidates')} "
              f"localized={row.get('localized')} tokens={tok.get('prompt', 0) + tok.get('completion', 0)} {row.get('seconds')}s", flush=True)
        return row

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(work, tasks))
    print("ledger total:", client.budget.total, f"| about ${client.budget.usd:.2f} spent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
