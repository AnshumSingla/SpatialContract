"""Three-arm evaluation on the seeded corpus. Arms 2 and 3 call Token Factory: needs SC_ALLOW_API=1 and a budget.

  python scripts/evaluate.py --arms arm1 --run-id main                       # no tokens
  SC_ALLOW_API=1 SC_TOKEN_CAP=1500000 python scripts/evaluate.py --arms arm2 arm3 --reps 3 --budget 1200000 --run-id main
Resumable: rows already in results/eval/<run-id>/raw.jsonl are skipped. Use --ids to restrict to a few cases (dry runs)."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import evaluation  # noqa: E402
from spatialcontract.llm import Client, TokenBudget, require_api_enabled  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["arm1", "arm2", "arm3"], choices=["arm1", "arm2", "arm3"])
    ap.add_argument("--reps", type=int, default=3, help="repetitions for the LLM arms (arm1 is deterministic: 1)")
    ap.add_argument("--ids", nargs="*", help="restrict to these case ids")
    ap.add_argument("--kinds", nargs="+", default=["bug", "control", "abstain"])
    ap.add_argument("--budget", type=int, default=1_200_000, help="max tokens this invocation may add to the ledger")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()

    client = None
    if any(x in ("arm2", "arm3") for x in a.arms):
        require_api_enabled()
        base = TokenBudget()
        client = Client(budget=TokenBudget(cap=min(base.cap, base.total + a.budget)), model=a.model)
        print("model:", client.model)
    out = ROOT / "results" / "eval" / a.run_id
    rows = evaluation.run_evaluation(a.arms, a.reps, client, out, kinds=tuple(a.kinds), only_ids=set(a.ids) if a.ids else None,
                                     max_iters=a.max_iters, max_tokens=a.max_tokens, parallel=a.parallel)
    print(f"{len(rows)} rows in {out / 'raw.jsonl'}")
    if client:
        print("ledger total:", client.budget.total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
