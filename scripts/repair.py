"""Run the repair agent on one or more scripts. Makes real Token Factory calls: needs SC_ALLOW_API=1 and a budget.

  SC_ALLOW_API=1 python scripts/repair.py corpus/seeded/R05_stale_affine.py --budget 40000
Logs go to results/runs/<run-id>/<case>.jsonl; the per-run budget is added on top of the persistent ledger total."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import agent  # noqa: E402
from spatialcontract.llm import Client, TokenBudget, require_api_enabled  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--model", default=None, help="default: NEBIUS_MODEL from .env (nemotron-3-super-120b-a12b)")
    ap.add_argument("--budget", type=int, default=40000, help="max tokens this invocation may add to the ledger")
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=6000, help="per-call completion cap (reasoning tokens count)")
    ap.add_argument("--thinking", choices=["default", "low", "off"], default="default",
                    help="Nemotron reasoning: off = chat_template_kwargs.enable_thinking=false; low = reasoning_effort=low")
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--no-autofix", action="store_true", help="always call the model (skip the deterministic correction tried first)")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()

    require_api_enabled()
    base = TokenBudget()
    budget = TokenBudget(cap=min(base.cap, base.total + a.budget))
    client = Client(budget=budget, model=a.model)   # model=None -> NEBIUS_MODEL
    llm_kwargs = {"off": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                  "low": {"extra_body": {"reasoning_effort": "low"}}, "default": {}}[a.thinking]
    out_dir = ROOT / "results" / "runs" / a.run_id
    summary = []
    for case in a.cases:
        t = time.time()
        r = agent.repair(case, client, max_iters=a.max_iters, run_seed=a.run_seed, max_tokens=a.max_tokens, log_dir=out_dir,
                         llm_kwargs=llm_kwargs, autofix=not a.no_autofix)
        d = r.to_dict() | {"seconds": round(time.time() - t, 1), "model": client.model, "thinking": a.thinking}
        summary.append(d)
        print(f"{r.case:28s} {r.outcome:16s} iters={r.iterations} calls={r.llm_calls} tokens={r.tokens} "
              f"heldout={None if not r.heldout else r.heldout['summary']} {d['seconds']}s", flush=True)
        if r.outcome == "BUDGET":
            break
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"ledger total: {budget.total} tokens; logs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
