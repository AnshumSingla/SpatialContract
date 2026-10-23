"""Step 2: budget-capped completion + tool-calling smoke test against Nebius Token Factory.

Spends at most --budget tokens in total (default 8000) on top of whatever the ledger already holds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract.llm import BudgetExceeded, Client, TokenBudget  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    base = TokenBudget()
    budget = TokenBudget(cap=min(base.cap, base.total + a.budget))
    client = Client(budget=budget, model=a.model)
    res = {"model": a.model, "calls": []}

    try:
        t = time.time()
        r = client.chat([{"role": "user", "content": "Reply with exactly one short sentence saying hello."}],
                        max_tokens=a.max_tokens)
        m = r.choices[0].message
        res["calls"].append({
            "kind": "chat", "latency_s": round(time.time() - t, 2), "finish": r.choices[0].finish_reason,
            "content": m.content, "has_reasoning_field": bool(getattr(m, "reasoning_content", None)
                                                              or getattr(m, "reasoning", None)),
            "usage": r.usage.model_dump() if r.usage else None})

        t = time.time()
        r = client.chat([{"role": "user", "content": "What is 1234 plus 4321? Use the add tool."}],
                        max_tokens=a.max_tokens, tools=TOOLS, tool_choice="auto")
        m = r.choices[0].message
        tc = [{"name": c.function.name, "arguments": c.function.arguments} for c in (m.tool_calls or [])]
        valid = False
        if tc:
            try:
                args = json.loads(tc[0]["arguments"])
                valid = tc[0]["name"] == "add" and args.get("a") == 1234 and args.get("b") == 4321
            except ValueError:
                pass
        res["calls"].append({"kind": "tool_call", "latency_s": round(time.time() - t, 2),
                             "finish": r.choices[0].finish_reason, "tool_calls": tc, "tool_call_valid": valid,
                             "content": m.content, "usage": r.usage.model_dump() if r.usage else None})
    except BudgetExceeded as e:
        res["error"] = f"BudgetExceeded: {e}"
    except Exception as e:  # report plainly, never include secrets
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    res["ledger_total_tokens"] = budget.total
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"smoke_{a.model.split('/')[-1]}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 1 if "error" in res else 0


if __name__ == "__main__":
    raise SystemExit(main())
