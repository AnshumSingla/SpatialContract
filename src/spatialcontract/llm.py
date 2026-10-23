"""Nebius Token Factory client with a hard, persistent cumulative token cap.

The cap is enforced BEFORE each call (prompt estimate + max_tokens must fit) and the real
usage returned by the API is added to a ledger afterwards, so the total survives restarts.
The API key is read from the environment / .env and is never printed or logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results" / "token_ledger.json"
load_dotenv(ROOT / ".env")


class BudgetExceeded(RuntimeError):
    pass


class TokenBudget:
    def __init__(self, cap: int | None = None, ledger: Path = LEDGER):
        self.cap = int(cap if cap is not None else os.environ.get("SC_TOKEN_CAP", "200000"))
        self.ledger = ledger
        self.total = self._load()

    def _load(self) -> int:
        try:
            return int(json.loads(self.ledger.read_text())["total_tokens"])
        except (OSError, ValueError, KeyError):
            return 0

    def check(self, estimated: int) -> None:
        if self.total + estimated > self.cap:
            raise BudgetExceeded(
                f"token cap {self.cap} would be exceeded: used {self.total}, next call up to {estimated}"
            )

    def record(self, prompt: int, completion: int) -> None:
        self.total += prompt + completion
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(json.dumps({"total_tokens": self.total}))
        print(f"[budget] +{prompt}p +{completion}c => {self.total}/{self.cap} tokens")


def settings() -> dict:
    key = os.environ.get("NEBIUS_API_KEY", "")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set (put it in .env)")
    return {
        "api_key": key,
        "base_url": os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
        "model": os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
    }


class Client:
    def __init__(self, budget: TokenBudget | None = None, model: str | None = None):
        from openai import OpenAI

        s = settings()
        self.model = model or s["model"]
        self.budget = budget or TokenBudget()
        self._c = OpenAI(api_key=s["api_key"], base_url=s["base_url"], timeout=90)

    def chat(self, messages: list[dict], max_tokens: int = 256, **kw):
        est_prompt = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 50
        if "tools" in kw:
            est_prompt += len(json.dumps(kw["tools"])) // 3
        self.budget.check(est_prompt + max_tokens)
        resp = self._c.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens, **kw
        )
        u = resp.usage
        self.budget.record(u.prompt_tokens if u else est_prompt, u.completion_tokens if u else max_tokens)
        return resp
