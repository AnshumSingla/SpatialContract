"""Nebius Token Factory client with a hard, persistent cumulative token cap, a dollar cap and a credits stop.

The cap is enforced BEFORE each call (prompt estimate + max_tokens must fit) and the real
usage returned by the API is added to a ledger afterwards, so the total survives restarts.
``SC_USD_CAP`` adds a cap on estimated spend (listed prices, see PRICES). The first error that says the account is out
of credits or quota writes ``results/api_stopped.json``; from then on no call is made until that file is deleted, so a
long run cannot keep hitting an exhausted account. The API key is read from the environment / .env and is never printed
or logged.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results" / "token_ledger.json"
STOP_FILE = ROOT / "results" / "api_stopped.json"
load_dotenv(ROOT / ".env")

# USD per token (prompt, completion), from GET /models?verbose=true; unknown models are priced like Super
PRICES = {"nvidia/nemotron-3-super-120b-a12b": (0.30e-6, 0.90e-6), "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06e-6, 0.24e-6)}
DEFAULT_PRICE = PRICES["nvidia/nemotron-3-super-120b-a12b"]
# Spend before the ledger split prompt from completion tokens (12.43M tokens, 2026-10-23..29), priced conservatively:
# almost all of it was prompt tokens of Super; 10% of it is counted at the completion price.
USD_BEFORE_SPLIT = round(12_431_110 * (0.9 * 0.30e-6 + 0.1 * 0.90e-6), 2)
_CREDIT_WORDS = ("credit", "balance", "insufficient", "billing", "payment", "funds", "quota", "top up", "top-up")


class BudgetExceeded(RuntimeError):
    pass


class CreditsExhausted(BudgetExceeded):
    """The account is out of credits (or a stop was recorded earlier): no call until results/api_stopped.json is removed."""


def is_out_of_credits(exc: BaseException) -> bool:
    """True for an API error that means "no money or quota left", as opposed to a transient rate limit or a bad request."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 402:
        return True
    return status in (401, 403, 429) and any(w in text for w in _CREDIT_WORDS) and "rate limit" not in text


def api_stopped() -> dict | None:
    try:
        return json.loads(STOP_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class TokenBudget:
    """Cumulative token cap, safe to share between threads. ``check`` reserves the worst case of an in-flight call so
    parallel workers cannot jointly overshoot the cap; ``record`` releases the reservation and adds the real usage."""

    def __init__(self, cap: int | None = None, ledger: Path = LEDGER, usd_cap: float | None = None):
        self.cap = int(cap if cap is not None else os.environ.get("SC_TOKEN_CAP", "200000"))
        cap_env = os.environ.get("SC_USD_CAP")
        self.usd_cap = usd_cap if usd_cap is not None else (float(cap_env) if cap_env else None)
        self.ledger = ledger
        d = self._load()
        self.total = int(d.get("total_tokens", 0))
        self.usd = float(d.get("usd_estimated", USD_BEFORE_SPLIT if self.total else 0.0))
        self._inflight = 0
        self._inflight_usd = 0.0
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            return json.loads(self.ledger.read_text())
        except (OSError, ValueError):
            return {}

    def check(self, estimated: int, price: tuple = DEFAULT_PRICE) -> int:
        worst_usd = estimated * max(price)
        with self._lock:
            if self.total + self._inflight + estimated > self.cap:
                raise BudgetExceeded(
                    f"token cap {self.cap} would be exceeded: used {self.total}, in flight {self._inflight}, "
                    f"next call up to {estimated}"
                )
            if self.usd_cap is not None and self.usd + self._inflight_usd + worst_usd > self.usd_cap:
                raise BudgetExceeded(f"dollar cap ${self.usd_cap:.2f} would be exceeded: spent about ${self.usd:.2f}, "
                                     f"next call up to ${worst_usd:.4f}")
            self._inflight += estimated
            self._inflight_usd += worst_usd
        return estimated

    def release(self, reserved: int, price: tuple = DEFAULT_PRICE) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - reserved)
            self._inflight_usd = max(0.0, self._inflight_usd - reserved * max(price))

    def record(self, prompt: int, completion: int, reserved: int = 0, price: tuple = DEFAULT_PRICE) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - reserved)
            self._inflight_usd = max(0.0, self._inflight_usd - reserved * max(price))
            self.total += prompt + completion
            self.usd += prompt * price[0] + completion * price[1]
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            self.ledger.write_text(json.dumps({"total_tokens": self.total, "usd_estimated": round(self.usd, 4)}))
            total, usd = self.total, self.usd
        cap = f", ${usd:.2f}/${self.usd_cap:.2f}" if self.usd_cap is not None else f", about ${usd:.2f} spent"
        print(f"[budget] +{prompt}p +{completion}c => {total}/{self.cap} tokens{cap}", flush=True)


def require_api_enabled() -> None:
    """Kill switch: no network call to Token Factory unless SC_ALLOW_API=1 is set explicitly."""
    if os.environ.get("SC_ALLOW_API") != "1":
        raise RuntimeError("API calls are disabled. Set SC_ALLOW_API=1 to allow Token Factory calls.")


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
        # transient 429/5xx/connection errors are retried by the client with exponential backoff (unbilled); out-of-credit errors
        # still surface after that and stop all further calls (see chat)
        self._c = OpenAI(api_key=s["api_key"], base_url=s["base_url"], timeout=90, max_retries=4)

    def chat(self, messages: list[dict], max_tokens: int = 256, **kw):
        require_api_enabled()
        stop = api_stopped()
        if stop:
            raise CreditsExhausted(f"Token Factory calls are stopped since {stop.get('time')}: {stop.get('reason')} "
                                   f"(delete {STOP_FILE.name} after topping up to resume)")
        price = PRICES.get(self.model, DEFAULT_PRICE)
        est_prompt = sum(len(str(m.get("content", ""))) for m in messages) // 3 + 50
        if "tools" in kw:
            est_prompt += len(json.dumps(kw["tools"])) // 3
        reserved = self.budget.check(est_prompt + max_tokens, price)
        try:
            resp = self._c.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, **kw
            )
        except BaseException as e:
            self.budget.release(reserved, price)
            if is_out_of_credits(e):
                import time
                STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
                STOP_FILE.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": getattr(e, "status_code", None),
                                                 "reason": str(e)[:300]}), encoding="utf-8")
                raise CreditsExhausted(f"out of Token Factory credits; all further calls are stopped: {str(e)[:200]}") from e
            raise
        u = resp.usage
        self.budget.record(u.prompt_tokens if u else est_prompt, u.completion_tokens if u else max_tokens, reserved, price)
        return resp
