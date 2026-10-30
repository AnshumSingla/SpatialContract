import pytest

from spatialcontract import llm


def test_api_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SC_ALLOW_API", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        llm.require_api_enabled()


def test_chat_refuses_without_flag_before_any_network(monkeypatch, tmp_path):
    monkeypatch.delenv("SC_ALLOW_API", raising=False)
    monkeypatch.setenv("NEBIUS_API_KEY", "dummy")
    c = llm.Client(budget=llm.TokenBudget(cap=1000, ledger=tmp_path / "l.json"))
    with pytest.raises(RuntimeError, match="disabled"):
        c.chat([{"role": "user", "content": "hi"}])


class _ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


class _FakeCompletions:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    def create(self, **kw):
        self.calls += 1
        raise self.exc


def _client(monkeypatch, tmp_path, exc, **budget):
    monkeypatch.setenv("SC_ALLOW_API", "1")
    monkeypatch.setenv("NEBIUS_API_KEY", "dummy")
    monkeypatch.setattr(llm, "STOP_FILE", tmp_path / "api_stopped.json")
    c = llm.Client(budget=llm.TokenBudget(cap=10**9, ledger=tmp_path / "l.json", **budget))
    fake = _FakeCompletions(exc)
    c._c = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
    return c, fake


def test_out_of_credits_stops_all_further_calls(monkeypatch, tmp_path):
    c, fake = _client(monkeypatch, tmp_path, _ApiError(402, "Payment required: insufficient balance"))
    with pytest.raises(llm.CreditsExhausted):
        c.chat([{"role": "user", "content": "hi"}])
    assert (tmp_path / "api_stopped.json").exists()
    with pytest.raises(llm.CreditsExhausted, match="stopped"):     # refused before any network call
        c.chat([{"role": "user", "content": "hi"}])
    assert fake.calls == 1
    assert issubclass(llm.CreditsExhausted, llm.BudgetExceeded)    # agent loops end with outcome BUDGET


def test_rate_limit_is_not_out_of_credits(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path, _ApiError(429, "Rate limit exceeded, retry later"))
    with pytest.raises(_ApiError):
        c.chat([{"role": "user", "content": "hi"}])
    assert not (tmp_path / "api_stopped.json").exists()
    assert llm.is_out_of_credits(_ApiError(429, "You exceeded your current quota")) is True


def test_dollar_cap_refuses_before_the_call(monkeypatch, tmp_path):
    c, fake = _client(monkeypatch, tmp_path, _ApiError(500, "unused"), usd_cap=0.0001)
    with pytest.raises(llm.BudgetExceeded, match="dollar cap"):
        c.chat([{"role": "user", "content": "hi"}], max_tokens=1000)
    assert fake.calls == 0
