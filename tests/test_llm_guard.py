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
