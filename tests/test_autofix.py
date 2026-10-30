"""Deterministic repair: applied only when every visible phantom shows the same misplacement, and then it must pass the guard and the oracle."""
import ast
from pathlib import Path

import numpy as np
import pytest

from spatialcontract import agent, autofix, oracle
from spatialcontract.contract import load_from_source

SEEDED = Path(__file__).resolve().parents[1] / "corpus" / "seeded"


@pytest.mark.parametrize("case,applies", [("M03_not_converted", True), ("L03_no_conversion", True), ("R02_swap_target_codes", True),
                                          ("C01_stale_affine", False), ("S06_reshape_order", False), ("K01_R_reference", False)])
def test_autofix_only_for_input_independent_misplacement(tmp_path, case, applies):
    src = (SEEDED / f"{case}.py").read_text(encoding="utf-8")
    contract = load_from_source(src)
    corr = autofix.consistent_map(oracle.check(SEEDED / f"{case}.py", contract), contract)
    assert (corr is not None) is applies
    if not applies:
        return
    args = [a.arg for a in next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "process").args.args]
    patched = autofix.wrapper_source(src, corr, contract, args)
    assert agent.check_patch(src, patched) == []
    p = tmp_path / f"{case}.py"
    p.write_text(patched, encoding="utf-8")
    assert oracle.check(p, contract).passed
    assert isinstance(corr, str) or np.allclose(corr[3], [0, 0, 0, 1])


class _NoModel:
    def chat(self, *a, **k):
        raise AssertionError("the model must not be called when the deterministic correction applies")


def test_agent_tries_deterministic_fix_first(tmp_path):
    res = agent.repair(SEEDED / "M03_not_converted.py", _NoModel(), log_dir=tmp_path, autofix=True)
    assert res.outcome == "REPAIRED" and res.llm_calls == 0 and res.heldout["passed"]
    assert "deterministic" in res.reasons[0]
