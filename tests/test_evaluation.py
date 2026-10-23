from pathlib import Path

import pytest

from spatialcontract import evaluation, mutate

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "seeded"
src = lambda cid: (CORPUS / f"{cid}.py").read_text()   # noqa: E731


def test_wilson_interval_known_values():
    lo, hi = evaluation.wilson(7, 7)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.646, abs=0.01)    # Wilson (not Clopper-Pearson) for 7/7
    lo, hi = evaluation.wilson(0, 10)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.278, abs=0.01)


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    vals = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    a = evaluation.boot_mean_ci([float(v) for v in vals])
    assert a == evaluation.boot_mean_ci([float(v) for v in vals])
    assert a[1] <= a[0] <= a[2] and a[0] == pytest.approx(0.7)


def test_paired_difference_uses_only_common_cases():
    a, b = {"x": 1.0, "y": 1.0, "z": 0.0}, {"x": 0.0, "y": 1.0, "w": 1.0}
    mean, lo, hi = evaluation.boot_diff_ci(a, b)
    assert mean == pytest.approx(0.5) and lo <= mean <= hi


def test_per_case_averages_repetitions():
    rows = [{"arm": "a", "id": "c1", "repaired": True}, {"arm": "a", "id": "c1", "repaired": False},
            {"arm": "a", "id": "c2", "repaired": True}, {"arm": "b", "id": "c1", "repaired": True}]
    assert evaluation.per_case(rows, "a", "repaired") == {"c1": 0.5, "c2": 1.0}


def test_mutation_candidates_include_the_known_single_edit_fixes():
    cands = dict(mutate.generate(src("R05_stale_affine")))
    assert any(name.startswith("aff->new_aff") for name in cands)
    names = [n for n, _ in mutate.generate(src("S06_reshape_order"))]
    assert any(n.startswith("drop-kw(order)") for n in names)
    names = [n for n, _ in mutate.generate(src("C02_index_order"))]
    assert any(n.startswith("unwrap[::-1]") for n in names)


def test_mutations_are_valid_python_and_never_touch_the_contract():
    from spatialcontract import agent
    original = src("R01_swap_transpose")
    cands = mutate.generate(original)
    assert len(cands) > 50
    for _, code in cands[:200]:
        compile(code, "<m>", "exec")
        assert not any("CONTRACT" in p for p in agent.check_patch(original, code))


def test_scoring_distinguishes_repaired_unchanged_and_overfit():
    ref = src("K01_R_reference")
    bug = src("R05_stale_affine")
    fixed = evaluation.score(bug, ref, 0)
    assert fixed["repaired"] and fixed["changed"]
    untouched_bug = evaluation.score(bug, None, 0)
    assert not untouched_bug["visible_pass"] and not untouched_bug["repaired"] and not untouched_bug["changed"]
    untouched_ctrl = evaluation.score(ref, None, 0)
    assert untouched_ctrl["visible_pass"] and not untouched_ctrl["changed"]
    assert not evaluation.score(src("A01_no_contract_clean"), None, 0)["scorable"]


def test_search_repairs_a_single_edit_bug_without_any_model():
    r = mutate.search(CORPUS / "R05_stale_affine.py", workers=8)
    assert r.outcome == "REPAIRED" and r.visible_passed and r.heldout["passed"]
    assert mutate.search(CORPUS / "A02_no_contract_buggy.py").outcome == "ABSTAIN"
