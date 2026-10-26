"""Agent-loop mechanics with a scripted fake model (no network, no tokens)."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import nibabel as nib
import pytest

from spatialcontract import agent
from spatialcontract.llm import BudgetExceeded
from spatialcontract.phantom import make_phantom

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "seeded"
src = lambda cid: (CORPUS / f"{cid}.py").read_text()   # noqa: E731


def tool_resp(source, rationale="fix", n=0):
    call = NS(id=f"call{n}", function=NS(name="submit_patch", arguments=json.dumps(
        {"rationale": rationale, "patched_source": source})))
    msg = NS(content=None, tool_calls=[call], reasoning_content="thinking")
    return NS(choices=[NS(message=msg, finish_reason="tool_calls")], usage=NS(prompt_tokens=100, completion_tokens=50))


def text_resp(text):
    return NS(choices=[NS(message=NS(content=text, tool_calls=None), finish_reason="stop")],
              usage=NS(prompt_tokens=10, completion_tokens=5))


class Fake:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def chat(self, messages, **kw):
        self.calls.append(list(messages))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def run(tmp_path, cid, fake, **kw):
    return agent.repair(CORPUS / f"{cid}.py", fake, log_dir=tmp_path / "log", **kw)


def test_abstains_without_contract_and_never_calls_model(tmp_path):
    fake = Fake()
    r = run(tmp_path, "A02_no_contract_buggy", fake)
    assert r.outcome == "ABSTAIN" and fake.calls == [] and r.llm_calls == 0


def test_ambiguous_contract_abstains(tmp_path):
    assert run(tmp_path, "A05_corner_convention", Fake()).outcome == "ABSTAIN"


def test_correct_code_is_left_alone(tmp_path):
    fake = Fake()
    r = run(tmp_path, "K05_C_reference", fake)
    assert r.outcome == "NO_CHANGE_NEEDED" and fake.calls == []


def test_repairs_and_passes_heldout(tmp_path):
    fake = Fake(tool_resp(src("K01_R_reference")))
    r = run(tmp_path, "R05_stale_affine", fake)
    assert r.outcome == "REPAIRED" and r.iterations == 1 and r.heldout["passed"]
    assert r.tokens == {"prompt": 100, "completion": 50}
    events = [json.loads(l)["event"] for l in Path(r.log_path).read_text().splitlines()]
    assert events == ["initial_oracle", "llm_response", "patch", "oracle_after_patch", "heldout"]


def test_contract_edit_is_rejected_then_real_fix_accepted(tmp_path):
    cheat = src("R05_stale_affine").replace('"array_orientation": "RAS"', '"array_orientation": "LPS"')
    fake = Fake(tool_resp(cheat, n=0), tool_resp(src("K01_R_reference"), n=1))
    r = run(tmp_path, "R05_stale_affine", fake)
    assert r.outcome == "REPAIRED" and r.iterations == 2
    rejection = fake.calls[1][-1]
    assert rejection["role"] == "tool" and "contract" in rejection["content"].lower()


def test_forbidden_import_and_call_rejected():
    base = src("K01_R_reference")
    assert any("os" in p for p in agent.check_patch(base, "import os\n" + base))
    assert any("open" in p for p in agent.check_patch(base, base + "\nopen('x')\n"))
    assert any("signature" in p for p in agent.check_patch(base, base.replace("def process(in_path, out_path)", "def process(a)")))
    assert any("syntax" in p for p in agent.check_patch(base, "def process(:\n"))
    assert agent.check_patch(base, base) == []


def test_overfit_patch_is_caught_by_heldout(tmp_path):
    visible = {"".join(nib.aff2axcodes(make_phantom(s).affine)) for s in (0, 1, 2)}
    overfit = src("R05_stale_affine").replace(
        "    nib.save(nib.Nifti1Image(new, aff), out_path)\n",
        f"    ok = ''.join(nib.aff2axcodes(aff)) in {sorted(visible)!r}\n"
        "    nib.save(nib.Nifti1Image(new, new_aff if ok else aff), out_path)\n")
    r = run(tmp_path, "R05_stale_affine", Fake(tool_resp(overfit)))
    assert r.visible_passed and r.outcome == "OVERFIT" and not r.heldout["passed"]
    assert any(v == "FAIL" for v in r.heldout["summary"].values())


def test_iteration_cap(tmp_path):
    same = src("R05_stale_affine")
    r = run(tmp_path, "R05_stale_affine", Fake(tool_resp(same, n=0), tool_resp(same, n=1)), max_iters=2)
    assert r.outcome == "FAILED" and r.iterations == 2 and not r.visible_passed


def test_budget_exhaustion_stops_cleanly(tmp_path):
    r = run(tmp_path, "R05_stale_affine", Fake(BudgetExceeded("cap")))
    assert r.outcome == "BUDGET"


def test_plain_code_block_fallback(tmp_path):
    fake = Fake(text_resp("Here you go:\n```python\n" + src("K01_R_reference") + "```\n"))
    assert run(tmp_path, "R05_stale_affine", fake).outcome == "REPAIRED"


# ---------------------------------------------------------------- arm 2 (repo tests, no gate, report_no_bug / abstain)
def tool_named(name, **args):
    call = NS(id="c0", function=NS(name=name, arguments=json.dumps(args)))
    return NS(choices=[NS(message=NS(content=None, tool_calls=[call], reasoning_content=None), finish_reason="tool_calls")],
              usage=NS(prompt_tokens=10, completion_tokens=5))


def arm2_factory(base, spec=None):
    from spatialcontract import repo_tests

    def factory(_contract):
        def evaluate(p):
            r = repo_tests.run_repo_tests(p, base, spec or {})
            return r.passed, r.text(), "PASS" if r.passed else "FAIL"
        return evaluate
    return factory


def run2(tmp_path, cid, fake, base, **kw):
    return agent.repair(CORPUS / f"{cid}.py", fake, log_dir=tmp_path / "log", evaluate_factory=arm2_factory(base),
                        gate=False, validate=False, arm="tests", **kw)


def test_arm2_is_not_gated_and_sees_repo_test_results(tmp_path):
    fake = Fake(tool_named("abstain", reason="no contract"))
    r = run2(tmp_path, "A02_no_contract_buggy", fake, "R")
    assert r.outcome == "ABSTAIN" and r.llm_calls == 1               # the LLM was called: there is no deterministic gate
    first_user = fake.calls[0][1]["content"]
    assert "No SPATIAL_CONTRACT is declared" in first_user and "Repo test results" in first_user


def test_arm2_report_no_bug_leaves_code_unchanged(tmp_path):
    r = run2(tmp_path, "K01_R_reference", Fake(tool_named("report_no_bug", reason="looks right")), "R")
    assert r.outcome == "NO_CHANGE_CLAIMED" and r.final_source == src("K01_R_reference")


def test_arm2_stops_when_repo_tests_pass_even_if_oracle_would_not(tmp_path):
    # C01 (stale affine in a crop) is invisible to the crop repo tests: a do-nothing "patch" is accepted by them.
    fake = Fake(tool_resp(src("C01_stale_affine") + "\n# touched\n"))
    r = run2(tmp_path, "C01_stale_affine", fake, "C")
    assert r.outcome == "ACCEPTED_BY_TESTS" and r.iterations == 1


def test_arm2_can_add_nothing_to_a_missing_contract(tmp_path):
    added = src("A02_no_contract_buggy").replace("import numpy as np", 'import numpy as np\nSPATIAL_CONTRACT = {"world_frame": "RAS"}')
    assert any("contract" in p.lower() for p in agent.check_patch(src("A02_no_contract_buggy"), added))


def test_arm0_first_patch_is_final_and_no_feedback_is_shown(tmp_path):
    calls = []

    def factory(_contract):
        def evaluate(p):
            calls.append(p)
            return len(calls) > 1, "", "NONE"
        return evaluate
    fake = Fake(tool_resp(src("K01_R_reference")))
    r = agent.repair(CORPUS / "R05_stale_affine.py", fake, log_dir=tmp_path / "log", evaluate_factory=factory,
                     gate=False, validate=False, arm="none")
    assert r.outcome == "ACCEPTED_BY_TESTS" and r.iterations == 1 and r.llm_calls == 1
    first_user = fake.calls[0][1]["content"]
    assert "No tests or measurements are available" in first_user and "Repo test" not in first_user and "Oracle" not in first_user
    assert "one look" in fake.calls[0][0]["content"]


def test_arm0_can_abstain_or_claim_no_bug(tmp_path):
    fake = Fake(tool_named("abstain", reason="unknown units"))
    r = agent.repair(CORPUS / "A04_units_unknown.py", fake, log_dir=tmp_path / "log", evaluate_factory=lambda c: (lambda p: (False, "", "NONE")),
                     gate=False, validate=False, arm="none")
    assert r.outcome == "ABSTAIN" and r.llm_calls == 1
