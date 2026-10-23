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
