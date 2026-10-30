"""Contract suggester: the model's answer is validated, unsupported values become 'unknown', and nothing is written to the file."""
import json
from types import SimpleNamespace as NS

from spatialcontract import suggest


class FakeClient:
    def __init__(self, args):
        self.args = args

    def chat(self, messages, **kw):
        call = NS(function=NS(name="propose_contract", arguments=json.dumps(self.args)), id="c1")
        return NS(choices=[NS(message=NS(tool_calls=[call], content=""))], usage=NS(prompt_tokens=100, completion_tokens=20))


ITEM = lambda v, b="stated", e="docstring: returns RAS mm": {"value": v, "basis": b, "evidence": e}  # noqa: E731


def test_valid_suggestion_is_usable():
    s = suggest.suggest("def process(i, o): ...", FakeClient({"world_frame": ITEM("RAS"), "units": ITEM("mm"),
                                                              "affine_convention": ITEM("voxel_center_to_world"),
                                                              "array_orientation": ITEM("none")}))
    assert s.problems == [] and s.contract["array_orientation"] is None
    assert "SPATIAL_CONTRACT = {" in s.block() and "REVIEW" in s.block()


def test_unsupported_or_unknown_values_abstain():
    s = suggest.suggest("x", FakeClient({"world_frame": ITEM("RAS+ish"), "units": ITEM("mm", b="unknown"),
                                         "affine_convention": ITEM("voxel_corner"), "array_orientation": ITEM("RASX")}))
    assert s.contract["world_frame"] == "unknown" and s.contract["units"] == "unknown"
    assert s.contract["affine_convention"] == "unknown" and s.contract["array_orientation"] == "unknown"
    assert s.problems


def test_strip_contract():
    src = 'import x\nSPATIAL_CONTRACT = {\n    "world_frame": "RAS",\n}\ndef process(i, o):\n    pass\n'
    out = suggest.strip_contract(src)
    assert "SPATIAL_CONTRACT" not in out and "def process" in out and "import x" in out
