"""Library-repair machinery. Tests that need the git worktrees and virtualenvs under _work/ skip themselves when those are absent."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from spatialcontract import libagent, realbugs
from spatialcontract.contract import load_from_source

CASE = next(c for c in realbugs.CASES if c.id == "torchio_crop")
HAVE = CASE.worktree("pre").exists() and realbugs.ENVS[CASE.env].exists()
needs_env = pytest.mark.skipif(not HAVE, reason="real-bug worktrees/environments not set up (scripts/setup_real_bugs.py)")


@pytest.fixture
def ws():
    w = libagent.Workspace(CASE)
    yield w
    w.cleanup()


@needs_env
def test_tools_list_read_search(ws):
    assert "torchio/transforms" in ws.list_dir("")
    hits = ws.search(r"class Crop\b")
    assert "crop.py" in hits and ":" in hits
    out = ws.read_file("torchio/transforms/preprocessing/spatial/crop.py", 1, 5)
    assert out.splitlines()[1].startswith("    1|") and "lines 1-5" in out
    assert "bad regex" in ws.search("(")
    assert len(ws.read_file("torchio/transforms/preprocessing/spatial/crop.py", 1, 10_000).splitlines()) <= libagent.MAX_READ_LINES + 2


@needs_env
@pytest.mark.parametrize("glob", ["*.py", "crop.py", "crop*", "torchio/*.py", "torchio/**/*.py", "torchio/transforms/preprocessing/spatial/crop.py",
                                   "**/crop.py", "transforms/*/spatial/*.py", "torchio/transforms/**/crop.py"])
def test_search_globs_models_actually_use_find_the_file(ws, glob):
    # the first full real-bug run searched with patterns like 'nibabel/processing.py' and got 'no matches' for text that was there
    assert "crop.py:" in ws.search("class Crop[(]", glob), glob


@needs_env
def test_edit_guards(ws):
    f = "torchio/transforms/preprocessing/spatial/crop.py"
    assert "outside" in ws.replace_in_file("../../etc/passwd", "a", "b") or "inside" in ws.replace_in_file("../../etc/passwd", "a", "b")
    assert "occurs 0 times" in ws.replace_in_file(f, "no such text anywhere", "x")
    assert "occurs" in ws.replace_in_file(f, "image", "img")                              # not unique
    assert "syntax error" in ws.replace_in_file(f, "    def apply_transform(self, sample) -> Subject:", "    def apply_transform(self, sample) -> Subject:\n  broken(")
    assert "not allowed" in ws.replace_in_file(f, "        low = self.bounds_parameters[::2]", "        import subprocess\n        low = self.bounds_parameters[::2]")
    assert ws.touched() == [] and ws.diff() == ""
    assert ws.replace_in_file(f, "        low = self.bounds_parameters[::2]", "        low = self.bounds_parameters[::2]  # touched").startswith("edited")
    assert ws.touched() == [f] and "+        low = self.bounds_parameters[::2]  # touched" in ws.diff()


@needs_env
def test_replace_lines_and_closest_match_hint(ws):
    f = "torchio/transforms/preprocessing/spatial/crop.py"
    # the failure that cost a real run 8 attempts: indentation copied from the numbered listing
    hint = ws.replace_in_file(f, "      def apply_transform(self, sample) -> Subject:", "x")
    assert "occurs 0 times" in hint and "Closest region" in hint and "def apply_transform" in hint
    lines = ws.read_file(f).splitlines()
    n = next(int(l.split("|")[0]) for l in lines if "image[DATA] = image[DATA]" in l)
    new = "            image[DATA] = image[DATA][:, i0:i1, j0:j1, k0:k1].clone()\n            image[AFFINE] = new_affine"
    out = ws.replace_lines(f, n, n, new)
    assert out.startswith("edited") and "image[AFFINE] = new_affine" in out
    assert "outside the file" in ws.replace_lines(f, 500, 501, "x")
    assert "syntax error" in ws.replace_lines(f, n, n, "  broken(")


def test_prune_keeps_edits_and_oracle_results():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": nm, "arguments": "{}"}}]}
            for i, nm in enumerate(["read_file", "replace_lines", "run_oracle", "read_file", "search", "read_file", "list_dir", "read_file"])]
    out = []
    for i, m in enumerate(msgs):
        out += [m, {"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"}]
    libagent._prune(out, keep_last=3)
    tool = {m["tool_call_id"]: m["content"] for m in out if m["role"] == "tool"}
    assert tool["c0"].startswith("[earlier read_file") and tool["c3"].startswith("[earlier read_file")
    assert tool["c1"] == "result 1" and tool["c2"] == "result 2"            # edit and oracle results survive
    assert tool["c5"] == "result 5" and tool["c7"] == "result 7"             # the last three stay


@needs_env
def test_gold_fix_passes_and_buggy_copy_fails(ws):
    contract = load_from_source(CASE.script.read_text(encoding="utf-8"))
    assert ws.oracle(contract).verdict == "FAIL"                                           # the copy is genuinely buggy
    for rel in libagent.fix_files(CASE):                                                   # drop in the upstream version of the file
        shutil.copy(CASE.worktree("post") / CASE.src_subdir / rel, ws.lib / rel)
    assert ws.oracle(contract).verdict == "PASS"


def test_upstream_files_are_found():
    if not (realbugs.WORK / "repos" / "torchio").exists():
        pytest.skip("repos not cloned")
    assert libagent.fix_files(CASE) == ["torchio/transforms/preprocessing/spatial/crop.py"]


# ---- the loop, with a scripted fake model ---------------------------------------------------------------------------------
def call(name, n, **args):
    return NS(id=f"c{n}", function=NS(name=name, arguments=json.dumps(args)))


def resp(*calls):
    return NS(choices=[NS(message=NS(content=None, tool_calls=list(calls)), finish_reason="tool_calls")], usage=NS(prompt_tokens=100, completion_tokens=20))


class Fake:
    def __init__(self, *rs):
        self.rs, self.calls = list(rs), []

    def chat(self, messages, **kw):
        self.calls.append(list(messages))
        return self.rs.pop(0)


F = "torchio/transforms/preprocessing/spatial/crop.py"
EDIT1 = call("replace_in_file", 2, path=F, old_string="from ....torchio import DATA\n", new_string="from ....torchio import DATA, AFFINE\n")
EDIT2 = call("replace_in_file", 3, path=F, old_string="clone()\n        return sample", new_string="clone()\n            image[AFFINE] = new_affine\n        return sample")


@needs_env
def test_loop_gives_up_cleanly(tmp_path):
    r = libagent.repair_library(CASE, Fake(resp(call("give_up", 1, reason="cannot find it"))), log_path=tmp_path / "l.jsonl")
    assert r.outcome == "GAVE_UP" and r.touched == [] and not r.localized and r.tool_counts == {"give_up": 1}


@needs_env
def test_loop_repairs_with_scripted_tool_calls_and_validates(tmp_path):
    fake = Fake(resp(call("search", 1, pattern=r"class Crop\b"), call("read_file", 2, path=F, start=1, end=40)),
                resp(EDIT1, EDIT2), resp(call("run_oracle", 4)))
    r = libagent.repair_library(CASE, fake, log_path=tmp_path / "l.jsonl")
    assert r.outcome == "REPAIRED" and r.visible_passed and r.heldout["passed"], (r.outcome, r.reasons, r.diff)
    assert r.touched == [F] and r.localized and r.oracle_runs == 1
    assert "+            image[AFFINE] = new_affine" in r.diff
    events = [json.loads(x)["event"] for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert events[0] == "start" and "oracle" in events and events[-2:] == ["heldout", "end"]
    # every tool call got a tool message back (the protocol the API requires)
    tool_msgs = [m for m in fake.calls[1] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}


@needs_env
def test_loop_step_cap_is_a_failure_not_a_crash(tmp_path):
    fake = Fake(resp(call("list_dir", 1, path="torchio")), resp(call("list_dir", 2, path="torchio")))
    r = libagent.repair_library(CASE, fake, max_steps=2, log_path=tmp_path / "l.jsonl")
    assert r.outcome == "FAILED" and "step cap reached" in r.reasons and r.steps == 2
