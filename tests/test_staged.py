"""Whole-function replacement used by the staged library repair (no model, no network)."""
import textwrap

from spatialcontract.libagent import Workspace
from spatialcontract.staged import _reset, _source, apply_edits

SRC = '''
import numpy as np


class Crop:
    def __init__(self, n):
        self.n = n

    @staticmethod
    def helper(x):
        return x

    def apply(self, affine):
        new_affine = affine.copy()
        new_affine[:3, 3] += self.n
        return affine
'''


def _ws(tmp_path):
    ws = Workspace.__new__(Workspace)
    ws.lib, ws.pkg, ws.originals = tmp_path / "lib", "fakelib", {}
    (ws.lib / "fakelib").mkdir(parents=True)
    (ws.lib / "fakelib" / "__init__.py").write_text("")
    (ws.lib / "fakelib" / "crop.py").write_text(textwrap.dedent(SRC))
    return ws


def test_replace_method_reindents_and_reverts(tmp_path):
    ws = _ws(tmp_path)
    new = "def apply(self, affine):\n    new_affine = affine.copy()\n    new_affine[:3, 3] += self.n\n    return new_affine\n"
    assert apply_edits(ws, [{"path": "fakelib/crop.py", "function": "Crop.apply", "new_source": new}]) == []
    text = (ws.lib / "fakelib" / "crop.py").read_text()
    assert "        return new_affine" in text and "def helper" in text and "@staticmethod" in text
    assert ws.touched() == ["fakelib/crop.py"]
    assert "return new_affine" in _source(ws, "fakelib/crop.py", "Crop.apply")
    _reset(ws)
    assert "return new_affine" not in (ws.lib / "fakelib" / "crop.py").read_text()


def test_bad_edits_are_reported_not_applied(tmp_path):
    ws = _ws(tmp_path)
    probs = apply_edits(ws, [{"path": "fakelib/crop.py", "function": "Crop.missing", "new_source": "def missing(self):\n    pass\n"},
                             {"path": "fakelib/crop.py", "function": "Crop.apply", "new_source": "def apply(self, affine):\n    return (\n"},
                             {"path": "../outside.py", "function": "f", "new_source": "def f():\n    pass\n"},
                             {"path": "fakelib/crop.py", "function": "Crop.apply", "new_source": "def apply(self, a):\n    exec('x')\n"}])
    assert len(probs) == 4 and ws.touched() == []


def test_edits_given_as_json_string(tmp_path):
    import json
    ws = _ws(tmp_path)
    edits = json.dumps([{"path": "fakelib/crop.py", "function": "Crop.helper", "new_source": "def helper(x):\n    return x + 1\n"}])
    assert apply_edits(ws, edits) == [] and ws.touched() == ["fakelib/crop.py"]
    assert apply_edits(ws, ["not an object"]) == ["each edit must be an object with path, function and new_source"]


def test_library_mutants_include_the_fix():
    from spatialcontract.libmutate import mutants
    cands = mutants(textwrap.dedent(SRC), "Crop.apply")
    assert len(cands) > 5
    assert any("return new_affine" in src for _, src in cands)
    dead = textwrap.dedent('''
    class Crop:
        def apply(self, image, lo):
            new_affine = image.affine.copy()
            new_affine[:3, 3] += lo
            image.data = image.data[lo:]
            return image
    ''')
    assert mutants(dead, "Crop.apply")       # spatial local swaps and statement deletions exist
