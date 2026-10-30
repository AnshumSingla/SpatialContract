from pathlib import Path

import numpy as np
import pytest

from spatialcontract import diagnose, oracle

ROOT = Path(__file__).resolve().parents[1]
E = np.random.default_rng(0).uniform(-60, 60, size=(6, 3))


def test_translation_in_whole_voxels():
    aff = np.diag([2.0, 1.5, 3.0, 1.0])
    d = diagnose.describe(E, E + [4.0, 0.0, -6.0], out_affine=aff)
    assert d["kind"] == "translation"
    assert d["shift_voxels"] == [2, 0, -2]


def test_axis_negation_and_permutation():
    d = diagnose.describe(E, E * [-1, -1, 1])
    assert d["kind"] == "signed_permutation" and "x and y negated" in d["text"]
    d = diagnose.describe(E, E[:, [1, 0, 2]] + 5)
    assert "axes permuted" in d["text"] and "(y, x, z)" in d["text"]


def test_units_scale():
    d = diagnose.describe(E, E * 1000.0)
    assert d["kind"] == "scaled" and "1000" in d["text"]


def test_rotation_and_not_affine():
    th = np.radians(30)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    d = diagnose.describe(E, E @ R.T)
    assert d["kind"] == "general_affine" and abs(d["angle_deg"] - 30) < 0.1
    scrambled = E[[3, 0, 5, 1, 4, 2]]
    assert diagnose.describe(E, scrambled)["kind"] == "not_affine"


def test_too_few_landmarks():
    got = [None, None, None] + list(E[3:])
    assert diagnose.describe(E, got) is None


@pytest.mark.parametrize("case,expect", [("L03_no_conversion", "x and y negated"), ("M03_not_converted", "scaled by 1000"),
                                         ("S06_reshape_order", "not moved by any single affine map")])
def test_oracle_feedback_names_the_pattern(case, expect):
    rep = oracle.check(ROOT / "corpus" / "seeded" / f"{case}.py", None)
    assert rep.verdict == "FAIL"
    assert expect in rep.feedback()
    rep.diagnose = False
    assert "pattern:" not in rep.feedback()
