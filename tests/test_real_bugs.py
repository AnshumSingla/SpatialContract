"""Consistency checks on the committed real-bug results. Re-running them needs the git worktrees and the two
virtualenvs under _work/ (see README), so these tests only read results/real_bugs_oracle.json."""
import json

import numpy as np
import pytest

from spatialcontract import phantom, realbugs

RESULTS = realbugs.ROOT / "results" / "real_bugs_oracle.json"


def test_every_case_has_a_repro_script_and_pinned_commits():
    for c in realbugs.CASES:
        assert c.script.exists(), c.id
        assert len(c.pre) == 40 and len(c.post) == 40 and c.pre != c.post
        assert c.env in realbugs.ENVS


def test_committed_results_cover_all_cases_and_are_internally_consistent():
    rows = json.loads(RESULTS.read_text())
    assert {r["id"] for r in rows} == {c.id for c in realbugs.CASES}
    for r in rows:
        assert r["reproduced"] == (r["pre"]["verdict"] == "FAIL" and r["post"]["verdict"] == "PASS")


def test_real_corpus_is_not_labelled_synthetic_only():
    rows = json.loads(RESULTS.read_text())
    assert sum(r["reproduced"] for r in rows) >= 5   # the project's own threshold for a real-bug claim


@pytest.mark.parametrize("seed", [200, 203, 207])
def test_strong_oblique_phantoms_are_actually_strongly_oblique(seed):
    p = phantom.make_phantom(seed)
    A = p.affine[:3, :3] / np.sqrt((p.affine[:3, :3] ** 2).sum(0))
    # compare with the nearest signed permutation: a strongly oblique matrix is far from every one of them
    best = min(np.linalg.norm(A - P) for P in phantom._signed_perms())
    assert best > 0.7
