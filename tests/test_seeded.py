import json

import pytest

from spatialcontract import seeded


def test_corpus_meets_size_requirements():
    bugs = [c for c in seeded.ALL_CASES if c.kind == "bug"]
    assert len(bugs) >= 20
    assert {c.bug_class for c in bugs} == set(seeded.CLASSES)
    assert sum(c.kind == "control" for c in seeded.ALL_CASES) >= 10
    assert sum(c.kind == "abstain" for c in seeded.ALL_CASES) >= 6
    assert len({c.id for c in seeded.ALL_CASES}) == len(seeded.ALL_CASES)


def test_every_case_source_is_valid_python_and_replacements_apply():
    for c in seeded.ALL_CASES:
        compile(c.source(), c.id, "exec")


def test_committed_corpus_matches_generator():
    manifest = json.loads((seeded.CORPUS / "manifest.json").read_text())
    assert [m["id"] for m in manifest] == [c.id for c in seeded.ALL_CASES]
    for c in seeded.ALL_CASES:
        assert (seeded.CORPUS / f"{c.id}.py").read_text(encoding="utf-8") == c.source(), c.id


@pytest.fixture(scope="module")
def rows():
    return seeded.evaluate(workers=6)


def test_every_seeded_bug_detected(rows):
    s = seeded.summarize(rows)
    assert s["missed_bugs"] == [], s


def test_no_false_alarm_on_correct_controls(rows):
    s = seeded.summarize(rows)
    assert s["false_alarm_ids"] == [], s


def test_abstain_cases_abstain_without_running_code(rows):
    for r in rows:
        if r["kind"] == "abstain":
            assert r["verdict"] == "ABSTAIN", r["id"]


def test_strong_repo_tests_flag_every_bug_and_no_control():
    """The stronger baseline suite (arm 2s) must never fail a correct control, otherwise it would not be a fair baseline."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from spatialcontract import repo_tests
    from spatialcontract.seeded import CORPUS
    m = [x for x in json.loads((CORPUS / "manifest.json").read_text()) if x["kind"] in ("bug", "control")]

    def run(x):
        return x["kind"], repo_tests.run_repo_tests(CORPUS / x["file"], x["base"], x["spec"], strong=True).passed
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, m))
    assert all(not passed for kind, passed in res if kind == "bug")
    assert all(passed for kind, passed in res if kind == "control")
