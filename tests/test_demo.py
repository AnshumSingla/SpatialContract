import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

import build_static  # noqa: E402
import server  # noqa: E402

REC = json.loads((ROOT / "demo" / "recordings.json").read_text(encoding="utf-8"))
RAW = [json.loads(l) for l in (ROOT / "results" / "eval" / "main" / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
src = lambda cid: (ROOT / "corpus" / "seeded" / f"{cid}.py").read_text(encoding="utf-8")   # noqa: E731


def test_every_recording_is_a_real_run_that_matches_the_evaluation_rows():
    assert len(REC["recordings"]) >= 8
    for r in REC["recordings"]:
        assert r["source"] == src(r["id"])
        m = re.search(r"arm3_r(\d)", r["source_run"])
        row = next(x for x in RAW if x["arm"] == "arm3" and x["rep"] == int(m.group(1)) and x["id"] == r["id"])
        assert r["outcome"] == row["outcome"] and r["tokens_total"] == row["tokens"]
        assert len(r["steps"]) == row["iterations"]
        assert (ROOT / r["source_run"]).exists()


def test_hero_case_shows_green_tests_and_failing_oracle():
    hero = REC["recordings"][0]
    assert hero["id"] == "L04_right_multiply"
    assert hero["repo_tests"]["passed"] is True
    assert hero["initial"]["verdict"] == "FAIL" and hero["initial"]["worst_error_mm"] > 100
    assert hero["steps"][-1]["verdict"] == "PASS" and hero["heldout"]["passed"]
    assert hero["repo_tests"]["baseline_oracle"]["verdict"] == "FAIL"


def test_a_failed_run_and_abstentions_are_included():
    outcomes = {(r["id"], r["outcome"]) for r in REC["recordings"]}
    assert ("S06_reshape_order", "FAILED") in outcomes
    assert any(o == "ABSTAIN" for _, o in outcomes) and any(o == "NO_CHANGE_NEEDED" for _, o in outcomes)
    failed = next(r for r in REC["recordings"] if r["outcome"] == "FAILED")
    assert failed["heldout"] is None


def test_static_page_is_up_to_date_and_self_contained():
    built = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    data = json.dumps(REC, separators=(",", ":")).replace("</", "<\\/")
    assert built == html.replace("/*__DATA__*/null", data), "docs/index.html is stale: run python demo/build_static.py"
    assert not re.search(r"https?://", built) and "<script src" not in built and "<link " not in built


def test_live_check_endpoint_logic():
    r = server.run_check(src("L04_right_multiply"))
    assert r["report"]["verdict"] == "FAIL" and r["repo_tests"]["passed"] is True
    assert server.run_check(src("K05_C_reference"))["report"]["verdict"] == "PASS"
    ab = server.run_check(src("A02_no_contract_buggy"))
    assert ab["report"] is None and ab["reasons"]


def test_live_server_refuses_unsafe_code_and_calls_when_api_is_off():
    assert "os" in server.run_check("import os\n" + src("R05_stale_affine"))["error"]
    assert "open" in server.run_check(src("R05_stale_affine") + "\nopen('x')\n")["error"]
    assert "too large" in server.run_check("x = 1\n" * 20000)["error"]
    server.STATE["api"] = False
    assert "disabled" in server.start_job(src("R05_stale_affine"))["error"]


@pytest.mark.parametrize("cid", ["L04_right_multiply", "A05_corner_convention"])
def test_cli_replay_prints_the_story(cid, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo
    assert demo.replay(cid) == 0
    out = capsys.readouterr().out
    assert "OUTCOME:" in out and "SPATIAL_CONTRACT" in out
