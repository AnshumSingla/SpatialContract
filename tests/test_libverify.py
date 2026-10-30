"""The two library gates on real repairs from our runs: they reject the plausible-but-wrong ones and accept the real ones.
Needs the real-bug worktrees and environments under _work/ (skipped in a fresh clone)."""
import json
import subprocess
from pathlib import Path

import pytest

from spatialcontract import libagent, libmutate, libverify, realbugs

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.id: c for c in realbugs.CASES}
needs_work = pytest.mark.skipif(not (realbugs.WORK / "wt" / "torchio_sitk_ras_lps_pre").exists() or not realbugs.ENVS["modern"].exists(),
                                reason="needs the real-bug worktrees and environments under _work/")


def _ws_with_diff(case_id, diff):
    ws = libagent.Workspace(CASES[case_id])
    subprocess.run(["git", "apply", "-p1", "--directory=lib", "--unsafe-paths", str(ROOT / diff)], cwd=ws.root, check=True, capture_output=True)
    for p in (ws.lib / ws.pkg).rglob("*.py"):            # record the edited files so ws.touched() sees them
        rel = p.relative_to(ws.lib).as_posix()
        ref = ws.case.worktree("pre") / ws.case.src_subdir / rel
        if ref.exists() and ref.read_bytes() != p.read_bytes():
            ws.originals[rel] = ref.read_text(encoding="utf-8")
    return ws


@needs_work
@pytest.mark.parametrize("diff,verdict", [("results/real_repair/staged1/logs/torchio_sitk_ras_lps_r1.diff", "FAIL"),
                                          ("results/real_repair/real3/logs/torchio_sitk_ras_lps_r1.diff", "PASS")])
def test_reader_gate(diff, verdict):
    ws = _ws_with_diff("torchio_sitk_ras_lps", diff)
    try:
        assert libverify.reader(ws)["verdict"] == verdict
    finally:
        ws.cleanup()


@needs_work
def test_regression_gate_rejects_the_deleted_crop():
    row = next(json.loads(l) for l in (ROOT / "results/real_repair/mut1/raw.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(l)["case"] == "torchio_crop")
    target, label = row["winner"].split(" ", 1)
    rel, q = target.split("::")
    ws = libagent.Workspace(CASES["torchio_crop"])
    try:
        src = ws._resolve(rel)
        ws.originals[rel] = src.read_text(encoding="utf-8")
        src.write_text(dict(libmutate.mutants(ws.originals[rel], q))[label], encoding="utf-8")
        g = libverify.regression(ws)
        assert g["verdict"] == "FAIL" and any("test_crop" in t for t in g["newly_failing"])
    finally:
        ws.cleanup()
