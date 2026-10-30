"""Regression gate for library repairs: a repair must not break the library's own existing tests. No tokens.

The oracle and held-out validation see only what the driver exercises. A staged repair of TorchIO's SimpleITK bug made the
load/save round trip right by flipping the axes on READ, which made TorchIO's public `image.affine` 134 mm wrong for every NIfTI:
invisible to the driver, but exactly what a library's own tests guard. For every accepted library repair, this copies the pre-fix
worktree, applies the repair, runs the covering test files (as in run_prefix_tests.py) and reports tests that pass on the pre-fix
library but fail with the repair. A repair with such a failure is a REGRESSION.

  python scripts/check_regressions.py      -> results/regression_check.json"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from spatialcontract import libagent, libmutate, realbugs  # noqa: E402
import run_prefix_tests as rpt  # noqa: E402


def patched_tree(case, run_id: str, row: dict) -> Path:
    """A temporary copy of the pre-fix worktree with the repair applied to its library package."""
    ws = libagent.Workspace(case)
    try:
        name = row["case"] if run_id.startswith("mut") else f"{row['case']}_r{row['rep']}"
        if run_id.startswith("mut"):
            target, label = row["winner"].split(" ", 1)
            rel, q = target.split("::")
            src = ws._resolve(rel)
            original = src.read_text(encoding="utf-8")
            ws.originals[rel] = original
            src.write_text(dict(libmutate.mutants(original, q))[label], encoding="utf-8")
        else:
            subprocess.run(["git", "apply", "-p1", "--directory=lib", "--unsafe-paths", str(ROOT / f"results/real_repair/{run_id}/logs/{name}.diff")],
                           cwd=ws.root, check=True, capture_output=True)
            for p in (ws.lib / ws.pkg).rglob("*.py"):           # git apply leaves no originals record; diff against the worktree instead
                rel = p.relative_to(ws.lib).as_posix()
                ref = case.worktree("pre") / case.src_subdir / rel
                if ref.exists() and ref.read_bytes() != p.read_bytes():
                    ws.originals[rel] = ref.read_text(encoding="utf-8")
        tmp = Path(tempfile.mkdtemp(prefix=f"sc_regr_{case.id}_"))
        shutil.copytree(case.worktree("pre"), tmp / "wt", ignore=shutil.ignore_patterns(".git", "docs", "*.ipynb"))
        for rel in ws.originals:
            dest = tmp / "wt" / case.src_subdir / rel
            dest.write_text((ws.lib / rel).read_text(encoding="utf-8"), encoding="utf-8")
        return tmp
    finally:
        ws.cleanup()


class _Case:
    """The real case with its pre-fix worktree swapped for the patched copy (what run_prefix_tests.pytest_run reads)."""
    def __init__(self, case, tree):
        self.c, self.tree = case, tree

    def __getattr__(self, k):
        return getattr(self.c, k)

    def worktree(self, which):
        return self.tree / "wt"

    def pythonpath(self, which):
        return [self.tree / "wt" / self.c.src_subdir, realbugs.WORK / "shims"]


def main() -> int:
    cases = {c.id: c for c in realbugs.CASES}
    out = []
    for run_id in ("real3", "real4", "staged1", "staged2", "mut1"):
        f = ROOT / f"results/real_repair/{run_id}/raw.jsonl"
        if not f.exists():
            continue
        for row in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            if row["outcome"] != "REPAIRED" and row.get("outcome_before_noop_check") != "REPAIRED":
                continue
            case = cases[row["case"]]
            files = [t for t in rpt.test_files(case) if (case.worktree("pre") / t).exists()]
            if not files:
                out.append({"run": f"{run_id}/{row['case']}#r{row['rep']}", "status": "NO_EXISTING_TESTS"})
                continue
            base = rpt.names(rpt.pytest_run(case, "pre", files))           # failures already present before any repair
            tree = patched_tree(case, run_id, row)
            try:
                after = rpt.pytest_run(_Case(case, tree), "pre", files)
            finally:
                shutil.rmtree(tree, ignore_errors=True)
            new = sorted(rpt.names(after) - base)
            out.append({"run": f"{run_id}/{row['case']}#r{row['rep']}", "status": "REGRESSION" if new else "OK", "tests": after.get("counts"),
                        "newly_failing": new[:10]})
            print(f"{out[-1]['run']:42s} {out[-1]['status']:10s} {after.get('counts')} {new[:3]}", flush=True)
    (ROOT / "results/regression_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
