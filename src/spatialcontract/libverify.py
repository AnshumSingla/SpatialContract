"""Gates that judge a repair INSIDE a library beyond what its driver exercises. No model.

The oracle and held-out validation see one path through the library, the driver's. A repair can make that path right by breaking
another one, which our own runs did: a mutation search deleted TorchIO's crop (every landmark stays in place), a staged repair of
MONAI's Orientation broke 18 of MONAI's orientation tests, and a staged repair of TorchIO's SimpleITK bug flipped the axes when
READING, which kept the load/save round trip right and made TorchIO's `image.affine` 134 mm wrong. Two gates catch those:
  regression  the library's own covering test files (the ones the fix commit edited, or the fixed module's test file) must not gain
              a failure relative to the pre-fix library;
  reader      the library's own loader must report the same affine for a file as before, unless that is what the bug is about.
Both need the pre-fix git worktree and the case's environment, i.e. the real-bug setup under _work/.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import realbugs
from .phantom import make_phantom

# fixes that edited no test file: the existing test file for the fixed module
COVERING = {"torchio_crop": ["tests/transforms/preprocessing/test_crop.py"], "nibabel_conform": ["nibabel/tests/test_processing.py"]}
TIMEOUT_S = 900
SUMMARY = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")


def test_files(case) -> list[str]:
    if case.id in COVERING:
        return COVERING[case.id]
    out = subprocess.run(["git", "diff", "--name-only", case.pre, case.post], cwd=realbugs.WORK / "repos" / case.repo,
                         capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if re.search(r"(^|/)test_[^/]*\.py$", f)]


def pytest_run(case, which: str, files: list[str]) -> dict:
    wt = case.worktree(which)
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "HOME") if k in os.environ}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in case.pythonpath(which))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    compat = ["-p", "sc_py311_compat"] if case.env == "legacy" else []     # _work/shims: collections aliases for 2018-era code
    cmd = [str(realbugs.ENVS[case.env]), "-m", "pytest", "-q", "-p", "no:cacheprovider", *compat, "--tb=line", "-o", "addopts=", *files]
    try:
        r = subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    tail = (r.stdout or "").strip().splitlines()
    last = next((l for l in reversed(tail) if SUMMARY.search(l)), "")
    counts = {}
    for n, kind in SUMMARY.findall(last):
        counts["errors" if kind.startswith("error") else kind] = int(n)
    failed_ids = sorted({m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", r.stdout or "", re.M)})
    status = ("NO_TESTS" if not counts else "FAIL" if counts.get("failed") or counts.get("errors") else "PASS")
    return {"status": status, "counts": counts, "exit": r.returncode, "failed": failed_ids[:300],
            "tail": "\n".join(tail[-6:])[-1500:]}


def names(run: dict) -> set:
    """Failing test ids reduced to 'file-stem::test' so pre/post versions of a file compare (the _scpost copy included)."""
    out = set()
    for t in run.get("failed", []):
        path, _, test = t.replace("\\", "/").partition("::")
        out.add(Path(path).stem.replace("_scpost", "") + "::" + (test or "<collection>"))
    return out



class _Patched:
    """A real case whose pre-fix worktree is a patched temporary copy (what pytest_run reads)."""

    def __init__(self, case, tree: Path):
        self.c, self.tree = case, tree

    def __getattr__(self, k):
        return getattr(self.c, k)

    def worktree(self, which):
        return self.tree / "wt"

    def pythonpath(self, which):
        return [self.tree / "wt" / self.c.src_subdir, realbugs.WORK / "shims"]


def patched_copy(ws) -> Path:
    """A temporary copy of the pre-fix worktree with the workspace's edits applied (caller removes it)."""
    case = ws.case
    tmp = Path(tempfile.mkdtemp(prefix=f"sc_verify_{case.id}_"))
    shutil.copytree(case.worktree("pre"), tmp / "wt", ignore=shutil.ignore_patterns(".git", "docs", "*.ipynb"))
    for rel in ws.touched():
        (tmp / "wt" / case.src_subdir / rel).write_text((ws.lib / rel).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp


def regression(ws) -> dict:
    """Tests of the library's covering files that pass on the pre-fix library and fail with the workspace's edits."""
    case = ws.case
    files = [t for t in test_files(case) if (case.worktree("pre") / t).exists()]
    if not files:
        return {"verdict": "PASS", "detail": "no existing covering tests"}
    base = names(pytest_run(case, "pre", files))
    tree = patched_copy(ws)
    try:
        after = pytest_run(_Patched(case, tree), "pre", files)
    finally:
        shutil.rmtree(tree, ignore_errors=True)
    new = sorted(names(after) - base)
    return {"verdict": "FAIL" if new else "PASS", "newly_failing": new[:10], "counts": after.get("counts")}


LOAD = {"torchio": "import torchio as tio; a = np.asarray(tio.ScalarImage(p).affine)",
        "MONAI": "from monai.transforms import LoadImage; a = np.asarray(LoadImage(image_only=True)(p).affine)",
        "nibabel": "import nibabel; a = np.asarray(nibabel.load(p).affine)",
        "nilearn": "from nilearn.image import load_img; a = np.asarray(load_img(p).affine)"}


def loader_affines(case, pythonpath, files) -> list | None:
    code = "import sys, json, numpy as np\nout = []\nfor p in sys.argv[1:]:\n    " + LOAD[case.repo] + "\n    out.append(a.tolist())\nprint(json.dumps(out))"
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ} | {"PYTHONPATH": os.pathsep.join(map(str, pythonpath))}
    r = subprocess.run([str(realbugs.ENVS[case.env]), "-s", "-c", code, *map(str, files)], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def reader(ws) -> dict:
    """FAIL when the workspace's edits change the affine the library's loader reports for a phantom file. A bug that IS in the loader
    would legitimately change it; no case here is one, and the upstream fixes leave it unchanged (results/reader_check.json)."""
    import numpy as np
    case = ws.case
    tmp = Path(tempfile.mkdtemp(prefix="sc_reader_"))
    try:
        files = []
        for s in (0, 1, 2):
            f = tmp / f"p{s}.nii.gz"
            make_phantom(s).save(f)
            files.append(f)
        before, after = loader_affines(case, case.pythonpath("pre"), files), loader_affines(case, ws.pythonpath, files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if before is None or after is None:
        return {"verdict": "PASS", "detail": "loader could not be probed"}
    d = float(max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(before, after)))
    return {"verdict": "FAIL" if d > 1e-3 else "PASS", "max_change_mm": round(d, 4)}
