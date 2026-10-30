"""Reader invariant for library repairs: a repair must not change what the library's own loader reports for a file. No tokens.

The driver exercises one path through the library; a repair can make that path right by breaking another one. TorchIO's SimpleITK
bug was "repaired" by flipping axes when reading, which kept the load/save round trip right and made `image.affine` 134 mm wrong.
Every imaging library has a loader, so this loads a phantom through it (one line per library) with the pre-fix, upstream-fixed
and repaired library and compares the reported affine. A repair FAILS when it changes the loader's affine although the upstream fix
did not. The upstream fix is used only to show the rule would not have rejected the true fix.

  python scripts/check_reader.py      -> results/reader_check.json"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from spatialcontract import realbugs  # noqa: E402
from spatialcontract.phantom import make_phantom  # noqa: E402
import check_regressions as cr  # noqa: E402

LOAD = {"torchio": "import torchio as tio; a = np.asarray(tio.ScalarImage(p).affine)",
        "MONAI": "from monai.transforms import LoadImage; a = np.asarray(LoadImage(image_only=True)(p).affine)",
        "nibabel": "import nibabel; a = np.asarray(nibabel.load(p).affine)",
        "nilearn": "from nilearn.image import load_img; a = np.asarray(load_img(p).affine)"}


def probe(case, pythonpath, files) -> list | None:
    code = "import sys, json, numpy as np\nout = []\nfor p in sys.argv[1:]:\n    " + LOAD[case.repo] + "\n    out.append(a.tolist())\nprint(json.dumps(out))"
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ} | {"PYTHONPATH": os.pathsep.join(map(str, pythonpath))}
    r = subprocess.run([str(realbugs.ENVS[case.env]), "-s", "-c", code, *map(str, files)], capture_output=True, text=True, env=env)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def max_diff(a, b) -> float | None:
    import numpy as np
    if a is None or b is None:
        return None
    return round(float(max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, b))), 4)


def main() -> int:
    cases = {c.id: c for c in realbugs.CASES}
    tmp = Path(tempfile.mkdtemp(prefix="sc_reader_"))
    files = []
    for s in (0, 1, 2):
        f = tmp / f"p{s}.nii.gz"
        make_phantom(s).save(f)
        files.append(f)
    out = []
    try:
        for run_id in ("real3", "real4", "staged1", "staged2", "mut1"):
            f = ROOT / f"results/real_repair/{run_id}/raw.jsonl"
            if not f.exists():
                continue
            for row in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
                if row["outcome"] != "REPAIRED" and row.get("outcome_before_noop_check") != "REPAIRED":
                    continue
                case = cases[row["case"]]
                pre = probe(case, case.pythonpath("pre"), files)
                post = probe(case, case.pythonpath("post"), files)
                tree = cr.patched_tree(case, run_id, row)
                try:
                    rep = probe(case, [tree / "wt" / case.src_subdir, realbugs.WORK / "shims"], files)
                finally:
                    shutil.rmtree(tree, ignore_errors=True)
                d_up, d_rep = max_diff(pre, post), max_diff(pre, rep)
                fail = d_rep is not None and d_rep > 1e-3 and (d_up is not None and d_up <= 1e-3)
                out.append({"run": f"{run_id}/{row['case']}#r{row['rep']}", "status": "READER_CHANGED" if fail else "OK",
                            "upstream_changes_reader_mm": d_up, "repair_changes_reader_mm": d_rep})
                print(f"{out[-1]['run']:42s} {out[-1]['status']:15s} upstream {d_up} mm, repair {d_rep} mm", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (ROOT / "results/reader_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
