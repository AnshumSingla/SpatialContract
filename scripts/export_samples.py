"""Write example inputs and outputs to samples/ so they can be opened in a NIfTI viewer (ITK-SNAP, 3D Slicer, FSLeyes). No tokens.

One seeded bug, R01_swap_transpose (reorient to RAS, then swaps the first two array axes without touching the affine), its repair as
Nemotron wrote it in the recorded main run (results/eval/main/logs/arm3_r0/..._iter1.py), and the reference pipeline K01, each run on
  phantom_seed1/   a landmark phantom (seed 1: oblique, non-RAS), scored by the oracle in mm per landmark
  real_t1/         nibabel's real T1 test scan, scored by correlation with the input at the same physical points (as validate_real_scans.py)
Also writes samples/preview.png: an axial slice of every output resampled into the input's grid, so a wrong output looks wrong.

  python scripts/export_samples.py        -> samples/"""
import json
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatialcontract import oracle, realscan  # noqa: E402
from spatialcontract.contract import load_from_source  # noqa: E402
from spatialcontract.phantom import make_phantom  # noqa: E402

OUT = ROOT / "samples"
CODE = {"buggy": ROOT / "corpus/seeded/R01_swap_transpose.py",
        "repaired": ROOT / "results/eval/main/logs/arm3_r0/R01_swap_transpose_work/R01_swap_transpose_iter1.py",
        "reference": ROOT / "corpus/seeded/K01_R_reference.py"}
CONTRACT = load_from_source(CODE["reference"].read_text(encoding="utf-8"))


def run(code: Path, src: Path, dst: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sc_sample_") as d:
        ok, err = oracle._run_code(code.resolve(), src, dst, Path(d))
    if not ok:
        raise SystemExit(f"{code.name} failed: {err}")


def on_grid(data: np.ndarray, affine: np.ndarray, grid: nib.Nifti1Image) -> np.ndarray:
    """``data`` sampled (nearest neighbour) at the world position of every voxel of ``grid``, through both affines."""
    idx = np.indices(grid.shape[:3]).reshape(3, -1)
    world = grid.affine @ np.vstack([idx, np.ones(idx.shape[1])])
    v = np.rint((np.linalg.inv(affine) @ world)[:3]).astype(int)
    inside = np.all((v >= 0) & (v < np.array(data.shape[:3])[:, None]), axis=0)
    res = np.zeros(idx.shape[1], dtype=np.float32)
    res[inside] = data[v[0, inside], v[1, inside], v[2, inside]]
    return res.reshape(grid.shape[:3])


def axial(vol: np.ndarray, k: int) -> np.ndarray:
    """Axial slice of a RAS-ordered volume, anterior at the top (radiological left-right is not applied)."""
    return vol[:, :, k].T[::-1]


def png(path: Path, img: np.ndarray) -> None:
    """Minimal 8-bit greyscale PNG writer (no imaging dependency)."""
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].astype(np.uint8).tobytes() for r in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def tile(sl: np.ndarray, size: int = 160) -> np.ndarray:
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99.5)
    g = np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1) * 255
    ri = (np.arange(size) * sl.shape[0] / size).astype(int)
    ci = (np.arange(size) * sl.shape[1] / size).astype(int)
    return g[np.ix_(ri, ci)]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    summary = {"bug": "R01_swap_transpose: reorients to RAS, then swaps array axes 0 and 1 without updating the affine",
               "contract": CONTRACT.to_dict(), "code": {k: str(v.relative_to(ROOT)).replace("\\", "/") for k, v in CODE.items()}}
    rows = []

    # phantom
    ph = make_phantom(1)
    pdir = OUT / "phantom_seed1"
    pdir.mkdir(parents=True)
    ph.save(pdir / "input.nii.gz")
    summary["phantom_seed1"] = {"input_axcodes": "".join(nib.aff2axcodes(ph.affine)),
                                "ground_truth_ras_mm": {str(k): [round(float(x), 2) for x in v] for k, v in ph.world_ras_mm.items()}}
    prow = []
    for name, code in CODE.items():
        run(code, pdir / "input.nii.gz", pdir / f"{name}_output.nii.gz")
        r = oracle.measure(pdir / f"{name}_output.nii.gz", CONTRACT, ph)
        summary["phantom_seed1"][name] = {"status": r.status, "out_axcodes": r.out_axcodes,
                                          "landmark_error_mm": {str(lm.label): None if lm.error_mm is None else round(lm.error_mm, 2)
                                                                for lm in r.landmarks}}
        prow.append(pdir / f"{name}_output.nii.gz")
    rows.append((pdir, prow))

    # real T1
    name0, img = realscan.scans()[0]
    tdir = OUT / "real_t1"
    tdir.mkdir(parents=True)
    nib.save(img, str(tdir / "input.nii.gz"))
    summary["real_t1"] = {"scan": name0, "source": "nibabel/tests/data/anatomical.nii", "input_axcodes": "".join(nib.aff2axcodes(img.affine))}
    trow = []
    for name, code in CODE.items():
        c = realscan.consistency(code, CONTRACT, img)
        run(code, tdir / "input.nii.gz", tdir / f"{name}_output.nii.gz")
        summary["real_t1"][name] = {"correlation_with_input": c.get("r"), "covered_fraction": c.get("covered_fraction"),
                                    "out_axcodes": c.get("out_axcodes")}
        trow.append(tdir / f"{name}_output.nii.gz")
    rows.append((tdir, trow))

    # preview: columns input | buggy | repaired | reference, rows phantom | T1; every volume sampled on the input's closest-RAS grid
    # through its own affine, so an output whose array and affine disagree shows up misplaced
    lines = []
    for d, outs in rows:
        grid = nib.as_closest_canonical(nib.load(str(d / "input.nii.gz")))
        vols = [on_grid(np.asarray(nib.load(str(f)).dataobj, dtype=np.float32), nib.load(str(f)).affine, grid)
                for f in [d / "input.nii.gz"] + outs]
        k = int(np.argmax((vols[0] > 0).sum(axis=(0, 1)))) if d.name.startswith("phantom") else grid.shape[2] // 2
        lines.append(np.hstack([np.pad(tile(axial(v, k)), 4, constant_values=255) for v in vols]))
    png(OUT / "preview.png", np.vstack(lines))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("phantom_seed1", "real_t1")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
