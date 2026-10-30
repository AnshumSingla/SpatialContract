"""Export the real-brain 3D view for the demo page: demo/brain3d.json. No tokens.

The Colin27 brain (scripts/fetch_colin27.py), stored PIR at 2 mm, is run through the seeded bug R01_swap_transpose and through the
repair Nemotron wrote for it in the recorded main run (the same pair as samples/). For the input and each output, the three
orthogonal slices through the middle of the array (skipping background) are exported with their intensities and the file's own affine;
the page places every voxel in world RAS mm through that affine, so an output whose array and affine disagree lands somewhere else.
Correlation with the input at the same physical points is realscan.consistency, as in validate_real_scans.py.

  python scripts/fetch_colin27.py && python scripts/export_brain3d.py"""
import base64
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from export_samples import CODE, CONTRACT, run  # noqa: E402
from spatialcontract import realscan  # noqa: E402

SCAN = "colin27_PIR_2mm"


def planes(img: nib.Nifti1Image, lo: float, hi: float) -> dict:
    """The central axial/coronal/sagittal planes of the array as voxel indices (uint8) and intensities (0-255, scaled by lo..hi)."""
    data = np.asarray(img.dataobj, dtype=np.float32)
    idx = np.indices(data.shape).reshape(3, -1).T
    mid = np.array(data.shape) // 2
    val = np.clip((data.reshape(-1) - lo) / (hi - lo), 0, 1) * 255
    keep = np.any(idx == mid, axis=1) & (val > 18)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731
    return {"affine": np.round(img.affine, 4).tolist(), "idx": b64(idx[keep].astype(np.uint8)), "val": b64(val[keep].astype(np.uint8))}


def main() -> int:
    scans = dict(realscan.scans())
    if SCAN not in scans:
        print("Colin27 is missing: run python scripts/fetch_colin27.py first", file=sys.stderr)
        return 1
    img = scans[SCAN]
    out = {"scan": "Colin27 (MNI single-subject T1), stored PIR at 2 mm",
           "bug": "R01_swap_transpose: reorients to RAS, then swaps array axes 0 and 1 without updating the affine",
           "code": {k: str(CODE[k].relative_to(ROOT)).replace("\\", "/") for k in ("buggy", "repaired")},
           "clouds": {}}
    with tempfile.TemporaryDirectory(prefix="sc_brain_") as d:
        src = Path(d) / "input.nii.gz"
        nib.save(img, str(src))
        ref = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = np.percentile(ref, 1), np.percentile(ref, 99.5)
        out["clouds"]["input"] = {"axcodes": "".join(nib.aff2axcodes(img.affine)), **planes(img, lo, hi)}
        for name in ("buggy", "repaired"):
            dst = Path(d) / f"{name}.nii.gz"
            run(CODE[name], src, dst)
            o = nib.load(str(dst))
            c = realscan.consistency(CODE[name], CONTRACT, img)
            out["clouds"][name] = {"axcodes": "".join(nib.aff2axcodes(o.affine)), "r": None if c.get("r") is None else round(c["r"], 4),
                                   **planes(o, lo, hi)}
    path = ROOT / "demo" / "brain3d.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("wrote", path, f"{path.stat().st_size // 1024} KB;",
          {k: (v["axcodes"], v.get("r")) for k, v in out["clouds"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
