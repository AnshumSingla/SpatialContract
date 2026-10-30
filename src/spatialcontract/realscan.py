"""Check spatial code on REAL scans, where no landmark ground truth exists. Deterministic, no LLM.

The input scan is its own ground truth: a pipeline that only moves, crops, re-orients, resamples or re-expresses an image must
put the anatomy found at each physical point of the input back at that same physical point. The output (its affine read in
the contract's frame and units) is resampled onto the input grid with SimpleITK and compared with the input over the region it
covers: Pearson correlation of intensities, and the covered fraction of the input. This reads only physics, never the code.

Real brains are close to left-right symmetric, so a pure left-right mirror can keep a high correlation; that is the
reason the main oracle uses deliberately asymmetric phantoms, and the real-scan numbers are reported next to it, not instead.

Scans: two real scans that ship with every nibabel install (a T1 anatomical, LAS, 2 mm; the first volume of an EPI series,
LAS, 9.3 degrees oblique, 2x2x2.2 mm) and, when the ANTsPy checkout under _work/ is present, the Colin27 single-subject brain
downsampled to 2 mm, stored once as RAS and once in a sagittal-style PIR layout (same anatomy and world positions).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from . import oracle
from .contract import SpatialContract

ROOT = Path(__file__).resolve().parents[2]
COLIN = ROOT / "_work" / "repos" / "ANTsPy" / "data" / "ch2.nii.gz"


def _clean(data: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=1)
    return img


def scans() -> list[tuple[str, nib.Nifti1Image]]:
    """(name, image) for every real scan available here; the nibabel ones always are."""
    data_dir = Path(nib.__file__).parent / "tests" / "data"
    out = []
    anat = nib.load(str(data_dir / "anatomical.nii"))
    out.append(("t1_nibabel_LAS_2mm", _clean(np.asarray(anat.dataobj).astype(np.float32), anat.affine)))
    epi = nib.load(str(data_dir / "example4d.nii.gz"))
    out.append(("epi_nibabel_LAS_oblique", _clean(np.asarray(epi.dataobj)[..., 0].astype(np.float32), epi.affine)))
    if COLIN.exists():
        from nibabel.processing import resample_to_output
        colin = resample_to_output(nib.load(str(COLIN)), voxel_sizes=(2.0, 2.0, 2.0), order=1)
        data, aff = np.asarray(colin.dataobj).astype(np.float32), colin.affine
        out.append(("colin27_RAS_2mm", _clean(data, aff)))
        ornt = nib.orientations.ornt_transform(nib.orientations.io_orientation(aff), nib.orientations.axcodes2ornt(tuple("PIR")))
        out.append(("colin27_PIR_2mm", _clean(nib.orientations.apply_orientation(data, ornt),
                                              aff @ nib.orientations.inv_ornt_aff(ornt, data.shape))))
    return out


def consistency(code_path, contract: SpatialContract, img: nib.Nifti1Image, python=None, pythonpath=None) -> dict:
    """Run ``process`` on a real scan and compare the output with the input at the same physical points."""
    import SimpleITK as sitk

    with tempfile.TemporaryDirectory(prefix="sc_real_") as d:
        d = Path(d)
        src, out_p, ras_p = d / "scan.nii.gz", d / "out.nii.gz", d / "out_ras_mm.nii.gz"
        nib.save(img, str(src))
        ok, err = oracle._run_code(Path(code_path).resolve(), src, out_p, d, python, pythonpath)
        if not ok:
            return {"verdict": "ERROR", "detail": str(err)[:200]}
        try:
            out = nib.load(str(out_p))
            data = np.asarray(out.dataobj).astype(np.float32)
            if data.ndim != 3:
                return {"verdict": "ERROR", "detail": f"output has {data.ndim} dims"}
            aff = contract.contract_to_ras_affine_mm(np.asarray(out.affine, dtype=float))
            out_axcodes = "".join(nib.aff2axcodes(aff))
            nib.save(_clean(data, aff), str(ras_p))
            res = sitk.Resample(sitk.ReadImage(str(ras_p)), sitk.ReadImage(str(src)), sitk.Transform(), sitk.sitkLinear,
                                -1e9, sitk.sitkFloat32)
        except Exception as e:
            return {"verdict": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:150]}"}
    a = np.transpose(sitk.GetArrayFromImage(res), (2, 1, 0))       # back to (i, j, k)
    b = np.asarray(img.dataobj, dtype=np.float32)
    valid = a > -1e8
    row = {"covered_fraction": round(float(valid.mean()), 4), "out_axcodes": out_axcodes,
           "orientation_ok": None if not contract.array_orientation else out_axcodes == contract.array_orientation}
    if valid.sum() < 1000 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        row.update(r=None)
        return row
    row["r"] = round(float(np.corrcoef(a[valid], b[valid])[0, 1]), 4)
    return row
