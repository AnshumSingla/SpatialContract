"""Held-out validation of a patched pipeline. The repair agent never sees any of this.

A patch that passes the visible oracle (phantom seeds 0-2, whose failures are shown to the model) is then run on:
  hidden_bank   seven phantoms with seeds the model never saw (5 ordinary, 2 strongly oblique)
  per_run       one phantom whose seed is derived from the run seed AND the patched source, i.e. generated after the patch
  paired        the visible phantoms re-stored in a different valid orientation (same physical content)
  differential  a NON-phantom volume (linear intensity ramp over world coordinates) pushed through the patch, the output
                resampled onto the input grid with SimpleITK (a reader/resampler independent of nibabel) and compared voxelwise
A patch that passes the visible oracle but fails any of these counts as overfit.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from . import oracle
from .contract import SpatialContract
from .phantom import make_paired, make_phantom

HIDDEN_SEEDS = (1000, 1001, 1002, 1003, 1004, 250, 251)
PAIRED_BASE_SEEDS = (0, 1, 2)
RAMP_SEED = 9000
RAMP_SLOPES = np.array([0.9, 1.7, 2.9])   # intensity units per mm along world R, A, S
MIN_VALID_FRACTION = 0.3
BORDERS = (5, 8, 11)   # voxels of zero background around the ramp, per axis


def per_run_seed(run_seed: int, patched_source: str) -> int:
    h = hashlib.sha256(f"{run_seed}:{patched_source}".encode()).hexdigest()
    return 20_000 + int(h[:8], 16) % 100_000      # outside every fixed bank (0-2, 200-299, 1000-1004)


def _contract_to_ras_affine_mm(contract: SpatialContract, aff: np.ndarray) -> np.ndarray:
    return contract.contract_to_ras_affine_mm(aff)


def ramp_differential(code_path, contract: SpatialContract, python=None, pythonpath=None) -> dict:
    import SimpleITK as sitk

    ph = make_phantom(RAMP_SEED)
    ijk = np.stack(np.indices(ph.data.shape), axis=-1).reshape(-1, 3).astype(float)
    world = ijk @ ph.affine[:3, :3].T + ph.affine[:3, 3]
    ramp = (1000.0 + world @ RAMP_SLOPES).reshape(ph.data.shape).astype(np.float32)
    interior = np.zeros(ph.data.shape, bool)
    b0, b1, b2 = BORDERS
    ramp[:b0] = ramp[-b0:] = 0.0                   # zero border, different on every axis (so axis-order bugs in a crop show)
    ramp[:, :b1] = ramp[:, -b1:] = 0.0
    ramp[:, :, :b2] = ramp[:, :, -b2:] = 0.0
    interior[b0 + 2:-b0 - 2, b1 + 2:-b1 - 2, b2 + 2:-b2 - 2] = True
    slope_norm = float(np.linalg.norm(RAMP_SLOPES))
    with tempfile.TemporaryDirectory(prefix="sc_diff_") as d:
        d = Path(d)
        src, out_p, ref_p = d / "ramp_in.nii.gz", d / "ramp_out.nii.gz", d / "ramp_out_ras_mm.nii.gz"
        img = nib.Nifti1Image(ramp, ph.affine)
        img.set_sform(ph.affine, code=1)
        img.set_qform(ph.affine, code=1)
        nib.save(img, str(src))
        ok, err = oracle._run_code(Path(code_path).resolve(), src, out_p, d, python, pythonpath)
        if not ok:
            return {"verdict": "ERROR", "detail": str(err)}
        try:
            out = nib.load(str(out_p))
            data = np.asarray(out.dataobj).astype(np.float32)
            aff_ras = _contract_to_ras_affine_mm(contract, np.asarray(out.affine, dtype=float))
            fixed = nib.Nifti1Image(data, aff_ras)
            fixed.set_sform(aff_ras, code=1)
            fixed.set_qform(aff_ras, code=1)
            nib.save(fixed, str(ref_p))
            zooms_out = np.sqrt((aff_ras[:3, :3] ** 2).sum(axis=0))
            zooms_in = np.sqrt((ph.affine[:3, :3] ** 2).sum(axis=0))
            inp_s, out_s = sitk.ReadImage(str(src)), sitk.ReadImage(str(ref_p))
            res = sitk.Resample(out_s, inp_s, sitk.Transform(), sitk.sitkLinear, -9999.0, sitk.sitkFloat32)
        except Exception as e:
            return {"verdict": "ERROR", "detail": f"{type(e).__name__}: {str(e)[:150]}"}
    a = sitk.GetArrayFromImage(res)
    b = sitk.GetArrayFromImage(inp_s)
    valid = (a > -9000) & np.transpose(interior, (2, 1, 0))     # sitk arrays are (k, j, i)
    frac = float(valid.sum() / np.transpose(interior, (2, 1, 0)).sum())
    lossless = bool(np.allclose(np.sort(zooms_out), np.sort(zooms_in), atol=1e-3))
    tol = oracle.LOSSLESS_TOL_MM if lossless else min(max(oracle.DEFAULT_TOL_MM, oracle.HALF_VOXEL_DIAG * zooms_out.max()),
                                                      oracle.MAX_TOL_MM)
    if frac < MIN_VALID_FRACTION:
        return {"verdict": "FAIL", "valid_fraction": round(frac, 3), "tol_mm": tol, "detail": "output does not cover the input content"}
    p90 = float(np.percentile(np.abs(a[valid] - b[valid]), 90)) / slope_norm
    return {"verdict": "PASS" if p90 <= tol else "FAIL", "p90_equiv_mm": round(p90, 4), "tol_mm": round(tol, 3),
            "valid_fraction": round(frac, 3), "lossless": lossless}


def validate(code_path, contract: SpatialContract, run_seed: int = 0, python=None, pythonpath=None) -> dict:
    code_path = Path(code_path)
    kw = dict(python=python, pythonpath=pythonpath)
    src = code_path.read_text(encoding="utf-8")
    parts: dict = {}
    parts["hidden_bank"] = oracle.check(code_path, contract, phantoms=[make_phantom(s) for s in HIDDEN_SEEDS], **kw)
    parts["per_run"] = oracle.check(code_path, contract, phantoms=[make_phantom(per_run_seed(run_seed, src))], **kw)
    parts["paired"] = oracle.check(code_path, contract,
                                   phantoms=[make_paired(make_phantom(s), k) for s in PAIRED_BASE_SEEDS for k in (0, 1)], **kw)
    diff = ramp_differential(code_path, contract, **kw)
    summary = {name: rep.verdict for name, rep in parts.items()} | {"differential": diff["verdict"]}
    return {"passed": all(v == "PASS" for v in summary.values()), "summary": summary, "differential": diff,
            "per_run_seed": per_run_seed(run_seed, src),
            "failing_detail": {name: rep.feedback(max_lines=6) for name, rep in parts.items() if rep.verdict != "PASS"}}
