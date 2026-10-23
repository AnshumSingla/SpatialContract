"""Repo-style tests: what arm 2 ("LLM + the repo's own tests") sees instead of the oracle.

Each of the five seeded pipelines gets 3-4 ordinary unit tests of the kind a maintainer writes: the output is a valid 3-D image,
a simple known input (a small non-cubic volume with a RAS-aligned affine) behaves as documented, shapes/values are preserved.
They run the pipeline's ``process`` on NON-phantom inputs and say nothing about physical landmark positions. They were written
from the pipelines' documented purpose, not tuned against the seeded bugs; ``scripts/evaluate.py`` reports which seeded bugs they
flag so that arm 2 can be read in that light.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import oracle


@dataclass
class RepoTestResult:
    passed: bool
    results: list = field(default_factory=list)      # (name, ok, message)

    def text(self) -> str:
        bad = [r for r in self.results if not r[1]]
        lines = [f"{'FAILED' if not ok else 'passed'} {name}" + (f" - {msg}" if msg else "") for name, ok, msg in self.results]
        lines.append(f"{len(bad)} failed, {len(self.results) - len(bad)} passed")
        return "\n".join(lines)


def _vol(affine, shape=(6, 7, 8), seed=0, block=None):
    rng = np.random.default_rng(seed)
    if block is None:
        data = rng.integers(1, 100, size=shape).astype(np.float32)
    else:
        data = np.zeros(shape, np.float32)
        data[block] = rng.integers(1, 100, size=data[block].shape)
    img = nib.Nifti1Image(data, np.asarray(affine, float))
    img.set_sform(np.asarray(affine, float), code=1)
    img.set_qform(np.asarray(affine, float), code=1)
    return img


RAS_AFF = np.array([[1.0, 0, 0, -5.0], [0, 2.0, 0, 10.0], [0, 0, 3.0, 20.0], [0, 0, 0, 1]])
LAS_AFF = np.array([[-1.0, 0, 0, 0.0], [0, 2.0, 0, 10.0], [0, 0, 3.0, 20.0], [0, 0, 0, 1]])


def _run(code: Path, img, tmp: Path, name: str):
    src, dst = tmp / f"{name}_in.nii.gz", tmp / f"{name}_out.nii.gz"
    nib.save(img, str(src))
    ok, err = oracle._run_code(code, src, dst, tmp)
    if not ok:
        raise AssertionError(f"pipeline crashed: {err}")
    return nib.load(str(dst))


def _zooms(aff):
    return np.sqrt((np.asarray(aff)[:3, :3] ** 2).sum(axis=0))


# ---- tests: each takes (code, tmp) and raises AssertionError on failure --------------------------------------------
def t_valid_image(code, tmp, spec):
    out = _run(code, _vol(RAS_AFF), tmp, "valid")
    assert np.asarray(out.dataobj).ndim == 3, f"output has {np.asarray(out.dataobj).ndim} dimensions"
    assert np.all(np.isfinite(out.affine)), "output affine contains non-finite values"
    assert np.all(_zooms(out.affine) > 0), f"non-positive voxel size {_zooms(out.affine)}"


def t_data_unchanged(code, tmp, spec):
    img = _vol(RAS_AFF)
    out = _run(code, img, tmp, "data")
    assert np.array_equal(np.asarray(out.dataobj), np.asarray(img.dataobj)), "voxel data differs from the input"


def r_ras_unchanged(code, tmp, spec):
    img = _vol(RAS_AFF)
    out = _run(code, img, tmp, "rasid")
    assert out.shape == img.shape, f"shape {out.shape} != {img.shape}"
    assert np.allclose(np.asarray(out.dataobj), np.asarray(img.dataobj)), "voxel data changed for an input that is already RAS"
    assert np.allclose(out.affine, img.affine), "affine changed for an input that is already RAS"


def r_values_preserved(code, tmp, spec):
    img = _vol(LAS_AFF, seed=1)
    out = _run(code, img, tmp, "vals")
    assert sorted(out.shape) == sorted(img.shape), f"shape {out.shape} is not a permutation of {img.shape}"
    assert np.allclose(np.sort(np.asarray(out.dataobj), axis=None), np.sort(np.asarray(img.dataobj), axis=None)), "voxel values not preserved"


def r_flipped_x_canonicalised(code, tmp, spec):
    img = _vol(LAS_AFF, seed=2)
    out = _run(code, img, tmp, "las")
    if not spec.get("canonicalise", True):          # a variant documented to leave the grid alone
        assert np.allclose(np.asarray(out.dataobj), np.asarray(img.dataobj)) and np.allclose(out.affine, img.affine), "input was modified"
        return
    assert "".join(nib.aff2axcodes(out.affine)) == "RAS", f"output orientation {''.join(nib.aff2axcodes(out.affine))}, expected RAS"
    assert np.allclose(np.asarray(out.dataobj), np.asarray(img.dataobj)[::-1]), "data is not the x-flipped input"


def s_voxel_size(code, tmp, spec):
    out = _run(code, _vol(RAS_AFF, shape=(11, 12, 13)), tmp, "vox")
    vox = spec.get("vox", 2.0)
    assert np.allclose(_zooms(out.affine), vox, atol=1e-3), f"voxel size {np.round(_zooms(out.affine), 3)} mm, expected {vox}"


def s_covers_extent(code, tmp, spec):
    img = _vol(RAS_AFF, shape=(11, 12, 13))
    out = _run(code, img, tmp, "ext")
    in_extent = (np.array(img.shape) - 1) * _zooms(img.affine)
    out_extent = (np.array(out.shape) - 1) * _zooms(out.affine)
    assert np.all(out_extent >= in_extent - 2.0) and np.all(out_extent <= in_extent + 2.0 * spec.get("vox", 2.0)), \
        f"output extent {np.round(out_extent, 1)} mm vs input {np.round(in_extent, 1)} mm"


def s_values_from_input(code, tmp, spec):
    img = _vol(RAS_AFF, shape=(11, 12, 13), seed=3)
    out = _run(code, img, tmp, "valsub")
    extra = set(np.unique(np.asarray(out.dataobj))) - set(np.unique(np.asarray(img.dataobj))) - {0.0}
    assert not extra, f"output contains values not present in the input: {sorted(extra)[:5]}"


BLOCK = (slice(4, 7), slice(5, 9), slice(6, 11))
C_SHAPE = (12, 13, 14)


def c_shape(code, tmp, spec):
    img = _vol(RAS_AFF, shape=C_SHAPE, block=BLOCK)
    out = _run(code, img, tmp, "cshape")
    m = spec.get("margin", 3)
    lo = np.maximum(np.array([4, 5, 6]) - m, 0)
    hi = np.minimum(np.array([6, 8, 10]) + 1 + m, C_SHAPE)
    assert out.shape == tuple(hi - lo), f"cropped shape {out.shape}, expected {tuple(hi - lo)} (foreground bounding box plus a margin of 3 voxels)"


def c_content(code, tmp, spec):
    img = _vol(RAS_AFF, shape=C_SHAPE, block=BLOCK)
    out = _run(code, img, tmp, "ccont")
    m = spec.get("margin", 3)
    lo = np.maximum(np.array([4, 5, 6]) - m, 0)
    hi = np.minimum(np.array([6, 8, 10]) + 1 + m, C_SHAPE)
    want = np.asarray(img.dataobj)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    assert out.shape == want.shape and np.allclose(np.asarray(out.dataobj), want), "cropped content differs from the expected sub-volume"


def c_not_larger(code, tmp, spec):
    img = _vol(RAS_AFF, shape=C_SHAPE, block=BLOCK)
    out = _run(code, img, tmp, "clarge")
    assert all(o <= i for o, i in zip(out.shape, img.shape)), f"output {out.shape} larger than input {img.shape}"


def l_zooms_preserved(code, tmp, spec):
    img = _vol(RAS_AFF)
    out = _run(code, img, tmp, "lz")
    assert np.allclose(_zooms(out.affine), _zooms(img.affine), atol=1e-4), f"voxel sizes {np.round(_zooms(out.affine), 3)} != {np.round(_zooms(img.affine), 3)}"


def m_zooms_in_metres(code, tmp, spec):
    img = _vol(RAS_AFF)
    out = _run(code, img, tmp, "mz")
    assert np.allclose(_zooms(out.affine), _zooms(img.affine) * 1e-3, rtol=1e-3), \
        f"voxel sizes {_zooms(out.affine)} m, expected {_zooms(img.affine) * 1e-3} m"


SUITES = {
    "R": [t_valid_image, r_ras_unchanged, r_values_preserved, r_flipped_x_canonicalised],
    "S": [t_valid_image, s_voxel_size, s_covers_extent, s_values_from_input],
    "C": [t_valid_image, c_shape, c_content, c_not_larger],
    "L": [t_valid_image, t_data_unchanged, l_zooms_preserved],
    "M": [t_valid_image, t_data_unchanged, m_zooms_in_metres],
}


def run_repo_tests(code_path, base: str, spec: dict | None = None) -> RepoTestResult:
    code = Path(code_path).resolve()
    res = RepoTestResult(True)
    with tempfile.TemporaryDirectory(prefix="sc_repotest_") as d:
        tmp = Path(d)
        for fn in SUITES[base]:
            try:
                fn(code, tmp, spec or {})
                res.results.append((fn.__name__, True, ""))
            except AssertionError as e:
                res.results.append((fn.__name__, False, str(e)[:240]))
            except Exception as e:      # harness problem, not a pipeline verdict
                res.results.append((fn.__name__, False, f"{type(e).__name__}: {str(e)[:200]}"))
    res.passed = all(ok for _, ok, _ in res.results)
    return res
