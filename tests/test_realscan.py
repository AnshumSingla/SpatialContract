"""The real-scan check on nibabel's bundled real T1: a correct reorientation reproduces the anatomy exactly, a left-right mirror
and a stale affine do not."""
import pytest

from spatialcontract import realscan
from spatialcontract.contract import load_from_source

HEAD = '''SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}
import nibabel as nib, numpy as np
'''
CORRECT = HEAD + "def process(i, o):\n    nib.save(nib.as_closest_canonical(nib.load(i)), o)\n"
MIRROR = HEAD + ("def process(i, o):\n    img = nib.load(i)\n"
                 "    nib.save(nib.Nifti1Image(np.asarray(img.dataobj)[::-1].copy(), img.affine), o)\n")
STALE = HEAD + ("def process(i, o):\n    img = nib.load(i); c = nib.as_closest_canonical(img)\n"
                "    nib.save(nib.Nifti1Image(np.asarray(c.dataobj), img.affine), o)\n")


@pytest.fixture(scope="module")
def t1():
    return dict(realscan.scans())["t1_nibabel_LAS_2mm"]


@pytest.mark.parametrize("src,exact", [(CORRECT, True), (MIRROR, False), (STALE, False)])
def test_real_scan_consistency(tmp_path, t1, src, exact):
    p = tmp_path / "code.py"
    p.write_text(src)
    row = realscan.consistency(p, load_from_source(src), t1)
    assert (row["r"] is not None and row["r"] > 0.9999) is exact, row
