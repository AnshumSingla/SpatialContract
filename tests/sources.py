"""Reference pipeline (reorient any NIfTI to RAS+) and one-line mutations for each bug class."""

HEADER = '''import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS",
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))
    new = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
'''
SAVE = "    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n"

CORRECT = HEADER + SAVE


def _with(extra_before_save="", save=SAVE):
    return HEADER + extra_before_save + save


BUGS = {
    "axis_swap": _with("    new = new.transpose(1, 0, 2)\n"),
    "flipped_axis": _with("    new = new[::-1]\n"),
    "stale_affine": _with(save="    nib.save(nib.Nifti1Image(new, aff), out_path)\n"),
    "voxel_vs_world": _with(save="    nib.save(nib.Nifti1Image(new, np.eye(4)), out_path)\n"),
    "wrong_units": _with("    new_aff = new_aff.copy(); new_aff[:3, :] *= 1e-3\n"),
    "transposed_affine": _with(save="    nib.save(nib.Nifti1Image(new, new_aff.T), out_path)\n"),
}
