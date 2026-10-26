import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Ensure canonical RAS+ orientation
    canon = nib.as_closest_canonical(img)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    out_affine = canon.affine @ ras_to_lps
    nib.save(nib.Nifti1Image(np.asarray(canon.dataobj), out_affine), out_path)
