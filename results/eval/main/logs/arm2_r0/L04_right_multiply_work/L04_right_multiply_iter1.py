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
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    new_affine = img.affine @ ras_to_lps
    # Preserve header information (sform/qform) but update affine
    new_img = nib.Nifti1Image(img.get_fdata(), new_affine, header=img.header)
    nib.save(new_img, out_path)