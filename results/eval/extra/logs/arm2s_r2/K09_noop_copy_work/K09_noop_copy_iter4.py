import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to closest canonical (RAS) orientation
    canon_img = nib.as_closest_canonical(img)
    # Obtain a copy of the data and the affine
    data = canon_img.get_fdata()
    aff = canon_img.affine
    nib.save(nib.Nifti1Image(data, aff), out_path)