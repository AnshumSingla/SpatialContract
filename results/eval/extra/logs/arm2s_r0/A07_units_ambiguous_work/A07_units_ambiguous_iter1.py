import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm/m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    aff = img.affine.copy() * 0.001  # convert from meters to millimeters
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)