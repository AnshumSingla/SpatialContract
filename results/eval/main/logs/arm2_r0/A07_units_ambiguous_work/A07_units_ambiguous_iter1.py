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
    aff = img.affine.copy()
    # Convert voxel size from metres to millimetres (scale linear part)
    aff[:3, :3] *= 0.001
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)