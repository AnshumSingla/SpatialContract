import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    aff = img.affine.copy()
    aff *= 1e-3  # Convert entire affine from mm to meters (scale and translation)
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)