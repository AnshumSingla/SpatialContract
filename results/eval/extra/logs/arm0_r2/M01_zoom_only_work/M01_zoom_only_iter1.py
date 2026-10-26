import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to RAS+ (right-anterior-superior) to satisfy world_frame RAS
    img = nib.as_closest_canonical(img)
    aff = img.affine.copy()
    # Convert from mm to m (contract units)
    aff[:3, :3] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)