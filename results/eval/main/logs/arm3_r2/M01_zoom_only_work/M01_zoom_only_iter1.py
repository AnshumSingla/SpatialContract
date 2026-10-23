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
    # Reorient to RAS (canonical) orientation
    img_canonical = nib.as_closest_canonical(img)
    aff = img_canonical.affine.copy()
    # Convert from mm to m (scale entire affine)
    aff *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(img_canonical.dataobj), aff), out_path)