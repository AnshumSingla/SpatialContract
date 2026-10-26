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
    # Reorient to RAS (right-anterior-superior) if needed
    ras_img = nib.as_closest_canonical(img)
    aff = ras_img.affine.copy()
    # Convert from mm to meters (assuming input affine is in mm)
    aff[:3, :] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(ras_img.dataobj), aff), out_path)