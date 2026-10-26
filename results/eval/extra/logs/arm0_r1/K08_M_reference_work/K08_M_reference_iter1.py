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
    # Reorient to RAS (nearest canonical)
    canonical_img = nib.as_closest_canonical(img)
    data = np.asarray(canonical_img.dataobj)
    aff = canonical_img.affine.copy()
    # Convert units from mm to m (assuming input affine is in mm)
    aff[:3, :] *= 1e-3
    nib.save(nib.Nifti1Image(data, aff), out_path)