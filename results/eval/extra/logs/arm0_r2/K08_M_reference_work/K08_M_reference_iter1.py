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
    # Reorient to RAS
    current_ornt = o.io_orientation(img.affine)
    target_ornt = o.axcodes_to_orientation(('R', 'A', 'S'))
    transform = o.ornt_transform(current_ornt, target_ornt)
    img = img.apply_orient(transform)
    # Convert units from mm to m (assuming input is in mm)
    aff = img.affine.copy()
    aff[:3, :] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)
