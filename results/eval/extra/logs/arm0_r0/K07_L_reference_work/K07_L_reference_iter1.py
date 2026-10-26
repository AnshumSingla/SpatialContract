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
    # Determine current orientation of voxel axes (i,j,k) in RAS
    cur_ornt = o.io_orientation(img.affine)
    # Desired orientation: LPS corresponds to axcodes L,P,S
    target_ornt = o.axcodes2ornt(('L', 'P', 'S'))
    # Compute transform to reorient data to target
    transform = o.ornt_transform(cur_ornt, target_ornt)
    # Apply transformation
    data = img.get_fdata()
    data = o.apply_orientation(data, transform)
    # New affine after reorientation
    affine = o.apply_orientation(img.affine, transform)
    # Save
    nib.save(nib.Nifti1Image(data, affine), out_path)
