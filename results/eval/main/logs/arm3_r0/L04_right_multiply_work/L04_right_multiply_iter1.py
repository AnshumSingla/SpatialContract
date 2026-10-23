import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform, apply_orientation

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Determine current orientation of the voxel axes (in RAS+)
    src_ornt = axcodes2ornt(aff2axcodes(img.affine))
    # Desired orientation: LPS
    tgt_ornt = axcodes2ornt(('L', 'P', 'S'))
    # Compute transformation from source to target orientation
    transform = ornt_transform(src_ornt, tgt_ornt)
    # Apply orientation transformation to the data
    data = apply_orientation(img.get_fdata(), transform)
    # Compute new affine that maps voxel indices to world coordinates in LPS
    new_affine = nib.ornt_inv(transform) @ img.affine
    # Save the reoriented image
    nib.save(nib.Nifti1Image(data, new_affine), out_path)
