import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Determine current orientation (voxel axes -> RAS/LPS etc.)
    orig_ornt = o.io_orientation(img.affine)
    # Desired orientation: RAS
    target_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    # Compute transform to RAS
    transform = o.ornt_transform(orig_ornt, target_ornt)
    # Apply orientation change to data
    data = o.apply_orientation(img.get_fdata(), transform)
    # Update affine to reflect new orientation
    aff = o.ornt_mapping.transform_affine(transform, img.shape) @ img.affine
    nib.save(nib.Nifti1Image(data, aff), out_path)