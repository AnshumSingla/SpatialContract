import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert to closest canonical (usually RAS)
    canon = nib.as_closest_canonical(img)
    # Determine current orientation codes
    cur_codes = ''.join(o.aff2axcodes(canon.affine))
    target_codes = 'RAS'
    if cur_codes != target_codes:
        # Build transformation to target orientation
        orient = o.ornt_transform(o.ax2ornt(cur_codes), o.ax2ornt(target_codes))
        # Apply orientation change
        data = canon.get_fdata()
        data = o.apply_orient(data, orient)
        # New affine
        new_affine = o.apply_orient(canon.affine, orient)
        canon = nib.Nifti1Image(data, new_affine, canon.header)
    nib.save(canon, out_path)
