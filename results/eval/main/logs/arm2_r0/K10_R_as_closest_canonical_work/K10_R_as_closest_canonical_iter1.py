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
    canonical = nib.as_closest_canonical(img)
    # Determine current orientation codes
    cur_codes = nib.aff2axcodes(canonical.affine)
    target_codes = ('R', 'A', 'S')
    if cur_codes != target_codes:
        # Compute transformation to target orientation
        trans = o.ornt_transform(o.axcodes2ornt(cur_codes),
                                 o.axcodes2ornt(target_codes))
        # Apply orientation change
        data = canonical.get_fdata()
        data = o.apply_orientation(data, trans)
        # Build new affine
        new_affine = o.inv_ornt_aff(trans, canonical.shape) @ canonical.affine
        canonical = nib.Nifti1Image(data, new_affine, canonical.header)
    nib.save(canonical, out_path)
