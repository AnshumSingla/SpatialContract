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
    data = img.get_fdata()
    aff = img.affine
    # Determine current axis codes (RAS+)
    cur_codes = nib.aff2axcodes(aff)
    target_codes = ('L', 'P', 'S')
    if cur_codes != target_codes:
        # Transform to LPS orientation
        ornt_trans = o.axcodes2ornt(cur_codes, target_codes)
        data = o.apply_orientation(data, ornt_trans)
        aff = o.inv_ornt_aff(ornt_trans, data.shape) @ aff
    # Save with LPS affine
    nib.save(nib.Nifti1Image(data, aff), out_path)
