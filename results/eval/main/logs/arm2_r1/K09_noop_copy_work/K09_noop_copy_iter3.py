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
    # Work on copies to avoid altering the original image object
    data = img.get_fdata()  # returns a copy of the data
    aff = img.affine.copy()
    
    # Current orientation of the affine (voxel indices -> world)
    cur_codes = o.aff2axcodes(aff)
    desired_codes = ('R', 'A', 'S')
    
    if cur_codes != desired_codes:
        # Transform to RAS orientation
        transform = o.axcodes2ornt(cur_codes, desired_codes)
        data = o.apply_orientation(data, transform)
        aff = o.apply_orientation(aff, transform)
    
    nib.save(nib.Nifti1Image(data, aff), out_path)