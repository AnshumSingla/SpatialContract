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
    cur_codes = o.aff2axcodes(img.affine)
    desired_codes = ('R', 'A', 'S')
    if cur_codes != desired_codes:
        ornt = o.axcodes2ornt(cur_codes, desired_codes)
        img = img.as_reoriented(ornt)
    nib.save(img, out_path)