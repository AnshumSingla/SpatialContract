import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Current orientation of voxel axes (as RAS+ codes)
    src_ornt = axcodes2ornt(aff2axcodes(img.affine))
    # Desired orientation: LPS
    tgt_ornt = axcodes2ornt(('L', 'P', 'S'))
    # Transformation from source to target orientation
    transform = ornt_transform(src_ornt, tgt_ornt)
    # Apply orientation change (reorders data and updates affine)
    img_reoriented = img.apply_orientation(transform)
    nib.save(img_reoriented, out_path)
