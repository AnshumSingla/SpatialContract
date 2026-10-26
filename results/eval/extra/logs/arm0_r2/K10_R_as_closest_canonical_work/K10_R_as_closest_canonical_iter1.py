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
    # Determine current orientation (as ras codes)
    current_orient = io_orientation(img.affine)
    # Desired orientation is RAS
    ras_orient = axcodes2oriento('RAS')
    # Compute transform to RAS
    transform = o.ornt_transform(current_orient, ras_orient)
    # Apply reorientation
    img_reoriented = img.apply_transform(transform)
    nib.save(img_reoriented, out_path)
