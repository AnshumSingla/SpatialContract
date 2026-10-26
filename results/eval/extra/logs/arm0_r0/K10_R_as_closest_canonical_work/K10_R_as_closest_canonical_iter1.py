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
    # Ensure output orientation is RAS
    orig_ornt = o.io_orientation(img.affine)
    target_ornt = o.axcodes2ornt('RAS')
    transform = o.ornt_transform(orig_ornt, target_ornt)
    img_reoriented = img.apply_transform(transform)
    nib.save(img_reoriented, out_path)