import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert to closest canonical (usually RAS) orientation
    canonical_img = nib.as_closest_canonical(img)
    nib.save(canonical_img, out_path)