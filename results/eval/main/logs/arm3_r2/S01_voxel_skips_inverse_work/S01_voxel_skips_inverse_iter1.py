import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    # Load input image
    img = nib.load(in_path)
    # Reorient to RAS+ (closest canonical) while preserving voxel size
    ras_img = nib.as_closest_canonical(img)
    # Save the reoriented image
    nib.save(ras_img, out_path)
