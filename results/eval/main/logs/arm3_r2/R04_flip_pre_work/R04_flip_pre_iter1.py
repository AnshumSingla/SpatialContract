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
    # Reorient to RAS (right-anterior-superior) using nibabel's orientation tools
    ras_img = nib.as_closest_canonical(img)  # ensures RAS orientation
    nib.save(ras_img, out_path)