import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to RAS (closest canonical) which ensures voxel axes RAS and affine voxel_center_to_world
    canonical = nib.as_closest_canonical(img)
    nib.save(canonical, out_path)