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
    # Get closest canonical orientation
    canonical = nib.as_closest_canonical(img)
    # Determine current orientation codes
    current = o.aff2axcodes(canonical.affine)
    target = ('R', 'A', 'S')
    if current != target:
        # Compute transformation from current to target
        transform = o.ornt_transform(o.axcodes2ornt(current), o.axcodes2ornt(target))
        # Apply reordering
        canonical = canonical.as_reordered(transform)
    nib.save(canonical, out_path)