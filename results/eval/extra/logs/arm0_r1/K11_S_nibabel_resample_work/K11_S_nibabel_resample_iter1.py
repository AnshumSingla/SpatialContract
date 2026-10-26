import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    import nibabel.processing as nproc
    img = nib.load(in_path)
    # Reorient image to RAS+ if needed
    current_axcodes = nib.aff2axcodes(img.affine)
    target_axcodes = ('R', 'A', 'S')
    if current_axcodes != target_axcodes:
        current_ornt = nib.orientations.axcodes2ornt(current_axcodes)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
        img = img.apply_orient(transform)
    # Resample to isotropic 2.0 mm voxels
    resampled = nproc.resample_to_output(img, 2.0, order=0)
    nib.save(resampled, out_path)
