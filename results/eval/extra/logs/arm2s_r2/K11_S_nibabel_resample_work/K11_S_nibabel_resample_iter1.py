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
    resampled = nproc.resample_to_output(img, 2.0, order=0)
    # Preserve original sform/qform to keep spatial location
    resampled.set_sform(img.get_sform(), code=img.get_sform_code())
    resampled.set_qform(img.get_qform(), code=img.get_qform_code())
    nib.save(resampled, out_path)