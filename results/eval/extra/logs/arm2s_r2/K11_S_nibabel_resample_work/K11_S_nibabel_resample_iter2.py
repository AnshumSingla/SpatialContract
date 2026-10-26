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
    sform, scode = img.get_sform()
    qform, qcode = img.get_qform()
    resampled.set_sform(sform, code=scode)
    resampled.set_qform(qform, code=qcode)
    nib.save(resampled, out_path)