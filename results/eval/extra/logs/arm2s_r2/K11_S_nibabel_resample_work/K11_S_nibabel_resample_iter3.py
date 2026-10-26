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
    # Safely extract sform and qform (affine, code)
    sform_info = img.get_sform()
    qform_info = img.get_qform()
    if isinstance(sform_info, (tuple, list)) and len(sform_info) >= 2:
        sform, scode = sform_info[0], sform_info[1]
    else:
        sform, scode = sform_info, None
    if isinstance(qform_info, (tuple, list)) and len(qform_info) >= 2:
        qform, qcode = qform_info[0], qform_info[1]
    else:
        qform, qcode = qform_info, None
    # Set them on the resampled image
    resampled.set_sform(sform, code=scode)
    resampled.set_qform(qform, code=qcode)
    nib.save(resampled, out_path)