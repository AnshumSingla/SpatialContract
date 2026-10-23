# seeded case K11_S_nibabel_resample: nibabel.processing.resample_to_output(order=0) at 2 mm
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
    nib.save(nproc.resample_to_output(img, 2.0, order=0), out_path)
