# seeded case M01_zoom_only: converts the rotation/zoom block to metres but leaves the origin in mm
import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    aff = img.affine.copy()
    aff[:3, :3] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)
