# seeded case L02_rotation_only: flips the rotation block but leaves the origin in RAS
import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "LPS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    a = img.affine.copy()
    a[:3, :3] = ras_to_lps[:3, :3] @ a[:3, :3]
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), a), out_path)
