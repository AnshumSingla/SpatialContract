import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "m",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Reorient to canonical (usually RAS+) orientation
    canon_img = nib.as_closest_canonical(img)
    aff = canon_img.affine.copy()
    # Convert voxel size from millimetres to metres
    aff[:3, :3] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(canon_img.dataobj), aff), out_path)
