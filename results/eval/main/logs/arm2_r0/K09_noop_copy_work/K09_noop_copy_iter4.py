import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    ras_img = nib.as_closest_canonical(img)
    # Ensure we have a separate object if already canonical to avoid modifying input
    if ras_img is img:
        ras_img = ras_img.copy()
    data = np.asarray(ras_img.dataobj)
    aff = ras_img.affine
    nib.save(nib.Nifti1Image(data, aff), out_path)