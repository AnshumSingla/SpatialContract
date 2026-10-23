import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    # Convert to RAS orientation (closest canonical)
    ras_img = nib.as_closest_canonical(img)
    data = np.asarray(ras_img.dataobj)
    aff = ras_img.affine
    nib.save(nib.Nifti1Image(data, aff), out_path)