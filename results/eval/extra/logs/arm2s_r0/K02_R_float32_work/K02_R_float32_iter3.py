import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    canonical = nib.as_closest_canonical(img)
    data = np.asarray(canonical.dataobj)
    aff = canonical.affine
    # Ensure contiguous array and correct dtype
    data = np.ascontiguousarray(data, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, aff), out_path)