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
    # Determine current orientation of the image (e.g., 'RAS', 'LPS')
    cur_codes = nib.aff2axcodes(img.affine)
    # Desired orientation is LPS
    target_codes = ('L', 'P', 'S')
    # If already in target orientation, keep as is
    if cur_codes == target_codes:
        out_img = img
    else:
        # Convert orientation
        cur_ornt = o.axcodes2ornt(cur_codes)
        target_ornt = o.axcodes2ornt(target_codes)
        transform = o.ornt_transform(cur_ornt, target_ornt)
        # Apply reorientation
        data = img.get_fdata()
        data = o.apply_orientation(data, transform)
        # New affine after reorientation
        new_affine = img.affine.copy()
        new_affine[:3, :3] = np.dot(new_affine[:3, :3], o.inv_ornt_aff(transform, data.shape))
        out_img = nib.Nifti1Image(data, new_affine, img.header)
    nib.save(out_img, out_path)