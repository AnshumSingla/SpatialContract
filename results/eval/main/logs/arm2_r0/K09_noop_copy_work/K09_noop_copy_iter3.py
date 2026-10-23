import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    aff = img.affine
    # Check if already in RAS orientation
    cur_codes = o.aff2axcodes(aff)
    ras_codes = ('R', 'A', 'S')
    if cur_codes == ras_codes:
        data = np.asarray(img.dataobj)
        new_aff = aff
    else:
        # Compute orientation transform to RAS
        orient = o.io_orientation(aff)
        target_ornt = o.axcodes2ornt(ras_codes)
        transform = o.ornt_transform(orient, target_ornt)
        data = o.apply_orientation(np.asarray(img.dataobj), transform)
        new_aff = o.apply_orientation(aff, transform)
    nib.save(nib.Nifti1Image(data, new_aff), out_path)