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
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Determine current orientation (axcodes)
    cur_axcodes = nib.aff2axcodes(aff)
    target_axcodes = ('R', 'A', 'S')
    if cur_axcodes != target_axcodes:
        # Convert to RAS orientation
        cur_ornt = nib.orientations.axcodes2ornt(cur_axcodes)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(cur_ornt, target_ornt)
        data = nib.orientations.apply_orientation(data, transform)
        aff = nib.orientations.apply_orientation(aff, transform)
    # Save
    nib.save(nib.Nifti1Image(data, aff), out_path)
