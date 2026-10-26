import numpy as np
import nibabel as nib
from nibabel import orientations as o

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world",
    "array_orientation": "RAS"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Determine current orientation of voxel axes (i,j,k) in RAS terms
    src_ornt = o.io_orientation(aff)
    # Target orientation: RAS
    tgt_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    # Transform to bring data to RAS orientation
    xfm = o.ornt_transform(src_ornt, tgt_ornt)
    new_data = o.apply_orientation(data, xfm)
    # Update affine to reflect new voxel axes while keeping world mapping consistent
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    # Save with original header (which does not store affine)
    out_img = nib.Nifti1Image(new_data, new_aff, header=img.header)
    nib.save(out_img, out_path)