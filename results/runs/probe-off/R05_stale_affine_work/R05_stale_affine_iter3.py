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

    # Determine input orientation and transform it to RAS target orientation
    io = o.io_orientation(aff)
    target_ornt = o.axcodes2ornt(("R", "A", "S"))
    xfm = o.ornt_transform(io, target_ornt)

    # Apply orientation to the data
    oriented_data = o.apply_orientation(data, xfm)

    # Build the new affine so that voxel centre stays in the same world coordinate system
    inv_ornt_aff = o.inv_ornt_aff(xfm, oriented_data.shape)
    new_affine = aff @ inv_ornt_aff

    nib.save(nib.Nifti1Image(oriented_data, new_affine), out_path)