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

    # Get the input orientation in voxel->world order (axcodes)
    io = o.io_orientation(aff)
    # Target orientation: RAS (Right, Anterior, Superior)
    target_ornt = o.ornt_axcodes2ornt(("R", "A", "S"))
    # Transform the input orientation to the target orientation
    xfm = o.ornt_transform(io, target_ornt)

    # Apply orientation to the data array
    oriented_data = o.apply_orientation(data, xfm)

    # Build new affine preserving voxel_center_to_world semantics
    inv_ornt_aff = o.inv_ornt_aff(xfm, oriented_data.shape)
    new_affine = aff @ inv_ornt_aff

    nib.save(nib.Nifti1Image(oriented_data, new_affine), out_path)