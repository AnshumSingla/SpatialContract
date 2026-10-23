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

    # Compute the full orientation matrix that aligns input to RAS
    # This includes any axis swaps caused by the original orientation labels
    target_ornt = o.axcodes2ornt(("R", "A", "S"))
    io = o.io_orientation(aff)
    xfm = o.ornt_transform(io, target_ornt)

    # Orient the data
    oriented_data = o.apply_orientation(data, xfm)

    # Convert the input affine to the same orientation space as the target
    # by applying the same orientation transform to the input orientation matrix
    # Build a full 4×4 orientation matrix (including swaps) from the input orientation
    # and compose it with the input affine.
    # The proper way: transform the input orientation into scanner order then
    # recompute the affine in that space.
    # Here we achieve this by converting the input affine to the target orientation
    # via the inverse of the same transform.
    aligned_affine = o.inv_ornt_aff(xfm, img.shape) @ aff @ xfm

    nib.save(nib.Nifti1Image(oriented_data, aligned_affine), out_path)