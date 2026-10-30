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
    vox = 2.0
    # physical size of the input image
    voxel_sizes = np.array(img.header.get_zooms()[:3])
    world_size = np.array(data.shape) * voxel_sizes
    # output shape to cover same physical extent with isotropic voxels
    shape = np.ceil(world_size / vox).astype(int)
    # origin of the input image in world coordinates (voxel (0,0,0))
    origin = aff[:3, 3]
    # output affine: isotropic voxels aligned with world axes, same origin
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = origin
    # generate output voxel indices
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    # world coordinates of output voxel centers
    world_pts = out_aff @ out_idx
    # map world points back to input voxel indices (continuous)
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    # keep only those that fall inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)
