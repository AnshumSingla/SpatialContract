import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    # Voxel size along each axis (norm of first three columns)
    voxel_sizes = np.sqrt(np.sum(aff[:3, :3] ** 2, axis=0))
    # Direction unit vectors
    dirs = aff[:3, :3] / voxel_sizes
    vox = 3.0
    # Output affine: same orientation, isotropic voxel size
    out_aff = np.eye(4)
    out_aff[:3, :3] = dirs * vox
    # Bounding box of input in world space
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    # Align output origin to lo (first voxel corner)
    out_aff[:3, 3] = lo
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    # Resample using nearest neighbor
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    valid = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[valid] = data[in_idx[0, valid], in_idx[1, valid], in_idx[2, valid]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)
