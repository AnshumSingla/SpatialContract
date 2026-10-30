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
    vox = 2.0
    # Output affine: isotropic voxels, same world origin as input
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = aff[:3, 3]
    # Determine output shape covering the input's bounding box in world space
    # Transform the 8 corners of the input voxel grid to world coordinates
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)
    # Number of voxels needed in each dimension to cover [lo, hi] with spacing vox
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    # Adjust output affine so its first voxel corner aligns with lo
    out_aff[:3, 3] = lo
    # Map output voxel indices to input voxel indices
    # M converts output voxel (homogeneous) -> input voxel (homogeneous)
    M = np.linalg.inv(aff) @ out_aff
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T  # 4 x N
    in_idx_homog = M @ out_idx
    in_idx = np.rint(in_idx_homog[:3]).astype(int)
    # Keep only those that map inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    lin_idx = (in_idx[0, ok] +
               in_idx[1, ok] * data.shape[0] +
               in_idx[2, ok] * data.shape[0] * data.shape[1])
    out[ok] = data.flat[lin_idx]
    nib.save(nib.Nifti1Image(out.reshape(shape, order='F'), out_aff), out_path)
