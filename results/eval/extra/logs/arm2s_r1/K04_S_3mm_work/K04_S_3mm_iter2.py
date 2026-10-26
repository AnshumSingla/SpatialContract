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
    vox = 3.0

    # World coordinates of the 8 corners of the input voxel grid
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)

    # Unit direction vectors of input axes (preserve orientation)
    scales = np.linalg.norm(aff[:3, :3], axis=0)
    unit_dirs = aff[:3, :3] / scales  # 3x3, columns are unit vectors

    # Output affine: same orientation, isotropic voxel size
    M_out = unit_dirs * vox  # 3x3

    # Determine translation so that the grid's first voxel centre aligns
    # with the bounding box such that the whole bbox is covered.
    # Solve for t_out: world = M_out @ voxel_index + t_out
    # For each corner compute continuous voxel coordinates assuming t_out = lo
    inv_M_out = np.linalg.inv(M_out)
    v_cont = (inv_M_out @ (world - lo).T).T  # Nx3
    min_v = np.floor(v_cont.min(axis=0))
    max_v = np.ceil(v_cont.max(axis=0))
    t_out = lo - M_out @ min_v
    shape = (max_v - min_v + 1).astype(int)

    # Resample
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = M_out @ out_idx[:3] + t_out[:, None]
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape),
                             np.vstack([np.hstack([M_out, t_out[:, None]]),
                                        [0, 0, 0, 1]])), out_path)