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

    # Extract linear part and compute unit direction vectors of input axes
    R = aff[:3, :3]
    scales = np.linalg.norm(R, axis=0)
    unit_dirs = R / scales  # 3x3, columns are unit vectors preserving orientation

    # Output affine: same orientation, isotropic voxel size
    M_out = unit_dirs * vox  # 3x3
    out_aff = np.vstack([np.hstack([M_out, lo[:, None]]), [0, 0, 0, 1]])

    # Resample using nearest neighbor
    ii, jj, kk = np.indices(
        np.ceil((hi - lo) / vox).astype(int) + 1
    )
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx  # 4xN
    world_pts = world_pts[:3, :]   # 3xN
    in_idx = np.rint(np.linalg.inv(aff) @ np.vstack([world_pts, np.ones_like(world_pts[0:1, :])]))[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(out_idx.shape[1:])), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(ii.shape), out_aff), out_path)