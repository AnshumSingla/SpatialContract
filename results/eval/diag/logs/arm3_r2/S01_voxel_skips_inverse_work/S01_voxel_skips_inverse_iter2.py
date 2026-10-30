import numpy as np
import nibabel as nib

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    # Load and reorient to RAS (right-anterior-superior)
    img = nib.load(in_path)
    ras_img = nib.as_closest_canonical(img)
    data = np.asarray(ras_img.dataobj)
    aff = ras_img.affine

    # Desired isotropic voxel size
    vox = 2.0

    # Compute world coordinates of voxel centers
    shape = data.shape
    # Create grid of voxel indices (i,j,k)
    ii, jj, kk = np.np.ogrid[:shape[0], :shape[1], :shape[2]]
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # N x 3
    # Homogeneous coordinates
    homo = np.hstack([ijk, np.ones((ijk.shape[0], 1))])  # N x 4
    world_centers = (aff @ homo.T).T[:, :3]  # N x 3

    lo = world_centers.min(axis=0)
    hi = world_centers.max(axis=0)
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1

    # Output affine: voxel size vox, origin at lo (center of first voxel)
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Resample using nearest neighbor
    # Grid of output voxel indices
    oo, pp, qq = np.indices(out_shape)
    out_idx = np.stack([oo.ravel(), pp.ravel(), qq.ravel(), np.ones_like(oo).ravel()], axis=1).T  # 4 x M
    world_out = (out_aff @ out_idx).T[:, :3]  # M x 3 world coordinates of output voxel centers
    # Map world to input voxel coordinates (continuous)
    inv_aff = np.linalg.inv(aff)
    homo_out = np.hstack([world_out, np.ones((world_out.shape[0], 1))])  # M x 4
    in_idx_cont = (inv_aff @ homo_out.T).T[:, :3]  # M x 3
    in_idx = np.rint(in_idx_cont).astype(int)
    # Valid mask
    ok = np.all((in_idx >= 0) & (in_idx < np.array(shape)[:, None]), axis=1)
    out = np.zeros(int(np.prod(out_shape)), dtype=data.dtype)
    out[ok] = data[in_idx[ok, 0], in_idx[ok, 1], in_idx[ok, 2]]
    out_img = nib.Nifti1Image(out.reshape(out_shape), out_aff)
    nib.save(out_img, out_path)
