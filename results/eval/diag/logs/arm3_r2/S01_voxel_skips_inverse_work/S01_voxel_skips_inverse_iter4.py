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
    ii, jj, kk = np.ogrid[:shape[0], :shape[1], :shape[2]]
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
    ox = np.arange(out_shape[0])
    oy = np.arange(out_shape[1])
    oz = np.arange(out_shape[2])
    oox, ooy, ooz = np.meshgrid(ox, oy, oz, indexing='ij')
    out_idx = np.stack([oox.ravel(), ooy.ravel(), ooz.ravel()], axis=1)  # M x 3
    # Homogeneous
    out_homo = np.hstack([out_idx, np.ones((out_idx.shape[0], 1))])  # M x 4
    world_out = (out_aff @ out_homo.T).T[:, :3]  # M x 3 world coordinates
    # Map world to input voxel coordinates (continuous)
    inv_aff = np.linalg.inv(aff)
    in_homo = np.hstack([world_out, np.ones((world_out.shape[0], 1))])  # M x 4
    in_idx_cont = (inv_aff @ in_homo.T).T[:, :3]  # M x 3
    in_idx = np.rint(in_idx_cont).astype(int)
    # Valid mask
    ok = np.all((in_idx >= 0) & (in_idx < np.array(shape)[:, None]), axis=1)
    out = np.zeros(int(np.prod(out_shape)), dtype=data.dtype)
    out[ok] = data[in_idx[ok, 0], in_idx[ok, 1], in_idx[ok, 2]]
    out_img = nib.Nifti1Image(out.reshape(out_shape), out_aff)
    nib.save(out_img, out_path)
