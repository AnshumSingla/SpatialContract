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
    # Ensure data array is usable
    data = np.asarray(img.dataobj)
    aff = img.affine

    # Reorient to RAS (right-anterior-superior)
    cur_axcodes = nib.aff2axcodes(aff)
    ras_axcodes = ('R', 'A', 'S')
    if cur_axcodes != ras_axcodes:
        # Compute transformation to RAS
        trans = o.axcodes2ornt(cur_axcodes)
        ras_ornt = o.axcodes2ornt(ras_axcodes)
        transform = o.ornt_transform(trans, ras_ornt)
        # Apply orientation change
        data = o.apply_orientation(data, transform)
        # Update affine accordingly
        aff = aff @ o.inv_ornt_transform(transform)

    # Desired isotropic voxel size
    vox = 2.0
    # Determine output shape covering the whole RAS-oriented volume
    # Compute world bounds of the voxel centers
    ones = np.ones((data.shape[0] * data.shape[1] * data.shape[2], 1))
    ijk = np.vstack(np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing='ij'
    )).reshape(3, -1).T
    xyz_centers = (aff @ np.hstack([ijk, np.ones((ijk.shape[0], 1))]).T).T[:, :3]
    lo = xyz_centers.min(axis=0)
    hi = xyz_centers.max(axis=0)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1

    # Build output affine: voxel size vox, origin at lo (corner of first voxel center?)
    # Since affine maps voxel index to world coordinate of voxel center,
    # we set translation so that index (0,0,0) maps to lo.
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Resample using nearest neighbor (to match original intent)
    # Create grid of output voxel indices
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T  # 4 x N
    world_pts = out_aff @ out_idx  # 4 x N world coordinates (homogeneous)
    # Convert world to input voxel indices (continuous)
    inv_aff = np.linalg.inv(aff)
    in_idx_cont = (inv_aff @ world_pts)[:3, :]  # 3 x N
    # Nearest neighbor lookup
    in_idx = np.rint(in_idx_cont).astype(int)
    # Valid mask
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    out_img = nib.Nifti1Image(out.reshape(shape), out_aff)
    nib.save(out_img, out_path)
