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
    vox = 3.0
    # voxel corner indices: subtract 0.5 from min index, add 0.5 to max index
    mins = np.array([0, 0, 0]) - 0.5
    maxs = np.array(data.shape) - 0.5
    corners = np.array([[i, j, k, 1.0] for i in mins for j in mins for k in mins] +
                       [[i, j, k, 1.0] for i in maxs for j in maxs for k in maxs])
    # Actually need all combos of mins and maxs per axis -> 8 corners
    # Simpler: generate product
    corner_idx = np.array(np.meshgrid(mins, maxs, mins)).T.reshape(-1, 3)  # wrong
    # Let's just compute 8 corners properly:
    corner_idx = np.transpose([np.tile(mins, 4), np.repeat(mins, 2), np.tile([mins[0], maxs[0]], 4)])  # too messy
    # Instead compute using min and max per axis:
    # We'll compute world coordinates of the 8 corners directly:
    xyz_min = mins
    xyz_max = maxs
    corners = np.array([[x, y, z, 1.0]
                        for x in (xyz_min[0], xyz_max[0])
                        for y in (xyz_min[1], xyz_max[1])
                        for z in (xyz_min[2], xyz_max[2])])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)