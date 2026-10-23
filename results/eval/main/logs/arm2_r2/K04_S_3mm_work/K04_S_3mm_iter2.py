import numpy as np
import nibabel as nib
import itertools

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
    mins = np.array([-0.5, -0.5, -0.5])
    maxs = np.array(data.shape) - 0.5
    # generate all combinations of mins and maxs -> 8 corners
    corner_idx = np.array(list(itertools.product(zip(mins, maxs))))
    # Actually product of tuples gives ((xmin,xmax),(ymin,ymax),(zmin,zmax)) need to expand:
    # Let's do simple loops:
    corner_idx = np.array([[x, y, z, 1.0]
                           for x in (mins[0], maxs[0])
                           for y in (mins[1], maxs[1])
                           for z in (mins[2], maxs[2])])
    world = (aff @ corner_idx.T).T[:, :3]
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