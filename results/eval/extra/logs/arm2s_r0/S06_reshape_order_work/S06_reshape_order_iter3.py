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
    # voxel size of input (world units per index step)
    voxel_sizes = np.sqrt(np.sum(aff[:3, :3] ** 2, axis=0))
    # offset half a voxel in index space to get corners
    offset = 0.5
    # corner in index space: min corner = (-offset, -offset, -offset)
    min_idx = np.array([-offset, -offset, -offset])
    max_idx = np.array([data.shape[0] - 1 + offset,
                        data.shape[1] - 1 + offset,
                        data.shape[2] - 1 + offset])
    # world coordinates of corners
    lo_world = aff @ np.append(min_idx, 1.0)
    hi_world = aff @ np.append(max_idx, 1.0)
    lo = lo_world[:3]
    hi = hi_world[:3]
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
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)