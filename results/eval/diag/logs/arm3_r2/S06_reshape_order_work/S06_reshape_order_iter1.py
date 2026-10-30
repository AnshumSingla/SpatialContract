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
    # bounding box of voxel centers
    min_pt = aff @ np.array([0, 0, 0, 1.0])
    max_pt = aff @ np.array([data.shape[0]-1,
                             data.shape[1]-1,
                             data.shape[2]-1, 1.0])
    lo = np.minimum(min_pt[:3], max_pt[:3])
    hi = np.maximum(min_pt[:3], max_pt[:3])
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    ii, jj, kk = np.indices(shape)
    # world coordinates of output voxel centers
    world_pts = out_aff @ np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                                    np.ones_like(ii).ravel()], axis=0)
    # map to input voxel indices (continuous)
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)