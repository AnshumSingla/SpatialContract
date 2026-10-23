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
    # Reorient to RAS so that output axes align with RAS
    ras_ornt = o.axcodes2ornt(('R', 'A', 'S'))
    current_ornt = io_orientation(img.affine)
    transform = o.ornt_transform(current_ornt, ras_ornt)
    img_ras = img.apply_orientation(transform)
    data = np.asarray(img_ras.dataobj)
    aff = img_ras.affine  # now RAS-aligned
    
    vox = 2.0
    # world coordinates of input voxel centres
    ijk = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                    for j in (0, data.shape[1] - 1)
                    for k in (0, data.shape[2] - 1)])
    world = (aff @ ijk.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    # number of voxels needed so that lo + (shape-1)*vox covers hi
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo  # first voxel centre at lo
    
    # grid of output voxel indices
    ii, jj, kk = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    out_idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                        np.ones_like(ii.ravel())], axis=0)  # 4 x N
    # world coordinates of output voxel centres
    world_pts = out_aff @ out_idx
    # map back to input voxel indices (continuous)
    in_idx_cont = np.linalg.inv(aff) @ world_pts
    in_idx = np.rint(in_idx_cont[:3]).astype(int)
    # keep only those inside the input volume
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape, order="F"), out_aff), out_path)

# Helper from nibabel.orientations
def io_orientation(aff):
    """Return orientation codes (as nibabel ornt) for given affine."""
    return o.io_orientation(aff)