import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

SPATIAL_CONTRACT = {
    "world_frame": "RAS",
    "units": "mm",
    "affine_convention": "voxel_center_to_world"
}

def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine

    # World bounding box of input volume (voxel centres)
    shape_in = np.array(data.shape)
    corners = np.array([[i, j, k, 1.0]
                        for i in (0, shape_in[0] - 1)
                        for j in (0, shape_in[1] - 1)
                        for k in (0, shape_in[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)

    vox = 2.0
    # Generate output voxel centre coordinates covering the input bounding box
    xs = np.arange(lo[0], hi[0] + 1e-9, vox)
    ys = np.arange(lo[1], hi[1] + 1e-9, vox)
    zs = np.arange(lo[2], hi[2] + 1e-9, vox)
    # Build grid of indices
    ii, jj, kk = np.meshgrid(xs, ys, zs, indexing='ij')
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()])  # 3 × N

    # Output affine: maps index (i,j,k) -> world = lo + [i,j,k]*vox
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Convert output voxel indices (actually we have world coordinates already)
    # Instead compute continuous input voxel coordinates for sampling
    world_pts = coords  # because coords are world centres
    inv_aff = np.linalg.inv(aff)
    inp_coords = inv_aff[:3, :3] @ world_pts + inv_aff[:3, 3:4]

    # Sample with linear interpolation
    out = map_coordinates(data, inp_coords, order=1, mode='constant', cval=0.0)
    out = out.reshape(ii.shape)

    nib.save(nib.Nifti1Image(out, out_aff), out_path)
