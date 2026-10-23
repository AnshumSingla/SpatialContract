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

    # Determine world bounding box of input volume (voxel centres)
    shape_in = np.array(data.shape)
    corners = np.array([[i, j, k, 1.0]
                        for i in (0, shape_in[0] - 1)
                        for j in (0, shape_in[1] - 1)
                        for k in (0, shape_in[2] - 1)])
    world_corners = (aff @ corners.T).T[:, :3]
    lo = world_corners.min(axis=0)
    hi = world_corners.max(axis=0)

    # Output: 2 mm isotropic voxels, axes aligned with world (RAS)
    vox = 2.0
    out_shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo

    # Voxel indices of output grid
    ii, jj, kk = np.indices(out_shape)
    coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()])  # 3 × N

    # Convert output voxel indices → world coordinates
    world_pts = out_aff[:3, :3] @ coords + out_aff[:3, 3:4]

    # Convert world coordinates → input voxel coordinates (continuous)
    inv_aff = np.linalg.inv(aff)
    inp_coords = inv_aff[:3, :3] @ world_pts + inv_aff[:3, 3:4]

    # Sample input data at those coordinates (linear interpolation)
    out = map_coordinates(data, inp_coords, order=1, mode='constant', cval=0.0)
    out = out.reshape(out_shape)

    nib.save(nib.Nifti1Image(out, out_aff), out_path)
