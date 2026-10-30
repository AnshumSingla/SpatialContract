"""In-memory pipelines checked without restructuring them: a crop to the foreground bounding box, written four ways.

The buggy versions crop the array but keep the input affine, so every voxel's world position silently shifts by the
cropped margin (a few to tens of millimetres). Shapes, dtypes and values all look right; only physical position is wrong.

Run:  python examples/inmemory_example.py     or     spatialcontract check examples/inmemory_example.py
"""
import nibabel as nib
import numpy as np

from spatialcontract.api import check_function, spatial_contract, summary


def _bbox(data: np.ndarray, pad: int = 2):
    idx = np.argwhere(data != 0)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, data.shape)
    return lo, hi


@spatial_contract(world_frame="RAS", units="mm")
def crop_buggy(img: nib.Nifti1Image) -> nib.Nifti1Image:
    data = np.asanyarray(img.dataobj)
    lo, hi = _bbox(data)
    return nib.Nifti1Image(data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], img.affine)       # affine not updated


@spatial_contract(world_frame="RAS", units="mm")
def crop_fixed(img: nib.Nifti1Image) -> nib.Nifti1Image:
    lo, hi = _bbox(np.asanyarray(img.dataobj))
    return img.slicer[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]                               # slicer shifts the origin


@spatial_contract(world_frame="RAS", units="mm", signature="array")
def crop_array_buggy(data: np.ndarray, affine: np.ndarray):
    lo, hi = _bbox(data)
    return data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], affine


@spatial_contract(world_frame="RAS", units="mm", signature="array")
def crop_array_fixed(data: np.ndarray, affine: np.ndarray):
    lo, hi = _bbox(data)
    new_aff = affine.copy()
    new_aff[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]                                   # world position of new voxel 0
    return data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], new_aff


if __name__ == "__main__":
    import inmemory_example as ex   # the oracle's subprocess imports functions by module, and __main__ is not importable

    for f in (ex.crop_buggy, ex.crop_fixed, ex.crop_array_buggy, ex.crop_array_fixed):
        print(f"{f.__name__:18s} {summary(check_function(f))}")
