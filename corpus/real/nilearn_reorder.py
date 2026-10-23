# Real bug: nilearn reorder_img off-by-one in the origin for flipped y/z axes (fixed in 613c0d76)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import nibabel as nib
    from nilearn.image import reorder_img

    nib.save(reorder_img(nib.load(in_path), resample="nearest"), out_path)
