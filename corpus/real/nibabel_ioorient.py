# Real bug: nibabel io_orientation labelled axes inconsistently when axes compete (fixed in a9caf8f9 + 927be913)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world",
                    "array_orientation": "RAS"}


def process(in_path, out_path):
    import nibabel as nib

    nib.save(nib.as_closest_canonical(nib.load(in_path)), out_path)
