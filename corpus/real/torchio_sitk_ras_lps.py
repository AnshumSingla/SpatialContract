# Real bug: TorchIO wrote affines to SimpleITK without RAS->LPS conversion, double-flipping on save (fixed in 09138e0)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import torchio as tio

    tio.ScalarImage(in_path).save(out_path)
