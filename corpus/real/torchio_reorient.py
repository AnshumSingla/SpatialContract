# Real bug: TorchIO Reorient applied permutation/flip in the wrong order and used the wrong shape for the affine (fixed in 95b1b74)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world",
                    "array_orientation": "RAS"}


def process(in_path, out_path):
    import torchio as tio

    subject = tio.Subject(t1=tio.ScalarImage(in_path))
    tio.Reorient(orientation="RAS")(subject).t1.save(out_path)
