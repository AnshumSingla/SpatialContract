# Real bug: TorchIO Crop did not update the image affine (fixed in 7e1decf, issue #351)
SPATIAL_CONTRACT = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}


def process(in_path, out_path):
    import torchio

    subject = torchio.Subject(img=torchio.ScalarImage(in_path))
    out = torchio.Crop((4, 1, 3, 5, 5, 2))(subject)
    out.img.save(out_path)
