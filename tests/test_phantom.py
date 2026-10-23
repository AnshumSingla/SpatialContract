import nibabel as nib
import numpy as np

from spatialcontract.phantom import SHAPE, make_phantom


def test_deterministic():
    a, b = make_phantom(3), make_phantom(3)
    assert np.array_equal(a.data, b.data) and np.allclose(a.affine, b.affine)
    assert not np.array_equal(make_phantom(4).data, a.data)


def test_never_already_ras_and_anisotropic():
    for s in range(30):
        p = make_phantom(s)
        assert "".join(nib.aff2axcodes(p.affine)) != "RAS"
        z = np.sqrt((p.affine[:3, :3] ** 2).sum(0))
        assert z.std() > 1e-3


def test_landmarks_match_affine_and_are_asymmetric():
    for s in range(10):
        p = make_phantom(s)
        assert set(np.unique(p.data)) == {0, 1, 2, 3, 4, 5, 6}
        centre = (np.array(SHAPE) - 1) / 2
        for label, c in p.voxel_centres.items():
            idx = np.argwhere(p.data == label)
            np.testing.assert_allclose(idx.mean(0), c, atol=1e-9)           # centroid is the centre
            np.testing.assert_allclose(p.affine[:3, :3] @ c + p.affine[:3, 3], p.world_ras_mm[label])
            assert np.all(np.abs(c - centre) >= 6)                          # off-centre on every axis
        for ax in range(3):                                                 # no flip maps phantom onto itself
            assert not np.array_equal(np.flip(p.data, ax), p.data)
