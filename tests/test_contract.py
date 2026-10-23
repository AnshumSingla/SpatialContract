import numpy as np

from spatialcontract.contract import SpatialContract, from_dict, load_from_source

OK = dict(world_frame="RAS", units="mm", affine_convention="voxel_center_to_world")


def test_complete_contract_has_no_problems():
    assert from_dict(OK).problems() == []
    assert from_dict(OK | {"array_orientation": "LPS"}).problems() == []


def test_missing_and_unknown_fields_are_problems():
    assert SpatialContract().problems()
    assert any("units" in p for p in from_dict(OK | {"units": "unknown"}).problems())
    assert any("world_frame" in p for p in from_dict({k: v for k, v in OK.items() if k != "world_frame"}).problems())


def test_ambiguous_values_rejected():
    assert from_dict(OK | {"affine_convention": "voxel_corner_to_world"}).problems()
    assert from_dict(OK | {"world_frame": "RPI"}).problems()
    assert from_dict(OK | {"array_orientation": "RAL"}).problems()   # L and R both used
    assert from_dict(OK | {"array_orientation": "ras"}).problems()


def test_load_from_source_does_not_execute():
    src = ('import os\nSPATIAL_CONTRACT = {"world_frame": "LPS", "units": "m", '
           '"affine_convention": "voxel_center_to_world"}\n')
    c = load_from_source(src)
    assert c.world_frame == "LPS" and c.units == "m"
    assert load_from_source("x = 1\n") is None
    assert load_from_source('SPATIAL_CONTRACT = dict(units="mm")\n').problems()  # non-literal => unusable
    assert load_from_source("def broken(:\n") is None


def test_frame_unit_conversion():
    c = from_dict(OK | {"world_frame": "LPS", "units": "m"})
    np.testing.assert_allclose(c.ras_mm_to_contract([10, 20, 30]), [-0.01, -0.02, 0.03])
