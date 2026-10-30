"""pytest plugin, registered through the ``pytest11`` entry point (or ``-p spatialcontract.pytest_plugin`` /
``pytest_plugins = ["spatialcontract.pytest_plugin"]`` when the package is not installed).

    def test_crop_keeps_world_positions(spatial_oracle):
        spatial_oracle(crop, world_frame="RAS", units="mm")

``spatial_oracle(fn, **kwargs)`` takes ``api.check_function``'s arguments, returns the Report on PASS and otherwise fails the
test with the oracle's one-line summary and millimetre feedback (no traceback into the plugin). The api is imported lazily so
that loading the plugin costs nothing for test sessions that never use it.
"""
import pytest


@pytest.fixture
def spatial_oracle():
    from .api import SpatialContractViolation, assert_spatial_contract

    def run(fn, **kwargs):
        try:
            return assert_spatial_contract(fn, **kwargs)
        except SpatialContractViolation as e:
            pytest.fail(str(e), pytrace=False)
    return run
