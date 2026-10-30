"""SpatialContract: contract-verified repair of spatial bugs in imaging code."""

_API = ("spatial_contract", "check_function", "assert_spatial_contract", "summary")


def __getattr__(name):   # lazy, so importing any submodule does not pull in the oracle and nibabel
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'spatialcontract' has no attribute {name!r}")
