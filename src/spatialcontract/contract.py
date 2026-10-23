"""Spatial contract: what a pipeline declares about the physical meaning of its output.

A pipeline declares a module-level literal::

    SPATIAL_CONTRACT = {
        "world_frame": "RAS",            # RAS or LPS: direction in which +x/+y/+z point in patient space
        "units": "mm",                   # mm, m or um
        "affine_convention": "voxel_center_to_world",
        "array_orientation": "RAS",      # optional: output voxel axes must run toward these patient directions
    }

The contract is read with ``ast.literal_eval`` (never executed). If it is missing, incomplete or
ambiguous, ``problems()`` is non-empty and the oracle/agent must ABSTAIN instead of guessing.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict

import numpy as np

FRAMES = ("RAS", "LPS")
UNIT_TO_MM = {"mm": 1.0, "m": 1000.0, "um": 0.001}
CONVENTIONS = ("voxel_center_to_world",)
_AXIS_PAIRS = ({"R", "L"}, {"A", "P"}, {"S", "I"})
REQUIRED = ("world_frame", "units", "affine_convention")


@dataclass(frozen=True)
class SpatialContract:
    world_frame: str | None = None
    units: str | None = None
    affine_convention: str | None = None
    array_orientation: str | None = None  # optional

    def problems(self) -> list[str]:
        """Reasons the contract cannot be used. Empty list means the oracle may run."""
        out = []
        for f in REQUIRED:
            if getattr(self, f) in (None, "", "unknown"):
                out.append(f"missing field '{f}'")
        if self.world_frame not in (None, "", "unknown") and self.world_frame not in FRAMES:
            out.append(f"world_frame {self.world_frame!r} not one of {FRAMES}")
        if self.units not in (None, "", "unknown") and self.units not in UNIT_TO_MM:
            out.append(f"units {self.units!r} not one of {tuple(UNIT_TO_MM)}")
        if self.affine_convention not in (None, "", "unknown") and self.affine_convention not in CONVENTIONS:
            out.append(f"affine_convention {self.affine_convention!r} unsupported/ambiguous "
                       f"(supported: {CONVENTIONS})")
        ao = self.array_orientation
        if ao is not None:
            valid = (isinstance(ao, str) and len(ao) == 3 and all(c in "RLAPSI" for c in ao)
                     and {frozenset({c, _opposite(c)}) for c in ao} == {frozenset(p) for p in _AXIS_PAIRS})
            if not valid:
                out.append(f"array_orientation {ao!r} is not a valid 3-letter code using each of R/L, A/P, S/I once")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    # --- conversions the oracle needs -------------------------------------------------
    def ras_mm_to_contract(self, p_ras_mm: np.ndarray) -> np.ndarray:
        """Convert world points (...,3) expressed in RAS millimetres into this contract's frame/units."""
        p = np.asarray(p_ras_mm, dtype=float).copy()
        if self.world_frame == "LPS":
            p[..., 0] *= -1
            p[..., 1] *= -1
        return p / UNIT_TO_MM[self.units]

    def contract_to_ras_affine_mm(self, affine: np.ndarray) -> np.ndarray:
        """Re-express an affine whose output world is in this contract's frame/units as RAS mm."""
        a = np.asarray(affine, dtype=float).copy()
        a[:3, :] *= UNIT_TO_MM[self.units]
        if self.world_frame == "LPS":
            a[0, :] *= -1
            a[1, :] *= -1
        return a


def _opposite(c: str) -> str:
    return {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}[c]


def from_dict(d: dict | None) -> SpatialContract | None:
    if d is None:
        return None
    known = {"world_frame", "units", "affine_convention", "array_orientation"}
    return SpatialContract(**{k: v for k, v in d.items() if k in known})


def load_from_source(code: str) -> SpatialContract | None:
    """Read ``SPATIAL_CONTRACT = {...}`` from source text without executing it. None if absent."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SPATIAL_CONTRACT" for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return SpatialContract()  # present but not a literal -> every field missing -> abstain
            return from_dict(value) if isinstance(value, dict) else SpatialContract()
    return None
