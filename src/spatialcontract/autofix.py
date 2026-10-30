"""Deterministic repair, no model: when the oracle measures the SAME misplacement on every visible phantom, correct it directly.

Many spatial bugs are input-independent maps of physical space: millimetres written as metres (a scale), a missing RAS<->LPS
conversion (two axes negated). For those, the affine map fitted by diagnose.py is identical on every phantom, and composing the
output affine with its inverse is an exact fix. The same holds when only the output's array orientation is wrong (every landmark
in place): re-orienting the output to the declared ``array_orientation`` is exact for any input. When the map differs between
phantoms (it depends on the input's orientation, size or crop), no single correction exists and the bug goes to the model.

The patch is a wrapper appended to the script: the original ``process`` runs unchanged, then the output's affine is corrected (and
its array re-oriented if the contract requires it). It corrects the symptom, not the faulty line, which is why it is applied only
when the visible measurements show one map for all inputs, and why it then faces the same patch guard and held-out validation as
any model patch. The decision uses only the visible oracle.
"""
from __future__ import annotations

import numpy as np

from . import oracle
from .contract import SpatialContract

M_TOL, T_TOL_MM = 1e-3, 0.05


def consistent_map(rep: oracle.Report, contract: SpatialContract):
    """The single correction (4x4, contract units) that the visible measurements support, or None. 'orient' means only the output
    orientation is wrong."""
    if rep.verdict != "FAIL" or any(p.status == "ERROR" for p in rep.phantoms):
        return None
    if all(lm.ok for p in rep.phantoms for lm in p.landmarks):
        return "orient" if contract.array_orientation and any(p.orientation_ok is False for p in rep.phantoms) else None
    maps = []
    for p in rep.phantoms:
        pat = p.pattern
        if p.status == "PASS":
            maps.append((np.eye(3), np.zeros(3)))
            continue
        if not pat or pat.get("kind") in (None, "not_affine") or "M" not in pat:
            return None
        maps.append((np.asarray(pat["M"], float), np.asarray(pat["t_mm"], float) / _unit_mm(contract)))
    M0, t0 = maps[0]
    unit = _unit_mm(contract)
    if any(np.abs(M - M0).max() > M_TOL or np.abs(t - t0).max() * unit > T_TOL_MM for M, t in maps[1:]):
        return None                                   # the misplacement depends on the input: no single correction
    Minv = np.linalg.inv(M0)
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = Minv, -Minv @ t0
    return C


def _unit_mm(contract: SpatialContract) -> float:
    from .contract import UNIT_TO_MM
    return UNIT_TO_MM[contract.units]


def wrapper_source(original: str, correction, contract: SpatialContract, arg_names: list[str]) -> str:
    """Append a correction wrapper to ``original``; ``process`` keeps its signature."""
    a_in, a_out = arg_names[:2]
    lines = ["", "", "# --- SpatialContract deterministic correction (generated, no model): the measured misplacement was the same",
             "# map on every visible phantom, so the output affine is corrected by its inverse.",
             "_sc_original_process = process", "", "", f"def process({a_in}, {a_out}):",
             f"    _sc_original_process({a_in}, {a_out})", "    import nibabel as _sc_nib", "    import numpy as _sc_np",
             f"    _sc_img = _sc_nib.load({a_out})", "    _sc_data = _sc_np.asarray(_sc_img.dataobj)", "    _sc_aff = _sc_np.asarray(_sc_img.affine, float)"]
    if isinstance(correction, np.ndarray):
        lines.append(f"    _sc_aff = _sc_np.array({np.round(correction, 10).tolist()}) @ _sc_aff")
    if contract.array_orientation:
        # re-orient the array so its axes run toward the declared directions (read in RAS, as the contract defines)
        flip = "_sc_np.diag([-1.0, -1.0, 1.0, 1.0])" if contract.world_frame == "LPS" else "_sc_np.eye(4)"
        lines += [f"    _sc_ras = {flip} @ _sc_aff",
                  "    _sc_ornt = _sc_nib.orientations.ornt_transform(_sc_nib.orientations.io_orientation(_sc_ras),",
                  f"                                                   _sc_nib.orientations.axcodes2ornt(tuple({contract.array_orientation!r})))",
                  "    _sc_data = _sc_nib.orientations.apply_orientation(_sc_data, _sc_ornt)",
                  "    _sc_aff = _sc_aff @ _sc_nib.orientations.inv_ornt_aff(_sc_ornt, _sc_np.asarray(_sc_img.dataobj).shape)"]
    lines += ["    _sc_out = _sc_nib.Nifti1Image(_sc_np.ascontiguousarray(_sc_data), _sc_aff)",
              "    _sc_out.set_sform(_sc_aff, code=1)", "    _sc_out.set_qform(_sc_aff, code=1)", f"    _sc_nib.save(_sc_out, {a_out})", ""]
    return original.rstrip("\n") + "\n" + "\n".join(lines)
