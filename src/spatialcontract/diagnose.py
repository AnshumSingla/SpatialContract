"""Describe HOW landmarks are misplaced, not only by how much. Deterministic, no LLM.

Six landmarks determine the affine map ``got = M @ expected + t`` that the code under test effectively applied to
physical space. Most spatial bugs are exactly such a map: an axis negated (a flip or a RAS/LPS mix-up), axes
permuted (index order), a uniform scale (units), a shift that is a whole number of output voxels (a stale affine
after a crop, an off-by-one), or a rotation (a direction matrix used the wrong way round). Stating the map in words
gives the repair agent, and a human, a signal that changes when a patch changes the error, where the raw per-landmark
list often looks the same from one round to the next (the S06 failure). It is still a measurement: it never names a
line of code or suggests a fix.
"""
from __future__ import annotations

import itertools

import numpy as np

AXES = "xyz"
PERM_TOL = 0.03      # max |M - s*P| entry to call M a (scaled) signed permutation
VOXEL_TOL = 0.05     # max distance from an integer, in output voxels, to call a shift "whole voxels"


def _signed_perms():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                P[row, col] = s
            yield P


def fit_affine(expected: np.ndarray, got: np.ndarray):
    """Least-squares ``got ~ M @ expected + t`` over N >= 4 points. Returns (M, t, rms residual)."""
    X = np.hstack([expected, np.ones((len(expected), 1))])
    sol, *_ = np.linalg.lstsq(X, got, rcond=None)
    M, t = sol[:3].T, sol[3]
    rms = float(np.sqrt(np.mean(np.sum((X @ sol - got) ** 2, axis=1))))
    return M, t, rms


def _expr(P: np.ndarray, s: float) -> str:
    terms = []
    for row in P:
        col = int(np.argmax(np.abs(row)))
        terms.append(("-" if row[col] < 0 else "") + AXES[col])
    scale = "" if abs(s - 1) < PERM_TOL else f"{_num(s)} * "
    return f"{scale}({', '.join(terms)})"


def _num(v: float) -> str:
    return f"{v:.4g}"


def _vec(v, nd=2) -> str:
    return "(" + ", ".join(f"{x + 0.0:+.{nd}f}".replace("-0.00", "+0.00") for x in v) + ")"


def describe(expected, got, out_affine=None, unit_mm: float = 1.0, frame: str = "RAS", noise_mm: float = 1.0) -> dict | None:
    """Describe the map from expected to observed landmark positions (both in contract frame/units).

    ``out_affine`` (contract frame/units) lets a shift be expressed in output voxels; ``noise_mm`` (the oracle's tolerance
    for this phantom) is the fit residual still accepted as an affine map, since resampling moves centroids by up to a voxel. Returns None when fewer than
    four landmarks were found or the points are degenerate. Keys: ``kind``, ``text`` (one line) and the fitted numbers."""
    pairs = [(e, g) for e, g in zip(expected, got) if e is not None and g is not None]
    if len(pairs) < 4:
        return None
    E = np.array([p[0] for p in pairs], dtype=float)
    G = np.array([p[1] for p in pairs], dtype=float)
    if np.linalg.matrix_rank(E - E.mean(axis=0), tol=1e-6) < 3:
        return None
    M, t, rms = fit_affine(E, G)
    t_mm, rms_mm = t * unit_mm, rms * unit_mm
    names = {"RAS": "x=R, y=A, z=S", "LPS": "x=L, y=P, z=S"}.get(frame, "")
    out = {"M": M.round(4).tolist(), "t_mm": t_mm.round(3).tolist(), "residual_mm": round(rms_mm, 3)}

    if rms_mm > max(1.0, noise_mm):
        out.update(kind="not_affine",
                   text=f"the landmarks are not moved by any single affine map (fit residual {rms_mm:.1f} mm): "
                        "different landmarks went to inconsistent places")
        return out

    # uniform scale times a signed permutation?
    s = float(np.cbrt(abs(np.linalg.det(M)))) or 1.0
    best = min(_signed_perms(), key=lambda P: np.abs(M / s - P).max())
    if np.abs(M / s - best).max() < PERM_TOL:
        identity = np.allclose(best, np.eye(3)) and abs(s - 1) < PERM_TOL
        shift = f" + {_vec(t_mm)} mm" if np.linalg.norm(t_mm) > 0.05 else ""
        if identity:
            out["kind"] = "translation"
            text = f"every landmark is shifted by the same vector {_vec(t_mm)} mm (|shift| {np.linalg.norm(t_mm):.2f} mm)"
        else:
            parts = []
            neg = [AXES[i] for i in range(3) if best[i, i] == -1]
            if any(best[i, i] == 0 for i in range(3)):
                parts.append("axes permuted")
            elif neg:
                parts.append(" and ".join(neg) + " negated")
            if abs(s - 1) >= PERM_TOL:
                parts.append(f"scaled by {_num(s)}")
            out["kind"] = "signed_permutation" if abs(s - 1) < PERM_TOL else "scaled"
            text = f"got = {_expr(best, s)}{shift} applied to expected ({', '.join(parts)}; axes {names})"
        if out_affine is not None and np.linalg.norm(t_mm) > 0.05:
            A = np.asarray(out_affine, dtype=float)[:3, :3]
            # the translation part expressed along the OUTPUT voxel axes (a stale affine after a crop is a whole-voxel shift)
            try:
                v = np.linalg.solve(A, t)
                if np.all(np.abs(v - v.round()) < VOXEL_TOL) and np.any(np.abs(v.round()) > 0):
                    text += f"; the shift is a whole number of output voxels {tuple(int(x) for x in v.round())}"
                    out["shift_voxels"] = [int(x) for x in v.round()]
            except np.linalg.LinAlgError:
                pass
        out["text"] = text
        return out

    # rotation (with possible reflection) and scaling: polar decomposition M = R @ S
    U, sv, Vt = np.linalg.svd(M)
    R = U @ Vt
    refl = np.linalg.det(R) < 0
    if refl:
        R = R @ np.diag([1, 1, -1])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    parts = []
    if ang > 0.5:
        parts.append(f"rotated by {ang:.1f} degrees")
    if refl:
        parts.append("mirrored")
    if np.abs(sv - 1).max() > PERM_TOL:
        parts.append("stretched by " + "/".join(_num(x) for x in sorted(sv)))
    out.update(kind="general_affine", angle_deg=round(ang, 2), reflected=bool(refl),
               text="landmarks are " + (", ".join(parts) or "moved by a general affine map")
                    + (f", then shifted by {_vec(t_mm)} mm" if np.linalg.norm(t_mm) > 0.05 else "")
                    + f" (axes {names})")
    return out
