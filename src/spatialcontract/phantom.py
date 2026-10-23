"""Asymmetric landmark phantoms with known physical coordinates.

The phantom is a NIfTI-style label volume: N distinct integer-labelled balls on a zero background.
Input orientation, voxel size, obliquity and origin are all drawn from the seed, and the identity
(already-RAS) orientation is excluded so that "do nothing" pipelines cannot pass by luck.
Because every landmark has its own label and sits off-centre on every axis, no flip or axis
permutation of the array maps the phantom onto itself.

Ground truth is world position in RAS millimetres (the NIfTI convention) of each ball's centre.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import nibabel as nib
import numpy as np

N_LANDMARKS = 6
SHAPE = (48, 56, 40)
RADIUS_VOX = 3.0
MARGIN = 8       # voxels from the FOV edge
OFF_CENTRE = 6   # voxels away from the central plane on every axis
MIN_SEP = 12     # voxels between landmarks
MIN_MARGIN_GAP = 2  # voxels; foreground-bbox margins must differ pairwise across axes
STRONG_OBLIQUE_FROM = 200   # seeds >= this get a 50-70 degree tilt (opt-in bank; the default bank 0..2 is <= 12 degrees)
STRONG_OBLIQUE_DEG = (50.0, 70.0)


@dataclass(frozen=True)
class Phantom:
    seed: int
    data: np.ndarray               # int16 label volume
    affine: np.ndarray             # 4x4 voxel-centre -> RAS mm
    voxel_centres: dict            # label -> (3,) float voxel index of the ball centre
    world_ras_mm: dict             # label -> (3,) float ground-truth world position

    def save(self, path) -> None:
        img = nib.Nifti1Image(self.data, self.affine)
        img.set_sform(self.affine, code=1)
        img.set_qform(self.affine, code=1)
        nib.save(img, str(path))


def _signed_perms():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                P[row, col] = s
            yield P


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _rotation_angle(rng: np.random.Generator, axis: np.ndarray, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def make_phantom(seed: int, oblique: bool | None = None) -> Phantom:
    """Deterministic in ``seed``. ``oblique`` defaults to True for odd seeds; seeds >= STRONG_OBLIQUE_FROM
    are always strongly oblique (added after a real nibabel bug only showed up on such affines)."""
    rng = np.random.default_rng(10_000 + seed)
    perms = [P for P in _signed_perms() if not np.allclose(P, np.eye(3))]
    P = perms[int(rng.integers(len(perms)))]
    zooms = rng.uniform(0.8, 3.0, size=3)
    if oblique is None:
        oblique = bool(seed % 2)
    if seed >= STRONG_OBLIQUE_FROM:
        axis = rng.normal(size=3)
        R = _rotation_angle(rng, axis / np.linalg.norm(axis), rng.uniform(*STRONG_OBLIQUE_DEG))
    else:
        R = _rotation(rng, 12.0) if oblique else np.eye(3)
    A3 = R @ P @ np.diag(zooms)
    centre_vox = (np.array(SHAPE) - 1) / 2
    t = rng.uniform(-40, 40, size=3) - A3 @ centre_vox
    affine = np.eye(4)
    affine[:3, :3] = A3
    affine[:3, 3] = t

    lo = np.full(3, MARGIN)
    hi = np.array(SHAPE) - 1 - MARGIN
    while True:
        pts: list[np.ndarray] = []
        while len(pts) < N_LANDMARKS:
            c = rng.uniform(lo, hi).round()
            if np.any(np.abs(c - centre_vox) < OFF_CENTRE):
                continue
            if any(np.linalg.norm(c - q) < MIN_SEP for q in pts):
                continue
            pts.append(c)
        # The foreground bounding box must not be invariant under any axis permutation: the margins to the
        # FOV edge must differ by >= MIN_MARGIN_GAP voxels across axes (found when a crop bug that reversed
        # axis order slipped through palindromic margins such as [3, 6, 3]).
        arr = np.array(pts)
        lo_m = arr.min(axis=0) - RADIUS_VOX
        hi_m = (np.array(SHAPE) - 1) - (arr.max(axis=0) + RADIUS_VOX)
        if all(abs(m[a] - m[b]) >= MIN_MARGIN_GAP for m in (lo_m, hi_m) for a, b in ((0, 1), (0, 2), (1, 2))):
            break

    data = np.zeros(SHAPE, dtype=np.int16)
    ii, jj, kk = np.indices(SHAPE)
    vox, world = {}, {}
    for label, c in enumerate(pts, start=1):
        d2 = (ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2
        data[d2 <= RADIUS_VOX ** 2] = label
        vox[label] = c
        world[label] = (affine @ np.append(c, 1.0))[:3]
    return Phantom(seed, data, affine, vox, world)
