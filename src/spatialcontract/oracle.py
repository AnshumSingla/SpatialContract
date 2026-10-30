"""Deterministic physical-coordinate oracle. No LLM, no randomness beyond the phantom seeds.

The code under test is a Python file defining ``process(in_path, out_path)`` that reads a NIfTI
image and writes a NIfTI image. The oracle writes an asymmetric landmark phantom, runs the code in
a subprocess with a scrubbed environment, finds each landmark in the output volume, maps it
through the OUTPUT affine and compares with the ground-truth physical position expressed in the
frame/units the contract declares.

Verdicts: PASS | FAIL | ERROR (code crashed / no usable output) | ABSTAIN (contract unusable).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from . import diagnose
from .contract import SpatialContract, UNIT_TO_MM, load_from_source
from .phantom import Phantom, make_phantom

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_TOL_MM = 1.5
HALF_VOXEL_DIAG = 0.866   # sqrt(3)/2 of the largest output voxel edge
MAX_TOL_MM = 4.0
LOSSLESS_TOL_MM = 0.05
TIMEOUT_S = 90

_RUNNER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('code_under_test', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
    "m.process(sys.argv[2], sys.argv[3])\n"
)
_ENV_KEEP = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")


@dataclass
class LandmarkResult:
    label: int
    expected: list | None = None   # in contract frame/units
    got: list | None = None
    error_mm: float | None = None  # None => landmark lost in output
    ok: bool = False


@dataclass
class PhantomResult:
    seed: int
    status: str                    # PASS | FAIL | ERROR
    landmarks: list = field(default_factory=list)
    in_axcodes: str | None = None
    in_zooms_mm: list | None = None
    out_axcodes: str | None = None
    out_zooms_mm: list | None = None
    orientation_ok: bool | None = None
    tol_mm_used: float | None = None
    lossless: bool | None = None
    pattern: dict | None = None    # diagnose.describe of the misplacement, FAIL only
    detail: str = ""


@dataclass
class Report:
    verdict: str
    contract: dict | None
    reasons: list = field(default_factory=list)
    phantoms: list = field(default_factory=list)
    tol_mm: float = DEFAULT_TOL_MM
    diagnose: bool = True          # include the fitted misplacement pattern in feedback() (False reproduces the original text)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def worst(self):
        """(error_mm, seed, label) of the worst landmark across phantoms; lost landmark => inf."""
        w = None
        for p in self.phantoms:
            for lm in p.landmarks:
                e = float("inf") if lm.error_mm is None else lm.error_mm
                if w is None or e > w[0]:
                    w = (e, p.seed, lm.label)
        return w

    def feedback(self, max_lines: int = 14) -> str:
        """Text for the repair agent / human. States measurements, never a suggested fix."""
        if self.verdict == "ABSTAIN":
            return "ABSTAIN: spatial contract unusable: " + "; ".join(self.reasons)
        lines = [f"VERDICT: {self.verdict}  (tolerance {self.tol_mm} mm; contract {self.contract})"]
        for p in self.phantoms:
            if p.status == "ERROR":
                lines.append(f"phantom seed={p.seed}: ERROR {p.detail}")
                continue
            bad = [lm for lm in p.landmarks if not lm.ok]
            lines.append(
                f"phantom seed={p.seed}: {p.status}; input axcodes={p.in_axcodes} zooms_mm={p.in_zooms_mm} -> output axcodes={p.out_axcodes} "
                f"zooms_mm={p.out_zooms_mm}; orientation_ok={p.orientation_ok}; "
                f"lossless={p.lossless} tol_used={p.tol_mm_used} mm; {len(bad)}/{len(p.landmarks)} landmarks off")
            if self.diagnose and p.pattern:
                lines.append(f"  pattern: {p.pattern['text']}")
            for lm in bad[:2 if self.diagnose and p.pattern else 3]:
                err = "LOST" if lm.error_mm is None else f"{lm.error_mm:.2f} mm"
                lines.append(f"  landmark {lm.label}: expected {_fmt(lm.expected)} got {_fmt(lm.got)} error {err}")
        return "\n".join(lines[:max_lines])

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self) | {"verdict": self.verdict}


def _fmt(v):
    return "none" if v is None else "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _run_code(code_path: Path, in_path: Path, out_path: Path, cwd: Path,
              python: str | None = None, pythonpath: list | None = None, runner: str = _RUNNER,
              extra_args: tuple = ()) -> tuple[bool, str]:
    """Run ``process`` in a subprocess. ``python``/``pythonpath`` select an interpreter and extra import roots
    (used by the real-bug corpus to run a library at a specific commit); isolated mode (-I) is used otherwise.
    ``runner``/``extra_args`` swap in another driver (``locate`` runs the same call under a tracer)."""
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    cmd = [python or sys.executable]
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
        cmd += ["-s"]
    else:
        cmd += ["-I"]
    try:
        r = subprocess.run(cmd + ["-c", runner, str(code_path), str(in_path), str(out_path), *map(str, extra_args)],
                           capture_output=True, text=True, timeout=TIMEOUT_S, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TIMEOUT_S}s"
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return False, "exit %d: %s" % (r.returncode, tail[-1] if tail else "")
    return True, ""


def _check_one_inner(code_path: Path, contract: SpatialContract, ph: Phantom, tol_mm: float, tmp: Path,
               python: str | None = None, pythonpath: list | None = None, run=None) -> PhantomResult:
    in_path = tmp / f"phantom_{ph.seed}.nii.gz"
    out_path = tmp / f"out_{ph.seed}.nii.gz"
    ph.save(in_path)
    ok, err = (run or _run_code)(code_path, in_path, out_path, tmp, python, pythonpath)
    if not ok:
        return PhantomResult(ph.seed, "ERROR", detail=str(err))
    return measure(out_path, contract, ph, tol_mm)


def measure(out_path: Path, contract: SpatialContract, ph: Phantom, tol_mm: float = DEFAULT_TOL_MM) -> PhantomResult:
    """Score an output file that ``process`` wrote for phantom ``ph``. Raises on output it cannot analyse (see _check_one)."""
    try:
        out = nib.load(str(out_path))
        data = np.asarray(out.dataobj)
        aff = np.asarray(out.affine, dtype=float)
    except Exception as e:  # missing/unreadable output
        return PhantomResult(ph.seed, "ERROR", detail=f"unreadable output: {type(e).__name__}: {str(e)[:120]}")
    if data.ndim != 3:
        return PhantomResult(ph.seed, "ERROR", detail=f"output has {data.ndim} dims, expected 3")

    res = PhantomResult(ph.seed, "PASS")
    res.in_axcodes = "".join(nib.aff2axcodes(ph.affine))
    res.in_zooms_mm = [round(float(z), 3) for z in np.sqrt((ph.affine[:3, :3] ** 2).sum(axis=0))]
    ras_aff = contract.contract_to_ras_affine_mm(aff)
    res.out_axcodes = "".join(nib.aff2axcodes(ras_aff))
    res.out_zooms_mm = [round(float(z), 3) for z in np.sqrt((ras_aff[:3, :3] ** 2).sum(axis=0))]
    if contract.array_orientation:
        res.orientation_ok = res.out_axcodes == contract.array_orientation
    # Lossless pipelines (reorient, crop, copy) keep every landmark's exact voxel count and must reproduce its
    # world position essentially exactly; a real nilearn bug (reorder_img off-by-one voxel) slipped under the
    # old fixed 1.5 mm tolerance. Resampling pipelines change the counts; there nearest-neighbour moves a
    # centroid by up to ~half a voxel diagonal, so the tolerance grows with the output voxel size. The cap
    # stops a bug that inflates its own zooms from buying itself a lenient tolerance.
    in_counts = {lb: int(np.count_nonzero(ph.data == lb)) for lb in ph.world_ras_mm}
    res.lossless = all(int(np.count_nonzero(np.abs(data - lb) < 0.25)) == n for lb, n in in_counts.items())
    if res.lossless:
        tol_mm = min(tol_mm, LOSSLESS_TOL_MM)
    else:
        tol_mm = min(max(tol_mm, HALF_VOXEL_DIAG * max(res.out_zooms_mm)), MAX_TOL_MM)
    res.tol_mm_used = round(tol_mm, 3)
    def landmarks(centre_of) -> list:
        out = []
        for label, true_ras in ph.world_ras_mm.items():
            expected = contract.ras_mm_to_contract(true_ras)
            lm = LandmarkResult(label, expected=expected.tolist())
            centre = centre_of(label)
            if centre is not None:                      # None => lost
                got = (aff @ np.append(centre, 1.0))[:3]
                lm.got = got.tolist()
                lm.error_mm = float(np.linalg.norm(got - expected) * UNIT_TO_MM[contract.units])
                lm.ok = lm.error_mm <= tol_mm
            out.append(lm)
        return out

    def exact(label):
        idx = np.argwhere(np.abs(data - label) < 0.25)
        return idx.mean(axis=0) if len(idx) else None

    res.landmarks = landmarks(exact)
    # Linear or cubic resampling blends labels, so exact matching misplaces or loses landmarks of a CORRECT pipeline. The blob
    # reading may only turn such a failure into a pass, when it finds every landmark where it belongs; any other verdict and
    # every number of the exact reading stand unchanged.
    if not all(lm.ok for lm in res.landmarks) and not res.lossless and _values_changed(data, ph, ras_aff):
        for i, cand in enumerate(_soft_landmarks(data, ph, ras_aff)):
            alt = landmarks(cand.get)
            if all(lm.ok for lm in alt):
                res.landmarks = alt
                res.detail = ("landmarks located as blobs (weighted centroids) because output values are not the phantom's labels; "
                              + ("labels read from values" if i == 0 else "labels read from the blobs' intensity order"))
                break

    if res.orientation_ok is False or not all(lm.ok for lm in res.landmarks):
        res.status = "FAIL"
        if not all(lm.ok for lm in res.landmarks):
            res.pattern = diagnose.describe([lm.expected for lm in res.landmarks], [lm.got for lm in res.landmarks],
                                            out_affine=aff, unit_mm=UNIT_TO_MM[contract.units], frame=contract.world_frame,
                                            noise_mm=tol_mm)
    return res


def _ball_mm3(ph: Phantom) -> float:
    """Physical volume of one landmark ball; all balls share it (integer centres, same voxel radius)."""
    return np.count_nonzero(ph.data == 1) * abs(np.linalg.det(ph.affine[:3, :3]))


def _values_changed(data: np.ndarray, ph: Phantom, ras_aff: np.ndarray) -> bool:
    """True when the output's values are not the phantom's whole labels: blended by linear/cubic resampling, or remapped by an
    intensity normalisation. Float output shows it as non-integer values; integer output (nibabel keeps the int16 dtype) as a
    label covering far more physical volume than one ball (rounded edges of other balls take its value), or as a label missing."""
    if not np.issubdtype(data.dtype, np.integer):
        nz = data[np.abs(data) > 0.05]
        if nz.size and float(np.mean(np.abs(nz - np.round(nz)) > 0.05)) > 0.02:
            return True
    vox = abs(np.linalg.det(ras_aff[:3, :3]))
    vols = [np.count_nonzero(np.abs(data - lb) < 0.25) * vox / _ball_mm3(ph) for lb in ph.world_ras_mm]
    return any(v > 1.5 for v in vols) or not all(vols)


def _soft_landmarks(data: np.ndarray, ph: Phantom, ras_aff: np.ndarray) -> list[dict]:
    """Candidate landmark centres (voxel index) for an output whose values no longer equal the phantom's labels: linear or cubic
    resampling blends each ball with the background, and an intensity normalisation (z-score, rescaling) changes every value.
    Each ball is still a blob standing out from the background (the median value); its intensity-weighted centroid is its centre.
    Two readings name the blobs: (1) values kept their meaning, so a blob's plateau value, or its intensity mass over the one
    physical volume every ball shares, is its label; (2) values were remapped monotonically, so with exactly one blob per
    landmark the blobs' mass order is the label order. A label claimed twice, or by no blob, is lost in that reading."""
    from scipy import ndimage

    w = np.nan_to_num(data.astype(float))
    w = w - float(np.median(w))
    if abs(w.min()) > abs(w.max()):                      # a decreasing intensity map
        w = -w
    w = np.clip(w, 0, None)
    if w.max() <= 0:
        return []
    core, n = ndimage.label(w > w.max() / 12)            # label 1 sits at 1/6 of the top label's plateau
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    sizes = np.asarray(ndimage.sum(np.ones_like(w), core, ids))
    keep = ids[sizes >= 0.2 * np.median(sizes)]          # drop specks (cubic ringing)
    region = ndimage.grey_dilation(np.where(np.isin(core, keep), core, 0), size=(5, 5, 5))   # take in the faint tails
    region[w <= 0] = 0
    mass = np.asarray(ndimage.sum(w, region, keep)) * abs(np.linalg.det(ras_aff[:3, :3])) / _ball_mm3(ph)
    with np.errstate(invalid="ignore", divide="ignore"):
        centres = [np.asarray(c) for c in ndimage.center_of_mass(w, region, keep)]
    labels = sorted(ph.world_ras_mm)

    absolute: dict = {}
    for m, c in zip(mass, centres):
        if not np.all(np.isfinite(c)):
            continue
        v = float(w[tuple(np.clip(np.round(c).astype(int), 0, np.array(w.shape) - 1))])
        lb = int(round(v)) if abs(v - round(v)) < 0.25 and round(v) in ph.world_ras_mm else int(round(m))
        if lb in ph.world_ras_mm and (abs(v - lb) < 0.25 or abs(m - lb) < 0.35):
            absolute[lb] = None if lb in absolute else c
    candidates = [{lb: c for lb, c in absolute.items() if c is not None}]
    if len(keep) == len(labels):
        candidates.append({lb: centres[i] for lb, i in zip(labels, np.argsort(mass)) if np.all(np.isfinite(centres[i]))})
    return candidates


def _check_one(*args, **kwargs) -> PhantomResult:
    """The oracle must be total: output it cannot analyse (NaN or all-zero affine, ...) is an ERROR verdict, never an exception."""
    try:
        return _check_one_inner(*args, **kwargs)
    except Exception as e:
        ph = args[2]
        return PhantomResult(ph.seed, "ERROR", detail=f"could not analyse the output: {type(e).__name__}: {str(e)[:150]}")


def check(code_path, contract: SpatialContract | None, seeds=DEFAULT_SEEDS, tol_mm: float = DEFAULT_TOL_MM,
          python: str | None = None, pythonpath: list | None = None, phantoms: list | None = None, run=None) -> Report:
    """Run the oracle on the code at ``code_path`` under ``contract`` (None => read it from the source). ``run`` replaces the
    subprocess runner (same signature as ``_run_code``); api.check_function uses it to run a function in-process."""
    code_path = Path(code_path).resolve()
    if contract is None:
        contract = load_from_source(code_path.read_text(encoding="utf-8"))
    if contract is None:
        return Report("ABSTAIN", None, ["no SPATIAL_CONTRACT declared"], tol_mm=tol_mm)
    problems = contract.problems()
    if problems:
        return Report("ABSTAIN", contract.to_dict(), problems, tol_mm=tol_mm)

    rep = Report("PASS", contract.to_dict(), tol_mm=tol_mm)
    with tempfile.TemporaryDirectory(prefix="sc_oracle_") as d:
        tmp = Path(d)
        for ph in (phantoms if phantoms is not None else [make_phantom(s) for s in seeds]):
            res = _check_one(code_path, contract, ph, tol_mm, tmp, python, pythonpath, run)
            if res.status == "ERROR":      # one retry: a transient Windows file-lock/subprocess blip once turned a control into an ERROR
                again = _check_one(code_path, contract, ph, tol_mm, tmp, python, pythonpath, run)
                again.detail = f"{again.detail} (after 1 retry; first attempt: {res.detail})" if again.status == "ERROR" else again.detail
                res = again
            rep.phantoms.append(res)
    statuses = {p.status for p in rep.phantoms}
    rep.verdict = "ERROR" if "ERROR" in statuses else "FAIL" if "FAIL" in statuses else "PASS"
    return rep
