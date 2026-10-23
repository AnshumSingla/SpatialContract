"""Seeded-bug corpus: five small imaging pipelines, each with correct controls and injected bugs.

Every case is a standalone ``process(in_path, out_path)`` script written to ``corpus/seeded``
with a ``manifest.json``. Bugs are injected by exact-match source replacement (each ``old`` string
must occur exactly once), so the diff between a bug and its control is visible and minimal.

CIRCULARITY WARNING: these bug classes are the ones the oracle was built for. Results on this
corpus are a sanity check, not evidence about real-world bugs (see corpus/real, step 5).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / "seeded"

CLASSES = ("axis_swap", "flipped_axis", "stale_affine", "voxel_vs_world", "wrong_units", "transposed_affine")

C_RAS_MM = {"world_frame": "RAS", "units": "mm", "affine_convention": "voxel_center_to_world"}
C_RAS_ORI = C_RAS_MM | {"array_orientation": "RAS"}
C_LPS_MM = C_RAS_MM | {"world_frame": "LPS"}
C_RAS_M = C_RAS_MM | {"units": "m"}

# --------------------------------------------------------------------------- base pipelines
_IMPORTS = "import numpy as np\nimport nibabel as nib\nfrom nibabel import orientations as o\n"

BASES = {
    # R: reorient any volume to RAS+ array orientation
    "R": (C_RAS_ORI, '''
def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))
    new = o.apply_orientation(data, xfm)
    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)
    nib.save(nib.Nifti1Image(new, new_aff), out_path)
'''),
    # S: hand-written nearest-neighbour resampler to 2 mm isotropic, RAS-aligned grid
    "S": (C_RAS_MM, '''
def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    vox = 2.0
    corners = np.array([[i, j, k, 1.0] for i in (0, data.shape[0] - 1)
                        for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)])
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    shape = np.ceil((hi - lo) / vox).astype(int) + 1
    out_aff = np.diag([vox, vox, vox, 1.0])
    out_aff[:3, 3] = lo
    ii, jj, kk = np.indices(shape)
    out_idx = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).reshape(-1, 4).T
    world_pts = out_aff @ out_idx
    in_idx = np.rint(np.linalg.inv(aff) @ world_pts)[:3].astype(int)
    ok = np.all((in_idx >= 0) & (in_idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(int(np.prod(shape)), dtype=data.dtype)
    out[ok] = data[in_idx[0, ok], in_idx[1, ok], in_idx[2, ok]]
    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)
'''),
    # C: crop to foreground bounding box with margin; origin must follow the crop
    "C": (C_RAS_MM, '''
def process(in_path, out_path):
    img = nib.load(in_path)
    data = np.asarray(img.dataobj)
    aff = img.affine
    nz = np.argwhere(data > 0)
    lo = np.maximum(nz.min(axis=0) - 3, 0)
    hi = np.minimum(nz.max(axis=0) + 4, data.shape)
    cropped = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    shift = np.eye(4)
    shift[:3, 3] = lo
    new_aff = aff @ shift
    nib.save(nib.Nifti1Image(cropped, new_aff), out_path)
'''),
    # L: hand an ITK/SimpleITK-style LPS world to a downstream tool
    "L": (C_LPS_MM, '''
def process(in_path, out_path):
    img = nib.load(in_path)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), ras_to_lps @ img.affine), out_path)
'''),
    # M: export world coordinates in metres
    "M": (C_RAS_M, '''
def process(in_path, out_path):
    img = nib.load(in_path)
    aff = img.affine.copy()
    aff[:3, :] *= 1e-3
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), aff), out_path)
'''),
}


@dataclass
class Case:
    id: str
    base: str
    kind: str                      # bug | control | abstain
    expect: str                    # FAIL | PASS | ABSTAIN
    desc: str
    bug_class: str | None = None
    reps: list = field(default_factory=list)       # [(old, new)], each old must occur exactly once
    contract: object = "default"                   # "default" | None (omit) | dict
    notes: str = ""

    def source(self) -> str:
        default_contract, body = BASES[self.base]
        for old, new in self.reps:
            if body.count(old) != 1:
                raise ValueError(f"{self.id}: replacement target occurs {body.count(old)}x: {old!r}")
            body = body.replace(old, new)
        c = default_contract if self.contract == "default" else self.contract
        block = "" if c is None else "\nSPATIAL_CONTRACT = " + json.dumps(c, indent=4) + "\n"
        return _IMPORTS + block + body   # no header comment: the id or description would leak the bug to the model


def _bug(id_, base, cls, desc, *reps):
    return Case(id_, base, "bug", "FAIL", desc, cls, list(reps))


SAVE_R = "    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n"
SAVE_S = "    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff), out_path)\n"
SAVE_C = "    nib.save(nib.Nifti1Image(cropped, new_aff), out_path)\n"

CASES: list[Case] = [
    # ------------------------------------------------------------------ R bugs
    _bug("R01_swap_transpose", "R", "axis_swap", "transposes axes 0/1 after reorienting, affine untouched",
         (SAVE_R, "    new = new.transpose(1, 0, 2)\n" + SAVE_R)),
    _bug("R02_swap_target_codes", "R", "axis_swap", "target codes (A,R,S) instead of (R,A,S): world is right, array order violates contract",
         ('("R", "A", "S")', '("A", "R", "S")')),
    _bug("R03_flip_post", "R", "flipped_axis", "flips axis 0 of the data after reorienting, affine untouched",
         (SAVE_R, "    new = new[::-1]\n" + SAVE_R)),
    _bug("R04_flip_pre", "R", "flipped_axis", "flips the last axis of the raw data before computing the orientation transform",
         ("    data = np.asarray(img.dataobj)\n", "    data = np.asarray(img.dataobj)[:, :, ::-1]\n")),
    _bug("R05_stale_affine", "R", "stale_affine", "saves the input affine with the reoriented data",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, aff), out_path)\n")),
    _bug("R06_stale_header", "R", "stale_affine", "builds the output from the input header, inheriting its old sform",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, None, header=img.header), out_path)\n")),
    _bug("R07_voxel_identity", "R", "voxel_vs_world", "writes an identity affine (voxel index treated as world)",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, np.eye(4)), out_path)\n")),
    _bug("R08_voxel_spacing_only", "R", "voxel_vs_world", "affine keeps only voxel sizes, dropping rotation/flip and origin",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, np.diag(list(img.header.get_zooms()[:3]) + [1.0])), out_path)\n")),
    _bug("R09_units_milli", "R", "wrong_units", "affine scaled by 1e-3 (metres) while contract says mm",
         (SAVE_R, "    new_aff[:3, :] *= 1e-3\n" + SAVE_R)),
    _bug("R10_units_cm", "R", "wrong_units", "affine scaled by 0.1 (centimetres) while contract says mm",
         (SAVE_R, "    new_aff[:3, :] *= 0.1\n" + SAVE_R)),
    _bug("R11_affine_transposed", "R", "transposed_affine", "saves new_aff.T",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, new_aff.T), out_path)\n")),
    _bug("R12_affine_inverted", "R", "transposed_affine", "saves the world->voxel inverse instead of voxel->world",
         (SAVE_R, "    nib.save(nib.Nifti1Image(new, np.linalg.inv(new_aff)), out_path)\n")),
    # ------------------------------------------------------------------ S bugs
    _bug("S01_voxel_skips_inverse", "S", "voxel_vs_world", "uses world points as input voxel indices (forgot inv(aff))",
         ("np.rint(np.linalg.inv(aff) @ world_pts)", "np.rint(world_pts)")),
    _bug("S02_stale_affine", "S", "stale_affine", "saves the INPUT affine on the resampled grid",
         (SAVE_S, "    nib.save(nib.Nifti1Image(out.reshape(shape), aff), out_path)\n")),
    _bug("S03_flip_output", "S", "flipped_axis", "flips the resampled array along axis 0 after sampling",
         (SAVE_S, "    nib.save(nib.Nifti1Image(out.reshape(shape)[::-1], out_aff), out_path)\n")),
    _bug("S04_units_cm_voxel", "S", "wrong_units", "output affine voxel size 10x too large (mm vs cm confusion)",
         (SAVE_S, "    out_aff = out_aff.copy()\n    out_aff[:3, :3] *= 10\n" + SAVE_S)),
    _bug("S05_swap_transpose", "S", "axis_swap", "transposes the resampled array, affine untouched",
         (SAVE_S, "    nib.save(nib.Nifti1Image(out.reshape(shape).transpose(1, 0, 2), out_aff), out_path)\n")),
    _bug("S06_reshape_order", "S", "axis_swap", "reshapes with Fortran order, scrambling index order",
         ("out.reshape(shape)", 'out.reshape(shape, order="F")')),
    _bug("S07_affine_transposed", "S", "transposed_affine", "saves out_aff.T",
         (SAVE_S, "    nib.save(nib.Nifti1Image(out.reshape(shape), out_aff.T), out_path)\n")),
    # ------------------------------------------------------------------ C bugs
    _bug("C01_stale_affine", "C", "stale_affine", "crops but keeps the input affine (origin not shifted)",
         ("    new_aff = aff @ shift\n", "    new_aff = aff\n")),
    _bug("C02_index_order", "C", "axis_swap", "crop offset applied with axes reversed (k,j,i)",
         ("shift[:3, 3] = lo", "shift[:3, 3] = lo[::-1]")),
    _bug("C03_voxel_added_to_world", "C", "voxel_vs_world", "adds the voxel offset directly to the world translation",
         ("    new_aff = aff @ shift\n", "    new_aff = aff.copy()\n    new_aff[:3, 3] += lo\n")),
    _bug("C04_flip_cropped", "C", "flipped_axis", "flips the cropped array along axis 2, affine untouched",
         (SAVE_C, "    nib.save(nib.Nifti1Image(cropped[:, :, ::-1], new_aff), out_path)\n")),
    # ------------------------------------------------------------------ L bugs
    _bug("L01_one_axis_only", "L", "flipped_axis", "RAS->LPS flips x only",
         ("np.diag([-1.0, -1.0, 1.0, 1.0])", "np.diag([-1.0, 1.0, 1.0, 1.0])")),
    _bug("L02_rotation_only", "L", "flipped_axis", "flips the rotation block but leaves the origin in RAS",
         ("    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), ras_to_lps @ img.affine), out_path)\n",
          "    a = img.affine.copy()\n    a[:3, :3] = ras_to_lps[:3, :3] @ a[:3, :3]\n"
          "    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), a), out_path)\n")),
    _bug("L03_no_conversion", "L", "stale_affine", "forgets the RAS->LPS conversion and saves the input affine",
         ("ras_to_lps @ img.affine", "img.affine")),
    _bug("L04_right_multiply", "L", "voxel_vs_world", "applies the frame change in voxel space (A @ F instead of F @ A)",
         ("ras_to_lps @ img.affine", "img.affine @ ras_to_lps")),
    # ------------------------------------------------------------------ M bugs
    _bug("M01_zoom_only", "M", "wrong_units", "converts the rotation/zoom block to metres but leaves the origin in mm",
         ("aff[:3, :] *= 1e-3", "aff[:3, :3] *= 1e-3")),
    _bug("M02_centimetres", "M", "wrong_units", "scales by 1e-2 (cm) instead of 1e-3",
         ("aff[:3, :] *= 1e-3", "aff[:3, :] *= 1e-2")),
    _bug("M03_not_converted", "M", "wrong_units", "never converts: affine still in mm while contract says m",
         ("    aff[:3, :] *= 1e-3\n", "")),
    _bug("M04_converted_twice", "M", "wrong_units", "converted twice (1e-6)",
         ("    aff[:3, :] *= 1e-3\n", "    aff[:3, :] *= 1e-3\n    aff[:3, :] *= 1e-3\n")),
]

# ---------------------------------------------------------------------- correct controls
_CONTROLS = [
    Case("K01_R_reference", "R", "control", "PASS", "reference reorient-to-RAS"),
    Case("K02_R_float32", "R", "control", "PASS", "reference plus float32 cast and contiguous copy", reps=[
        ("    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n",
         "    new = np.ascontiguousarray(new, dtype=np.float32)\n    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n")]),
    Case("K03_S_reference", "S", "control", "PASS", "reference 2 mm nearest-neighbour resampler (exercises interpolation tolerance)"),
    Case("K04_S_3mm", "S", "control", "PASS", "same resampler at 3 mm", reps=[("vox = 2.0", "vox = 3.0")]),
    Case("K05_C_reference", "C", "control", "PASS", "reference foreground crop"),
    Case("K06_C_margin5", "C", "control", "PASS", "crop with a 5-voxel margin", reps=[
        ("nz.min(axis=0) - 3", "nz.min(axis=0) - 5"), ("nz.max(axis=0) + 4", "nz.max(axis=0) + 6")]),
    Case("K07_L_reference", "L", "control", "PASS", "reference RAS->LPS"),
    Case("K08_M_reference", "M", "control", "PASS", "reference mm->m"),
    Case("K09_noop_copy", "R", "control", "PASS", "does nothing; contract only demands consistent world coordinates", reps=[
        ('    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))\n'
         '    new = o.apply_orientation(data, xfm)\n'
         '    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)\n', "    new, new_aff = data, aff\n")],
         contract=C_RAS_MM),
    Case("K10_R_as_closest_canonical", "R", "control", "PASS", "library call nib.as_closest_canonical", reps=[
        ('    data = np.asarray(img.dataobj)\n    aff = img.affine\n    xfm = o.ornt_transform(o.io_orientation(aff), o.axcodes2ornt(("R", "A", "S")))\n'
         '    new = o.apply_orientation(data, xfm)\n    new_aff = aff @ o.inv_ornt_aff(xfm, data.shape)\n    nib.save(nib.Nifti1Image(new, new_aff), out_path)\n',
         "    nib.save(nib.as_closest_canonical(img), out_path)\n")]),
    Case("K11_S_nibabel_resample", "S", "control", "PASS", "nibabel.processing.resample_to_output(order=0) at 2 mm", reps=[
        (BASES["S"][1].split("def process(in_path, out_path):\n")[1],
         "    import nibabel.processing as nproc\n    img = nib.load(in_path)\n"
         "    nib.save(nproc.resample_to_output(img, 2.0, order=0), out_path)\n")]),
]

# ---------------------------------------------------------------------- must-abstain cases
_ABSTAIN = [
    Case("A01_no_contract_clean", "R", "abstain", "ABSTAIN", "correct code, no contract declared", contract=None),
    Case("A02_no_contract_buggy", "R", "abstain", "ABSTAIN", "stale-affine bug, no contract declared", contract=None, bug_class="stale_affine",
         reps=[(SAVE_R, "    nib.save(nib.Nifti1Image(new, aff), out_path)\n")]),
    Case("A03_no_contract_buggy_S", "S", "abstain", "ABSTAIN", "flip bug in the resampler, no contract declared", contract=None,
         bug_class="flipped_axis", reps=[(SAVE_S, "    nib.save(nib.Nifti1Image(out.reshape(shape)[::-1], out_aff), out_path)\n")]),
    Case("A04_units_unknown", "R", "abstain", "ABSTAIN", "contract with units 'unknown'", contract=C_RAS_ORI | {"units": "unknown"}),
    Case("A05_corner_convention", "C", "abstain", "ABSTAIN", "contract declares voxel-corner affine (unsupported/ambiguous)",
         contract=C_RAS_MM | {"affine_convention": "voxel_corner_to_world"}, bug_class="stale_affine",
         reps=[("    new_aff = aff @ shift\n", "    new_aff = aff\n")]),
    Case("A06_frame_missing", "L", "abstain", "ABSTAIN", "contract omits world_frame",
         contract={"units": "mm", "affine_convention": "voxel_center_to_world"}),
    Case("A07_units_ambiguous", "M", "abstain", "ABSTAIN", "units declared as 'mm/m'", contract=C_RAS_MM | {"units": "mm/m"},
         bug_class="wrong_units", reps=[("    aff[:3, :] *= 1e-3\n", "")]),
]

ALL_CASES = CASES + _CONTROLS + _ABSTAIN


def build(out_dir: Path = CORPUS) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in ALL_CASES:
        (out_dir / f"{c.id}.py").write_text(c.source(), encoding="utf-8", newline="\n")
        manifest.append({"id": c.id, "file": f"{c.id}.py", "base": c.base, "kind": c.kind, "expect": c.expect,
                         "bug_class": c.bug_class, "description": c.desc})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest


def evaluate(corpus: Path = CORPUS, workers: int = 4, seeds=(0, 1, 2)) -> list[dict]:
    """Run the oracle on every case in the manifest; returns one row per case."""
    from . import oracle

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))

    def one(m):
        rep = oracle.check(corpus / m["file"], None, seeds=seeds)
        w = rep.worst()
        return m | {"verdict": rep.verdict, "correct": rep.verdict == m["expect"],
                    "worst_error_mm": None if w is None else (None if w[0] == float("inf") else round(w[0], 3)),
                    "landmark_lost": bool(w and w[0] == float("inf")),
                    "orientation_violation": any(p.orientation_ok is False for p in rep.phantoms),
                    "feedback": rep.feedback()}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, manifest))


def summarize(rows: list[dict]) -> dict:
    bugs = [r for r in rows if r["kind"] == "bug"]
    ctr = [r for r in rows if r["kind"] == "control"]
    ab = [r for r in rows if r["kind"] == "abstain"]
    by_class = {c: [sum(r["correct"] for r in bugs if r["bug_class"] == c), sum(r["bug_class"] == c for r in bugs)]
                for c in CLASSES}
    return {
        "bugs_detected": [sum(r["correct"] for r in bugs), len(bugs)],
        "detected_by_class": by_class,
        "false_alarms": [sum(not r["correct"] for r in ctr), len(ctr)],
        "abstain_correct": [sum(r["correct"] for r in ab), len(ab)],
        "missed_bugs": [r["id"] for r in bugs if not r["correct"]],
        "false_alarm_ids": [r["id"] for r in ctr if not r["correct"]],
    }
