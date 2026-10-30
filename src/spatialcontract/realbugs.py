"""Real-bug corpus (K2): fixed spatial bugs in public imaging libraries, reproduced at the pre-fix commit.

Each case pins a library commit pair (parent of the fix = PRE, the fix itself = POST) and a small
``process(in_path, out_path)`` script in ``corpus/real`` that calls the library. The oracle is run
on both: a reproduced bug means PRE fails and POST passes. Library source is imported straight from
a git worktree (``scripts/setup_real_bugs.py`` creates them under ``_work/``); dependencies come
from one of two pinned virtualenvs. Nothing here involves an LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "_work"
CORPUS = ROOT / "corpus" / "real"
ENVS = {
    "legacy": WORK / "env_legacy" / "Scripts" / "python.exe",   # numpy 1.23, torch 2.2, nibabel 4.0.2
    "modern": WORK / "env_modern" / "Scripts" / "python.exe",   # current numpy/torch/nibabel
}
REPOS = {
    "torchio": "https://github.com/fepegar/torchio.git",
    "nilearn": "https://github.com/nilearn/nilearn.git",
    "nibabel": "https://github.com/nipy/nibabel.git",
    "MONAI": "https://github.com/Project-MONAI/MONAI.git",
}


@dataclass
class RealCase:
    id: str
    repo: str
    pre: str                 # full sha, parent of the fix
    post: str                # full sha of the fix
    env: str                 # legacy | modern
    bug_class: str           # one of seeded.CLASSES (closest)
    summary: str
    fix_date: str
    src_subdir: str = ""     # "src" for the TorchIO 2.x layout
    fix_added_regression_test: bool = True
    notes: str = ""
    seeds: tuple = ()          # visible phantom bank override (empty => default 0,1,2)
    hidden_seeds: tuple = ()   # held-out bank override (empty => heldout.HIDDEN_SEEDS)

    def worktree(self, which: str) -> Path:
        return WORK / "wt" / f"{self.id}_{which}"

    def pythonpath(self, which: str) -> list:
        # _work/shims holds a fake dist-info so importlib.metadata.version("torchio") works from a plain worktree
        return [self.worktree(which) / self.src_subdir, WORK / "shims"]

    @property
    def script(self) -> Path:
        return CORPUS / f"{self.id}.py"


CASES = [
    RealCase("torchio_crop", "torchio", "87ebad7d5ad1ddc096ee85c0d67959c4243a4ee6", "7e1decfb346272e98ebaca426691c20c7d5a6f13",
             "legacy", "stale_affine", "Crop computed new_affine but never assigned it, so cropped images kept the old origin (issue #351)",
             "2020-11-05", fix_added_regression_test=False),
    RealCase("nilearn_reorder", "nilearn", "5eaeea35758c084de2e50fe35c181143e04154c9", "613c0d76e50c7e7149ea42d61bae28df397448af",
             "legacy", "flipped_axis", "reorder_img off-by-one: origin shifted by one voxel when flipping the y/z axes",
             "2018-01-10", hidden_seeds=(1000, 1002, 1003, 1004, 250, 251),
             notes="hidden seed 1001 removed: the upstream fix itself leaves a 0.87 mm error on it (unrelated residual issue)"),
    RealCase("nibabel_conform", "nibabel", "6e1929815f29c1a181cadc8cdb5bf2b39f6a3074", "07fa2543dd61bb7d301ee0c554ab97a540e052f1",
             "legacy", "axis_swap", "conform() re-oriented the output with wrong axis labels, so the result was not in the requested orientation",
             "2020-04-08", fix_added_regression_test=False),
    RealCase("monai_spacing_offset", "MONAI", "94673a97986327b09d9d5c1e26695ea8c6b9c178", "d76adc73791dc83aa70481b84f23fe9075160ef7",
             "legacy", "voxel_vs_world", "Spacing(scale_extent=True) computed a wrong output offset for oblique affines (issue #5253)",
             "2022-10-04"),
    RealCase("nibabel_ioorient", "nibabel", "0f61bcbc1c2e4c47291e6ed5ba4c97b24f083002", "927be9130b42ff636f665ece106e8be6cc44208d",
             "modern", "axis_swap", "io_orientation handled competing axes in input order, so strongly oblique affines got inconsistent labels",
             "2025-12-19", notes="fixed across two commits; POST is the second; needs the strong-oblique phantom bank",
             seeds=tuple(range(200, 208)), hidden_seeds=tuple(range(208, 224))),
    RealCase("torchio_reorient", "torchio", "83c4f08f1a0d3e59fe8681d0c0772b6a99ff6cc0", "95b1b7446642dd986475ab426001d0332087f23f",
             "modern", "axis_swap", "Reorient permuted before flipping and built the affine from the post-permutation shape",
             "2026-04-14", src_subdir="src"),
    RealCase("torchio_sitk_ras_lps", "torchio", "a5102eeaa9a22866d9d64ebf5d11b4c9004882ac", "09138e0e0cebed1bace4e72b1b1eddcdbe4dd375",
             "modern", "flipped_axis", "SimpleITK writer received RAS affines without RAS->LPS conversion, double-flipping x/y on save",
             "2026-04-05", src_subdir="src"),
    RealCase("monai_orientation_lps", "MONAI", "d4ba52e31574982300b37a99305d772ba41a1ac4", "0968da200ec86a4fb4216acc9257ab4a8eabce41",
             "modern", "flipped_axis", "Orientation ignored meta['space']=LPS and read LPS affines as RAS, so axcodes='RAS' produced an LPS array (PR #8473, issue #8467)",
             "2025-09-01", notes="world positions stay correct (affine is updated consistently); only the declared array_orientation contract catches it"),
]


STRONG_OBLIQUE_BANK = tuple(range(200, 208))


def check_case(case: RealCase, seeds=None) -> dict:
    """Run the oracle on PRE and POST of one case. Returns a JSON-serialisable row."""
    from . import oracle

    seeds = seeds or (case.seeds if case.seeds else (0, 1, 2))
    py = ENVS[case.env]
    row = {"id": case.id, "repo": case.repo, "env": case.env, "bug_class": case.bug_class, "summary": case.summary,
           "fix_date": case.fix_date, "pre_sha": case.pre, "post_sha": case.post,
           "fix_added_regression_test": case.fix_added_regression_test, "notes": case.notes}
    for which in ("pre", "post"):
        if not case.worktree(which).exists() or not py.exists():
            row[which] = {"verdict": "NOT_RUN", "reason": "worktree or environment missing"}
            continue
        rep = oracle.check(case.script, None, seeds=seeds, python=str(py), pythonpath=case.pythonpath(which))
        w = rep.worst()
        row[which] = {"verdict": rep.verdict,
                      "worst_error_mm": None if w is None or w[0] == float("inf") else round(w[0], 3),
                      "landmark_lost": bool(w and w[0] == float("inf")),
                      "feedback": rep.feedback()}
    pre, post = row["pre"]["verdict"], row["post"]["verdict"]
    row["reproduced"] = pre == "FAIL" and post == "PASS"
    return row


def save_results(rows: list, path: Path = ROOT / "results" / "real_bugs_oracle.json") -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
