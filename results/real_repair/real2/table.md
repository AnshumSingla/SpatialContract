Runs: 7 (7 bugs x 1 repetitions). Model: nvidia/nemotron-3-super-120b-a12b.

- **Repaired** (visible oracle + held-out validation): 1/7 runs (14%, Wilson 95% 3-51%)
- Passed the visible oracle: 2/7; of those, failed held-out validation (OVERFIT): 1
- Edited a file the upstream fix also edited (localisation): 5/7
- Bugs repaired at least once: 1/7
- Tokens per run: median 203,561; total 1,185,289

| bug | rep | outcome | edited files | upstream file edited | patch identical to upstream | oracle runs | steps | tokens |
|---|---|---|---|---|---|---|---|---|
| monai_spacing_offset | 0 | FAILED | array.py | no | no | 1 | 27 | 204,137 |
| nibabel_conform | 0 | FAILED | processing.py | yes | partly | 1 | 23 | 200,074 |
| nibabel_ioorient | 0 | FAILED | - | no | - | 1 | 29 | 203,561 |
| nilearn_reorder | 0 | REPAIRED | resampling.py | yes | yes | 1 | 11 | 48,519 |
| torchio_crop | 0 | FAILED | crop.py | yes | no | 2 | 28 | 207,300 |
| torchio_reorient | 0 | OVERFIT | reorient.py | yes | partly | 2 | 15 | 114,846 |
| torchio_sitk_ras_lps | 0 | FAILED | image.py | yes | partly | 3 | 25 | 206,852 |