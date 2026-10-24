Runs: 14 (7 bugs x 2 repetitions). Model: nvidia/nemotron-3-super-120b-a12b.

- **Repaired** (visible oracle + held-out validation): 3/14 runs (21%, Wilson 95% 8-48%)
- Passed the visible oracle: 3/14; of those, failed held-out validation (OVERFIT): 0
- Edited a file the upstream fix also edited (localisation): 6/14
- Bugs repaired at least once: 2/7
- Tokens per run: median 304,365; total 3,690,088

| bug | rep | outcome | edited files | upstream file edited | patch identical to upstream | oracle runs | steps | tokens |
|---|---|---|---|---|---|---|---|---|
| monai_spacing_offset | 0 | FAILED | - | no | - | 1 | 34 | 304,576 |
| monai_spacing_offset | 1 | FAILED | - | no | - | 1 | 36 | 306,564 |
| nibabel_conform | 0 | FAILED | - | no | - | 2 | 34 | 302,142 |
| nibabel_conform | 1 | FAILED | processing.py | yes | partly | 2 | 34 | 300,715 |
| nibabel_ioorient | 0 | FAILED | - | no | - | 1 | 36 | 304,365 |
| nibabel_ioorient | 1 | FAILED | - | no | - | 0 | 38 | 303,418 |
| nilearn_reorder | 0 | REPAIRED | resampling.py | yes | yes | 2 | 10 | 40,128 |
| nilearn_reorder | 1 | REPAIRED | resampling.py | yes | yes | 1 | 9 | 37,769 |
| torchio_crop | 0 | FAILED | crop.py | yes | no | 2 | 28 | 306,979 |
| torchio_crop | 1 | FAILED | - | no | - | 6 | 36 | 308,427 |
| torchio_reorient | 0 | FAILED | reorient.py | yes | partly | 3 | 40 | 307,548 |
| torchio_reorient | 1 | FAILED | - | no | - | 4 | 38 | 300,705 |
| torchio_sitk_ras_lps | 0 | FAILED | - | no | - | 1 | 36 | 310,203 |
| torchio_sitk_ras_lps | 1 | REPAIRED | image.py | yes | partly | 1 | 33 | 256,549 |