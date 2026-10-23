# SpatialContract

> STATUS: scaffold. Nothing below the "Setup" section has been measured yet.

An agent that repairs spatial / orientation / affine bugs in medical-imaging Python code
(nibabel, SimpleITK, MONAI, TorchIO style), verified by a **deterministic physical-coordinate
oracle** instead of the LLM's opinion. The pipeline declares a spatial contract; the oracle
pushes an asymmetric phantom through the code and reports which landmark landed how many mm off;
an NVIDIA Nemotron model (via Nebius Token Factory) proposes patches; the agent abstains when
the contract is missing or ambiguous.

## Setup
```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Linux/macOS: .venv/bin/pip
cp .env.example .env        # then paste your Nebius Token Factory key into .env
python scripts/list_models.py      # zero-token: lists Nemotron model ids
python scripts/smoke_test.py       # budget-capped completion + tool-call test
pytest
```

## Oracle (implemented, step 3)
- `contract.py`: `SPATIAL_CONTRACT = {...}` literal declared in the pipeline (read with `ast.literal_eval`,
  never executed): `world_frame` (RAS/LPS), `units` (mm/m/um), `affine_convention`, optional `array_orientation`.
  Missing / unknown / unsupported values make the oracle return `ABSTAIN` without running the code.
- `phantom.py`: seeded label volume, 6 distinct balls, off-centre on every axis, random non-RAS signed-permutation
  orientation, anisotropic voxels, optional obliquity, random origin. Ground truth is exact world position.
- `oracle.py`: runs `process(in_path, out_path)` in a subprocess with a scrubbed environment (no API key),
  locates each label in the output, maps it through the OUTPUT affine and reports error in mm per landmark.
  Verdicts: PASS / FAIL / ERROR (crash) / ABSTAIN.
- Tested: correct reorientation passes (0.00 mm); axis swap, flipped axis, stale affine, voxel-vs-world,
  wrong units and transposed affine each fail on every individual phantom (worst errors 60 to 98 mm).
  Caveat: the reference pipeline is a lossless reorientation, so interpolation tolerance (`tol_mm=1.5`) is not yet
  exercised by a resampling pipeline.

## How Nemotron / Token Factory are used
_TODO (filled after step 6)._

## Results
_TODO. No numbers yet. Seeded-bug and real-bug results will be reported separately._

## Limitations (known up front)
- The phantom-oracle idea is not novel (adjacent prior art: Imaging-101, AutoMedImg). The
  contribution is the declared contract + abstention + measured repair lift.
- Seeded bugs are circular (we inject what the oracle was built to catch). A real-bug corpus is
  reported separately; if fewer than 5 real bugs reproduce the result is labelled synthetic-only.
- Kill condition: if LLM+oracle does not beat LLM+repo-tests, this is "a CI harness", and we say so.
- Imaging-only scope.

## License
MIT
