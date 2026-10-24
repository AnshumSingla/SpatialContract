# Devpost submission text

Fill in the three links before submitting; nothing below invents them.

- Repository (public, MIT, license visible in About): `[REPO_URL]`
- Demo URL (static replay page `docs/index.html`, e.g. via GitHub Pages): `[DEMO_URL]`
- Video (public YouTube, under 3 minutes): `[VIDEO_URL]`
- Track: Coding and Agentic Engineering

## Tagline
Repairs orientation and affine bugs in medical-imaging code, and checks every patch against physics instead of the model's opinion.

## The problem
In medical imaging, a flipped axis, a stale affine after a crop, or millimetres written where metres were declared does not crash anything. The image is valid and in the wrong place, and ordinary unit tests often pass. In our seeded corpus, 12 of 31 such bugs pass the three or four unit tests we wrote for each pipeline.

## What it does
A pipeline declares a **spatial contract** (world frame, units, array orientation). A deterministic **oracle** pushes asymmetric landmark phantoms, with known physical positions, through the code and reports in millimetres where each landmark landed against where the contract says it must. An **agent** built on an NVIDIA Nemotron model, called through Nebius Token Factory, proposes a patch as a tool call; a guard keeps the contract immutable; the oracle re-checks; the loop runs up to four iterations. A patch that passes is then tested on checks the model never saw: hidden phantoms, a fresh phantom generated after the patch, the same image stored in another orientation, and a non-phantom image checked with SimpleITK. If the contract is missing or ambiguous the agent **abstains** rather than guessing a convention.

## How we use Nemotron and Token Factory
- `nvidia/nemotron-3-super-120b-a12b` through the Token Factory OpenAI-compatible endpoint, with tool calling. It reads a small pipeline plus the oracle's measurements and returns the whole patched file via a `submit_patch` call. In the baseline arm it gets unit-test results instead.
- We tried `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` first because it is cheapest. On three trial bugs it repaired two with default reasoning and one with reasoning off or low (runaway output, a hallucinated nibabel function, calls that hit the length limit). Super repaired all three and used fewer tokens per solved case because it reasons less.
- Token Factory helped in three concrete ways: the verbose model listing (pricing, rate limits, supported features) let us choose a model and set a budget before spending a token; `usage` on every response made a hard, thread-safe token cap simple; and tool calling returned a well-formed call on all 271 evaluation requests, with no throttling or transport errors. No other Nebius service and no Tavily call is used.
- Written feedback on both models and on Token Factory is in `FEEDBACK.md`.

## What we measured
Three arms on 31 seeded bugs, 11 correct controls and 7 must-abstain cases, scored identically (visible oracle, then held-out validation); LLM arms ran three times.

| | repaired (31 bugs) | modified correct code (11 controls x 3) | patched instead of abstaining (7 cases x 3) |
|---|---|---|---|
| oracle-only search, no LLM | 77% | 0 of 11 | 0 of 7 (gate) |
| same model + unit tests only | 86% | 18 of 33 (broke 8) | 16 of 21 |
| same model + oracle feedback | **98%** | 0 of 33 (gate) | 0 of 21 (gate) |

The paired difference between oracle feedback and unit tests is +12 points (95% CI +3 to +22). Most of the gap sits on the 12 bugs the unit tests do not flag (94% vs 72%; on the 19 they do flag, 100% vs 95%). The oracle also reproduces 7 of 7 real fixed bugs from TorchIO, nibabel, nilearn and MONAI at their pre-fix commits. We then let the agent loose on those library bugs: it searches, reads and edits a private copy of the library (no hint about which file), re-runs the oracle, and is judged by the same held-out checks. Result: **3 of 14 runs repaired, 2 of the 7 bugs**, 6 of 14 runs edited the same file as the upstream fix, 11 failures that all ended on a 300k-token cap. One repaired patch (TorchIO's SimpleITK frame bug) is a valid fix that differs from upstream; both repaired nilearn patches equal the public upstream fix, so memorisation is possible.

## Limitations (stated up front)
- The seeded bugs are the oracle's own classes, so those results are circular; the oracle was adjusted several times after it missed a bug.
- The unit-test baseline is ours; a stronger suite would shrink the gap. 31 cases and three repetitions is small.
- Library repair is partial (3 of 14 runs). The tool set was changed four times after failures on those same 7 bugs, the first full run was invalid because of a bug in our search tool, there is no unit-test baseline for the real bugs, and only 4 of 7 bugs were caught by the oracle on the first attempt.
- The abstention and no-false-alarm numbers for the oracle-gated arm are by construction (a deterministic gate decides before the model is called).
- No overfit was observed, so the held-out checks have not been shown to catch real LLM overfitting.
- A failed run (a Fortran-order reshape bug) is in the demo on purpose.

## Challenges
Making the evaluation trustworthy took more work than the agent. For the library runs, our own search tool silently returned "no matches" for natural glob patterns, which made the first full run meaningless; we found it by reading the logs and discarded the run. The oracle too: it missed a crop bug because the phantom's bounding-box margins were palindromic, false-alarmed on a coarse resampler, missed a one-voxel nilearn bug under its tolerance, and crashed on a degenerate affine. Each was fixed and is documented in `docs/ENGINEERING_NOTES.md`. We also found, mid-run, that the model could read a comment in the corpus file describing the bug; we removed the comments and discarded that run.

## What we learned
The model repairs well when shown measurements and guesses when not: even told to abstain when the convention is unknown, it patched 16 of 21 such cases. Stopping that needed an external check, not a better prompt.

## What's next
Higher step and token budgets and better localisation for the five library bugs the agent could not fix; a unit-test baseline for the real bugs; more real bugs from dipy, ANTsPy and SimpleITK; a stronger, independently written unit-test baseline for the seeded ones.

## Try it
`pip install -e ".[dev]"`, then `python scripts/demo.py replay L04_right_multiply` (no API, no tokens) or open `docs/index.html`. To run the real model: copy `.env.example` to `.env`, add a Token Factory key, and run `SC_ALLOW_API=1 python scripts/repair.py corpus/seeded/R05_stale_affine.py --budget 30000`.

## Built with
Python, NumPy, nibabel, SciPy, SimpleITK, NVIDIA Nemotron 3 Super (120B, A12B) and Nano (30B, A3B), Nebius Token Factory, pytest.
