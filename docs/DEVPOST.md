# Devpost submission text

Fill in the three links before submitting; nothing below invents them.

- Repository (public, MIT, license visible in About): `[REPO_URL]`
- Demo URL (static replay page `docs/index.html`, e.g. via GitHub Pages): `[DEMO_URL]`
- Video (public YouTube, under 3 minutes): `[VIDEO_URL]`
- Track: Coding and Agentic Engineering
- Image gallery: upload `docs/architecture.png` (architecture diagram, 2480x1220) and a screenshot of the 3D landmark view from the demo page

## Tagline
Finds and repairs the bugs that put medical images in the wrong place, and checks every fix against physics instead of the model's opinion.

## Inspiration
In April 2026 TorchIO fixed a bug that handed RAS affines to SimpleITK without converting them to LPS: every image saved that way moved by up to 123 mm, and nothing crashed. On the data side, wrong orientation in image headers is surprisingly common in public MRI datasets and invisible by eye, because brains are nearly symmetric (Glen et al., 2020). In code this family of bugs, a flipped axis, an affine left stale after a crop, RAS read as LPS, millimetres written where metres were declared, does not crash anything. The image is valid and in the wrong place. In our seeded corpus, 12 of 31 such bugs pass the three or four unit tests we wrote for each pipeline. Coding agents add a second risk: asked to fix a script "reported as wrong", a strong model happily rewrites code that was already right, or guesses a convention nobody declared.

## What it does
A pipeline declares a **spatial contract** (world frame, units, array orientation): one four-field dict, read without executing anything. A deterministic **oracle** pushes asymmetric landmark phantoms, with known physical positions, through the code and reports in millimetres where each landmark landed against where the contract says it must. It also fits the map from expected to actual positions and says it in words: "x and y negated", "shifted by a whole number of output voxels", "scaled by 1000", "not a single affine map".

An **agent** built on NVIDIA Nemotron 3 Super, called through Nebius Token Factory, repairs the code with tool calls; a guard keeps the contract immutable; the oracle re-checks. For bugs that live inside a library, a **runtime trace** runs the code under a profiler on phantoms it gets right and wrong and ranks the library functions that ran, so the agent starts from a short list of suspects instead of searching blind. A patch that passes is then tested on checks the model never saw: hidden phantoms, a fresh phantom generated after the patch, the same image stored in another orientation, and a non-phantom image checked with SimpleITK. If the contract is missing or ambiguous, it **abstains** rather than guessing.

If a pipeline has no contract yet, `spatialcontract init` asks Nemotron to draft one, quoting the source text behind each field and saying whether that text states the intent or only shows what the code does; it never edits the file, because on undocumented code a draft can describe the bug as the intent (it did in 1 of 6 usable drafts on our buggy scripts).

It is usable without the agent: a decorator and `check_function` for in-memory functions, a pytest fixture, a `spatialcontract check` CLI with CI exit codes, and a GitHub Actions example. Scope: one 3D image in, one out (resampling with any interpolation and intensity normalisation are fine); 4D, DICOM series and multi-input registration are not covered yet.

## How we use Nemotron and Token Factory
- `nvidia/nemotron-3-super-120b-a12b` through the Token Factory OpenAI-compatible endpoint, with tool calling. For scripts it returns the whole patched file via `submit_patch`; for libraries it drives `list_dir`, `read_file`, `search`, `replace_lines`, `run_oracle` and `get_suspects` over a private copy of the library.
- We tried `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` first because it is cheapest. On three trial bugs it repaired two with default reasoning and one with reasoning off or low (runaway output, a hallucinated nibabel function, calls that hit the length limit). Super repaired all three and used fewer tokens per solved case because it reasons less.
- Token Factory helped in three concrete ways: the verbose model listing (pricing, rate limits, supported features) let us choose a model and set a budget before spending a token; `usage` on every response made a hard, thread-safe token cap simple; and tool calling returned a well-formed call on every request we made, in short loops and in 40-step library runs, with no throttling or transport errors. We also built the client to stop for good on the first out-of-credits error and to respect a dollar cap, since the promo credits are finite. No other Nebius service and no Tavily call is used.
- Written feedback on both models and on Token Factory is in `FEEDBACK.md`.

## What we measured
**Real bugs from public libraries.** The oracle reproduces 8 of 8 fixed spatial bugs from TorchIO, nibabel, MONAI and nilearn at their pre-fix commits (5 on the first attempt). The libraries' own tests did not: at every pre-fix commit, the test files covering the fixed code passed (0 of 8 caught). The regression tests added with the fixes catch 5 of 8 (2 fixes added none; 1 is inconclusive in our environment). A ninth candidate, TorchIO #1310, is invisible to it by design, and we say why. Correct pipelines (the post-fix version of every real bug, and pipelines that resample with linear or cubic interpolation or normalise intensities) pass; that last group used to raise false alarms until an outside review caught it, and no earlier verdict changed with the fix.

Without any model, the runtime trace puts the file the upstream fix edited in its top 10 for 8 of 8 bugs, and the edited function in the top 5 for 4 of 8. A simple baseline, "functions the driver names first", gets the file into the top 3 more often (6 vs 4 of 8) but the function into the top 5 less often (2 of 8), and fails badly when the bug is away from the called API (MONAI's offset bug: rank 20 with the trace, 899 by name).

Repair inside libraries is the hard part and we report it as it is. With the trace: 4 of 16 runs repaired (3 of the 8 bugs); 9 of 16 runs edited a file the upstream fix edited. With our earlier tool set: 3 of 14 runs repaired, 2 of the 7 bugs; 6 of 14 runs edited the same file as the upstream fix, and every failure ended on a 300k-token cap. On the original seven bugs the trace did not change the repair count (3 of 14 both times).

**Better algorithms, verified harder.** We then tried three more repair methods and judged every library repair with two extra gates (the library's own covering tests must not newly fail; its loader must report the same affine as before). **Staged repair** (show the trace's top suspect functions, ask for three whole-function rewrites per round, let the oracle pick) verified 5 of 16 runs (3 of the 8 bugs) for $0.32 in total, about $0.06 per verified repair against $0.36 for the agent, and fixed TorchIO Crop, which the agent never did. The gates rejected 3 staged repairs that had passed every phantom check: two made TorchIO's reader 134 mm wrong while keeping the round trip right, one broke 18 of MONAI's own tests. Staged first, agent only where it fails, verifies 2 and 4 bugs in the two repetitions against 1 and 3 for the agent alone. A model-free mutation search "fixed" Crop by deleting the crop, which our no-op check and the regression gate both reject. And for scripts, a **deterministic fix** (invert the measured misplacement when it is identical on every phantom) repairs 8 of 31 seeded bugs with no model call, cutting model tokens by 18% at the same 98% repair rate. Both nilearn repairs equal the public 2018 upstream fix, so memorisation is possible; the TorchIO and MONAI repairs differ from upstream and pass every held-out check.

**Real scans.** Phantoms give exact ground truth, so they judge the repairs; we then ran every repair on four real scans (three real acquisitions, one stored in two layouts) where the input scan is the ground truth. All 91 phantom-validated seeded repairs pass on every scan, and for pipelines that only reorient, crop or re-express coordinates they reproduce the input exactly (288 of 288 runs). All 16 library repairs that passed the oracle produce output identical to the upstream fix on every scan (64 of 64), through the same driver. All 31 buggy originals fail on at least one real scan.

**Seeded bugs**, five arms, 31 bugs, 11 correct controls and 7 must-abstain cases, scored identically (visible oracle, then held-out validation); LLM arms ran three times.

| | repaired (31 bugs) | modified correct code (11 controls x 3) | patched instead of abstaining (7 cases x 3) |
|---|---|---|---|
| oracle-only search, no LLM | 77% | 0 of 11 | 0 of 7 (gate) |
| same model + unit tests only | 86% | 18 of 33 (broke 8) | 16 of 21 |
| same model + oracle feedback | **98%** | 0 of 33 (gate) | 0 of 21 (gate) |
| same model, no feedback (extra baseline) | 86% | 16 of 33 (broke 10) | 9 of 21 |
| same model + stronger unit tests, incl. a landmark test (extra baseline) | 97% | 26 of 33 (broke 4) | 14 of 21 |

The paired difference between oracle feedback and our original unit tests is +12 points (95% CI +3 to +22). Against the stronger suite it is +1 point: a good landmark test repairs as well as the oracle.

The table runs the test arms without a gate, as designed. A fair question is what happens if every arm skips the model when its own check passes. Recomputed from the same rows (no new runs): correct code modified 0 of 33 for the original tests, the stronger tests and the oracle alike; repaired 58%, 97% and 98%; must-abstain cases patched 6 of 21, 12 of 21 and 0 of 21. So any check that passes correct code protects it; the ordinary tests do it by also passing 12 of the bugs, and the strong suite (about 90 lines of hand-written landmark tests, one expected-position function per pipeline family) is a miniature oracle. The contract-driven oracle gets the same result from one four-field dict per pipeline, and it is the only check that can say the convention is unknown.

## Limitations (stated up front)
- The seeded bugs are the oracle's own classes, so those results are circular; the oracle was adjusted several times after it missed a bug.
- The unit-test baselines are ours, the stronger one written after the first evaluation. 31 cases and three repetitions is small.
- Library repair is partial and not a blind test: the tool set was changed after failures on the same bugs, the trace was built after seeing those runs, and the libraries' own tests were run for detection only, not as repair feedback.
- The abstention and no-false-alarm numbers for the gated arm are by construction (a deterministic gate decides before the model is called); the gated comparison of the test arms is recomputed, not re-run.
- No new bug was found: every real bug here was already fixed upstream, which is what makes the repairs scorable.
- Overfit was rare: 0 of 91 oracle passes in the seeded evaluation. The held-out checks caught two real LLM overfits (an earlier library run, and an S06 patch that passed the visible oracle on the first try), so their catch rate is two observations, not a measurement.
- Failed runs are in the demo on purpose.

## Challenges
Making the evaluation trustworthy took more work than the agent. For the library runs, our own search tool silently returned "no matches" for natural glob patterns, which made the first full run meaningless; we found it by reading the logs and discarded the run. The oracle too: it missed a crop bug because the phantom's bounding-box margins were palindromic, false-alarmed on a coarse resampler, missed a one-voxel nilearn bug under its tolerance, and crashed on a degenerate affine. Each was fixed and is documented in `docs/ENGINEERING_NOTES.md`. We also found, mid-run, that the model could read a comment in the corpus file describing the bug; we removed the comments and discarded that run. For the localisation ranking we fixed the weights before the first measurement and did not tune them afterwards.

## What we learned
The model repairs well when it gets any check that sees physical positions, and it needs a check it cannot argue with: with no check gating it, every arm rewrote correct code (16 to 26 of 33 runs) and, told to abstain when the convention is unknown, still patched 9 to 16 of 21 cases. A prompt did not stop that; a check did. What the contract adds over a good test suite is that it is written once per pipeline and can say "unknown". We also learned to grill our own oracle: an outside review found that correct interpolating pipelines failed it, which none of our corpora exercised. And the cheaper, more creative repair methods found the oracle's other blind spot, what the driver does not exercise: a search deleted a crop, a model flipped a reader. Checks that come from the library itself (its own tests, its loader) closed it. Inside real libraries, finding the right function is now the easier half: the trace puts the file the real fix changed in its top ten every time, yet the agent still repairs only the simplest bugs. Turning a located function into a correct edit, when the fix needs an understanding of the algorithm (MONAI's offset computation, nibabel's axis assignment), is where it fails.

## What's next
The oracle's millimetre error is a verifiable reward, which makes it a natural signal for reinforcement fine-tuning of a repair model (Token Factory's RL tooling). Contract properties beyond world position (grid centring, flip-equivariance) would close the TorchIO #1310 blind spot. More real bugs from dipy and ANTsPy, an independently written unit-test baseline, and a Nano-for-triage / Super-for-patching split to cut the cost of library runs.

## Try it
`pip install -e ".[dev]"`, then `python scripts/demo.py replay L04_right_multiply` (no API, no tokens), open `docs/index.html`, or run `spatialcontract check examples/inmemory_example.py`. To run the real model: copy `.env.example` to `.env`, add a Token Factory key, and run `SC_ALLOW_API=1 python scripts/repair.py corpus/seeded/R05_stale_affine.py --budget 30000`.

## Built with
Python, NumPy, nibabel, SciPy, SimpleITK, NVIDIA Nemotron 3 Super (120B, A12B) and Nano (30B, A3B), Nebius Token Factory, pytest.
