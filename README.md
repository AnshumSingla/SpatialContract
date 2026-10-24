# SpatialContract

**An agent that repairs orientation and affine bugs in medical-imaging Python code, judged by a physical-coordinate oracle instead of the model's opinion.**

Built for the Nebius x NVIDIA Global AI Hackathon (track: Coding and Agentic Engineering). Imaging only: NIfTI-style pipelines written with nibabel / SimpleITK / MONAI / TorchIO-style code.

## The problem, in 30 seconds
Spatial bugs do not crash. A flipped axis, a stale affine after a crop, or millimetres written where metres were declared produce a valid image in the wrong place, and ordinary unit tests often pass.
In this repo's seeded corpus, 12 of 31 bugs pass the 3-4 ordinary unit tests per pipeline that I wrote as a baseline.

SpatialContract makes the intended geometry explicit and measurable:

```
 script + SPATIAL_CONTRACT {frame, units, orientation}
        |
        v
 gate: usable contract? --no--> ABSTAIN (no model call, no guessed convention)
        | yes
        v
 oracle: asymmetric landmark phantoms -> process() -> where did each landmark land vs where the contract says (mm)
        | pass -> leave the code alone            | fail
        v                                         v
                           Nemotron (Nebius Token Factory, tool call) proposes a patch
                           -> patch guard (contract is immutable, import allow-list)
                           -> oracle again, up to 4 iterations
        | visible oracle passes
        v
 held-out validation the model never sees: hidden phantoms, a fresh phantom generated after the patch,
 the same content stored in another orientation, a non-phantom image checked with SimpleITK
        v
 REPAIRED  or  OVERFIT
```

**See it:** open `docs/index.html` in a browser (self-contained, nine real recorded runs) or run `python scripts/demo.py replay L04_right_multiply`.
The first screen shows a script whose unit tests all pass while the oracle reports landmarks up to 227 mm from where they belong.

## Results at a glance
| question | answer |
|---|---|
| Same model, oracle feedback vs only the repo's unit tests: which repairs more of 31 seeded bugs? | **98% vs 86%**; paired difference +12 points, 95% CI +3 to +22 (3 repetitions, one model) |
| Does it touch code that is already correct? | oracle-gated arm: 0 of 33 runs. Unit-test-only arm: modified 18 of 33, broke 8 |
| Does it guess when the contract is missing or ambiguous? | oracle-gated arm abstains on 21 of 21 runs (a deterministic gate). Unit-test-only arm patched 16 of 21 and abstained twice |
| Does the oracle detect real, fixed bugs from public libraries? | 7 of 7 reproduced at the pre-fix commit (TorchIO 3, nibabel 2, nilearn 1, MONAI 1); only 4 of 7 on the first attempt, see the notes. The target of 10 was not met |
| Can the agent repair a bug that lives inside a library? | **Not implemented, not measured** |

Read these with the limits below: the seeded bugs are the oracle's own classes (circular), the repo-test baseline is mine, and 31 cases is small.

## Quickstart
Python 3.10+. No GPU.
```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"      # Linux/macOS: .venv/bin/pip
pytest                                      # 70 tests, about 3 minutes, no network, no API key needed
python scripts/audit_docs.py                 # re-checks the numbers quoted in the docs against results/*
python scripts/demo.py replay L04_right_multiply   # terminal walkthrough of a recorded real run, no tokens
python scripts/demo.py serve                       # local web UI; paste any script that declares a SPATIAL_CONTRACT
```
To use the real model (Nebius Token Factory):
```bash
cp .env.example .env                       # put your key in NEBIUS_API_KEY; .env is gitignored
python scripts/list_models.py              # shows the Nemotron models, pricing and tool support (needs SC_ALLOW_API=1)
SC_ALLOW_API=1 python scripts/repair.py corpus/seeded/R05_stale_affine.py --budget 30000
SC_ALLOW_API=1 python scripts/demo.py serve --live --budget 60000      # adds the "Repair with Nemotron" button
```
Token Factory calls are disabled unless `SC_ALLOW_API=1` is set for that command, and every call is capped by a persistent token ledger (`SC_TOKEN_CAP`, default 200,000; `results/token_ledger.json`). The key is never printed or logged.
Reproduce the evaluation: `python scripts/evaluate.py --arms arm1 --run-id main` (no tokens), then `SC_ALLOW_API=1 SC_TOKEN_CAP=1500000 python scripts/evaluate.py --arms arm2 arm3 --reps 3 --budget 1000000 --run-id main` (about 600k tokens), then `python scripts/analyze_eval.py main`.

## How it works
- **Contract.** A pipeline declares `SPATIAL_CONTRACT = {"world_frame": "RAS"|"LPS", "units": "mm"|"m"|"um", "affine_convention": "voxel_center_to_world", "array_orientation": optional}`, read with `ast.literal_eval` (never executed). Missing, `unknown` or unsupported fields make everything downstream **abstain**.
- **Oracle** (`src/spatialcontract/oracle.py`, deterministic, no LLM). Writes a seeded label phantom (six distinct balls, off-centre on every axis, random non-RAS orientation, anisotropic voxels, optional obliquity), runs the script's `process(in_path, out_path)` in a subprocess with a scrubbed environment, finds each landmark in the output, maps it through the *output* affine, and reports the error in mm per landmark. Verdicts: PASS, FAIL, ERROR (crash or unreadable output), ABSTAIN. Tolerance is 0.05 mm for lossless pipelines and scales with voxel size for resampling pipelines (capped at 4 mm).
- **Agent** (`src/spatialcontract/agent.py`). Nemotron answers with a `submit_patch` tool call carrying the whole file. A guard rejects syntax errors, imports outside numpy/nibabel/scipy/SimpleITK, `open`/`exec`/`eval`, a changed `process` signature and **any change to the contract**. Rejections and oracle measurements go back to the model; at most four iterations. Every step is logged to JSONL.
- **Held-out validation** (`src/spatialcontract/heldout.py`). After a patch passes the visible oracle: seven hidden phantoms (two strongly oblique), one phantom whose seed derives from the run seed and the patched source, the visible phantoms re-stored in a different valid orientation, and a non-phantom linear-ramp image resampled onto the input grid with SimpleITK and compared voxelwise.
- **Three arms** (`src/spatialcontract/evaluation.py`): (1) oracle only, a generic single-mutation search with no LLM; (2) the same model with only ordinary unit tests as feedback, no gate, plus `report_no_bug` and `abstain` tools; (3) the repair agent. Every final patch from every arm is scored by the same visible oracle plus held-out validation.

## How Nemotron and Nebius Token Factory are used
- **Model:** `nvidia/nemotron-3-super-120b-a12b` through the Nebius Token Factory OpenAI-compatible API (`https://api.tokenfactory.nebius.com/v1/`), default reasoning, tool calling. It is the only model in the evaluation.
- **Why Super and not the cheapest model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` was tried first and repaired 2 of 3 trial cases with default thinking, 1 of 3 with thinking off or low (runaway output, a hallucinated nibabel function, calls ending on the length limit). Super repaired 3 of 3 on the same cases and used fewer tokens per solved case because it reasons less. These are small single runs; the table is in `docs/ENGINEERING_NOTES.md`.
- **Where Token Factory helped:** `GET /models?verbose=true` gave pricing, rate limits and `supported_features`, so models could be chosen and a budget set before the first completion; `usage` on every response made a hard, thread-safe token cap easy; tool calling returned a well-formed call on all 271 evaluation requests with no throttling or transport errors. Complaints and details: `FEEDBACK.md`.
- **Other Nebius tools:** none. No GPU instance was needed. No Tavily call is made.

## Limitations and failure cases
- **Circular seeded results.** The 31 bugs are the oracle's own six classes and each was created by a single edit; the oracle was adjusted after it missed a bug or false-alarmed (phantom bounding-box asymmetry, voxel-scaled tolerance, a lossless tolerance and a strong-oblique bank after real bugs). The seeded numbers are not a blind test. A model-free single-mutation search repairs 77% of them for the same reason.
- **No repair of real library bugs.** The seven real bugs are *detected* (reproduced at the pre-fix commit), but each defect lives inside a library file and the agent edits one small script. Doing this needs file-navigation tools and a patchable worktree. Also fewer than the targeted 10 real bugs were found, from four libraries I chose, and 3 of 7 only reproduced after I changed the oracle or the repro.
- **The measured lift depends on my unit tests.** They flag 19 of 31 bugs; most of the gap between the arms is on the 12 they do not flag (94% vs 72%; on the 19 they do flag it is 100% vs 95%). A stronger suite would shrink it.
- **By-construction numbers.** The oracle-gated arm cannot damage correct code or guess a convention because a deterministic gate decides before the model is called; the meaningful comparison is with the ungated arm.
- **No overfit observed (0 of 91 oracle passes failed held-out validation).** That means the held-out checks have not been shown to catch real LLM overfitting; they are exercised only by a hand-built overfit patch in the tests.
- **Failure case, published on purpose.** On the Fortran-order reshape bug (S06) the model rewrote correct code four times and never touched `order="F"`; the oracle's output was identical each round, so it had no new signal. One further run (S01, `inv` forgotten in a resampler) also failed.
- **Oracle blind spots.** Errors under about one coarse voxel are invisible on resampling pipelines; a bug needing a 50-70 degree oblique affine triggered on only 1 of 8 strong-oblique phantoms; the differential check is a linear ramp and cannot see intensity-dependent bugs; only NIfTI in, NIfTI out.
- **Small samples.** 31 bug cases, 3 repetitions, one model, one prompt, no prompt search. One unexplained transient oracle failure (a correct control reported wrong once, not reproducible in seven full reruns) led to an automatic retry of ERROR verdicts.
- **Not a novel idea.** Phantom oracles for orientation bugs have adjacent prior art (Imaging-101, AutoMedImg). The contribution is the declared contract, abstention, and the measured comparison.
- **Demo server.** `demo/server.py` executes submitted Python in a subprocess with a scrubbed environment and an import allow-list. That is not a security boundary; it binds to localhost only and must not be exposed.

## Repository layout
| path | what |
|---|---|
| `src/spatialcontract/` | contract, phantom, oracle, held-out validation, agent, mutation search, repo-style tests, evaluation, Token Factory client with token ledger |
| `corpus/seeded/` | 49 generated cases: 31 seeded bugs, 11 correct controls, 7 must-abstain (readable scripts + manifest) |
| `corpus/real/` | repro scripts for the 7 real library bugs (pinned commits in `src/spatialcontract/realbugs.py`) |
| `results/` | raw evaluation rows and logs (`results/eval/main`), oracle results, model listing, smoke test |
| `demo/`, `docs/index.html`, `scripts/demo.py` | demo server, recordings, static replay page |
| `docs/ENGINEERING_NOTES.md` | the detailed record: every number with its caveats, what went wrong, what was changed after |
| `docs/VIDEO_SCRIPT.md`, `docs/DEVPOST.md` | 3-minute video script, Devpost text |
| `FEEDBACK.md` | written feedback on Nebius Token Factory and each NVIDIA model used |

## License
MIT (see `LICENSE`).
