# Feedback on Nebius Token Factory and NVIDIA models

Kept as I go. Only things I actually observed are recorded here.

## Nebius Token Factory (OpenAI-compatible API)
- Onboarding zero -> hello world (observed 2026-10-23): key was already in `.env`; `GET /v1/models?verbose=true` returned 200 with 30 models in one call, no tokens spent. The verbose listing exposes `pricing`, `context_length`, `supported_features` (e.g. `["tools","reasoning"]`), `per_request_limits` and region per model, which made picking the cheapest Nemotron and confirming tool support possible without a single completion. That was genuinely useful.
- Standard `openai` Python SDK worked unchanged with `base_url=https://api.tokenfactory.nebius.com/v1/`; `usage` is returned on every response, which made a client-side token ledger easy.
- Friction: a first attempt on 2026-10-16 from a different network failed with a proxy `403 Forbidden` before reaching the API (not a Token Factory error; recorded because it cost a debugging round). The 2026-10-23 run on the current network was fine.
- Friction: the pip-resolved `openai` 3.37.0 depends on a package named `httpx2` rather than `httpx`; code that does `import httpx` directly breaks. I switched my listing script to stdlib `urllib`.

## NVIDIA Nemotron (model id used: nvidia/nemotron-3-super-120b-a12b, to confirm via /models)
Models listed as Nemotron on Token Factory (from `/models`, 2026-10-23): `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` and `nvidia/Nemotron-3_5-Lightning` ($0.06 in / $0.24 out per M tokens), `nvidia/nemotron-3-super-120b-a12b` ($0.30 / $0.90), `nvidia/Nemotron-3-Ultra-550b-a55b` ($1.00 / $3.00), `nvidia/Nemotron-3.5-Super-VL`. All advertise `tools` + `reasoning`.

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (smoke test only so far)
- What it is used for: planned first-choice repair model (cheapest); so far only a hello + tool-call smoke test.
- What worked: plain chat returned "Hello." (49 completion tokens for a 2-token answer, i.e. hidden reasoning is billed). Tool call `add(1234, 4321)` came back as a valid structured call, `finish_reason=tool_calls`, args correct, 305 prompt + 135 completion tokens, 1.5 s. Whole smoke test cost 514 tokens.
- What did not / cautions: first call latency was 13 s (cold start?; single sample, not a benchmark). Reasoning tokens count against `max_tokens`, so small caps risk truncated answers on real repair prompts. Repair quality is NOT yet evaluated.
- Would use again: undecided; depends on step 6/7 results.


## Step 6 observations (repair agent; all numbers from results/runs/*, 2026-10-24)
### Nebius Token Factory
- Tool calling: with `tools=[submit_patch]` and `tool_choice="auto"`, 16 of 16 Super-120B calls came back as a well-formed tool call whose `arguments` parsed as JSON (patched files up to about 1.4k characters). Nano-30B (excluding the discarded first run): 19 of 27 calls did; the rest were empty content with `finish_reason=length`, or plain text.
- Reasoning controls that worked through the OpenAI-compatible API: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (hello-world 57 -> 3 completion tokens) and `extra_body={"reasoning_effort": "low"}` (-> 17). I found these by trial, not from the model listing; the `/models` entry only says `supported_features: ["tools","reasoning"]`. A line in the docs or in the model entry would have saved calls.
- Reasoning tokens are billed and count against `max_tokens`: with `max_tokens=6000`, one Nano call spent all 6000 on reasoning and returned empty content (`finish_reason=length`). `usage.completion_tokens` does not split reasoning from answer, so I could only estimate the reasoning share from the length of `reasoning_content`.
- Latency (single samples, not a benchmark): Super 12-16 s per single-call repair; Nano 40-100 s when it reasons at length.
- Rate limits from `/models`: Super 200k tokens/min and 300 req/min; Nano 800k tokens/min. I never hit a limit.
- `usage` on every response made the hard token ledger trivial, and `/models` pricing allowed cost estimates before spending.

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B (repair)
- Worked: 2 of 3 trial cases with default thinking (a crop-index bug in 2 calls, a units bug in 1).
- Did not work: stale-affine bug failed after 4 calls; it hallucinated an API (`nibabel.orientations.ornt_from_axcodes` does not exist) and kept using it for three iterations; with thinking off, one call generated a 23,496-character "patched source" (repetition) that failed to parse; with `reasoning_effort=low`, 7 of 12 calls hit the length limit.
- Would I use it again for this task: not as the repair model. It used more tokens per solved case than Super.

### nvidia/nemotron-3-super-120b-a12b (repair)
- What for: the repair model (reads a pipeline of about 25 lines plus oracle measurements, returns the whole patched file via a tool call).
- Worked: 11 of 12 seeded bugs repaired across two runs, almost always in one call (about 300-450 completion tokens, 12-16 s). Every accepted patch also passed the held-out validation. It never tried to edit the SPATIAL_CONTRACT (the guard did not fire on a Super call).
- Did not work: S06 (Fortran-order reshape): four rewrites of unrelated correct code. It does not localise a one-token bug when the measurements only say "all landmarks off".
- Would use again: yes for this task. The three-arm comparison (step 7) will say whether the oracle feedback is what makes it work.
