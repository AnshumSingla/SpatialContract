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
