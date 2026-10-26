# Feedback on Nebius Token Factory and the NVIDIA Nemotron models

Written from what I actually ran (2026-10-23 and 2026-10-24). About 8.0 million tokens went through Token Factory in total: a 514-token smoke test, a few hundred tokens of probes, about 255k tokens of repair experiments on two models, about 600k tokens for the three-arm evaluation (271 model calls), about 6.4M tokens for the library-repair runs (pilots, one invalid run, and the final 14 runs), and about 0.7M tokens (344 model calls) for two extra baselines run afterwards. Latencies are single samples, not benchmarks.

**Used:** Token Factory (OpenAI-compatible chat completions with tool calling, and the model listing); `nvidia/nemotron-3-super-120b-a12b` (the repair model); `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (tried first, then dropped).
**Listed but not used:** `nvidia/Nemotron-3_5-Lightning`, `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/Nemotron-3.5-Super-VL`. I have no opinion on them.

## Nebius Token Factory

**What I used it for.** The runtime model calls of a repair agent (the model reads a small imaging script plus measurements and returns a patched file as a tool call), 4-way parallel for the evaluation. No other Nebius service.

**Onboarding, zero to hello world.** An API key was already in `.env`. The first successful call was `GET /v1/models?verbose=true`: HTTP 200, 30 models, no tokens spent. That listing was the most useful onboarding surface: per model it gives pricing, context length, `per_request_limits` (tokens and requests per minute), region and `supported_features` (all five Nemotron entries say `["tools","reasoning"]`), so I could pick the cheapest model, confirm tool support and set a token budget before the first completion. The standard `openai` Python client worked unchanged with `base_url=https://api.tokenfactory.nebius.com/v1/`. The first completion (hello) and the first tool call (`add(1234, 4321)`, valid arguments, `finish_reason=tool_calls`) together cost 514 tokens.

**What worked.**
- Tool calling: 271 of 271 evaluation requests (three tools offered, `tool_choice="auto"`, Super-120B) returned a well-formed tool call whose JSON arguments parsed; none ended on `finish_reason=length` at `max_tokens=8000`. (Nano-30B in my trial runs: 19 of 27 calls returned a tool call.)
- Reliability at modest scale: about 600k tokens in roughly 25 minutes with 4 parallel workers, no 429s, no transport errors. I never approached the listed 300 requests/minute for Super.
- `usage` on every response, which made a hard, thread-safe cumulative token cap with a persistent ledger easy to build.
- Parameters pass through `extra_body`: `{"chat_template_kwargs": {"enable_thinking": false}}` cut a hello from 57 to 3 completion tokens, and `{"reasoning_effort": "low"}` gave 17.
- Cost: the whole evaluation cost well under $1 at the listed prices.

**Long agentic loops.** The library-repair runs (7 bugs x 2 repetitions plus pilots, about 6.4M tokens, up to 40 steps and 300k tokens per run, 4 parallel) behaved like the short ones: 442 of 442 model calls in the final run returned a tool call, none ended on `finish_reason=length`, none had malformed arguments (one call named a tool that does not exist). Two things made long loops expensive and are worth knowing: prompt tokens are billed again at every step (a 35-step run costs about 300k tokens even though each step adds a few hundred), and although the model listing shows an `input_cache_read` price for Super ($0.06 against $0.30 per million), `usage.prompt_tokens_details` came back null, so I could not tell whether my repeated prefixes were cached or what they cost. I pruned old tool outputs myself and estimated cost at the full input price. Reporting cached tokens in `usage` would matter a lot for agent loops.

**What did not, and what I would change.**
- I found the reasoning controls above by trial. The model listing only says `supported_features: ["tools","reasoning"]`; the listing or the docs should state which switches each model accepts. This cost me several probe calls.
- Reasoning tokens are billed and count against `max_tokens`, but `usage.completion_tokens` does not separate reasoning from answer. With `max_tokens=6000`, one Nano call spent all 6000 reasoning and returned empty content. A `reasoning_tokens` field in `usage` would remove the guesswork.
- A first attempt on 2026-10-16 from a different network failed with a proxy `403 Forbidden` before reaching the API. Not a Token Factory error, but it cost a debugging round; I record it because it is what zero-to-hello-world looked like on that network.
- Environment note, not Token Factory: the pip-resolved `openai` 3.37.0 depends on `httpx2` rather than `httpx`, so code that imports `httpx` directly breaks.

**Would I use it again?** Yes. For an agent that makes a few thousand small tool-calling requests it was reliable, cheap and transparent about price and limits.

## nvidia/nemotron-3-super-120b-a12b ($0.30 in / $0.90 out per M tokens, 200k tokens/min, 300 requests/min)

- **What for.** The repair model in the final system: given a ~25-line pipeline, a contract and either oracle measurements or unit-test results, return the whole patched file through `submit_patch`; it can also call `report_no_bug` or `abstain` in the baseline arm.
- **What worked.** With oracle measurements it repaired 91 of 93 seeded-bug runs (98%), 85 on the first call; typically about 300-450 completion tokens and 12-16 s. With only unit-test results it repaired 80 of 93 (86%); 12 of the 13 misses were patches the unit tests accepted while the bug remained. Every accepted patch also passed held-out validation (hidden phantoms, a fresh phantom, a paired orientation, a non-phantom check). It never tried to edit the SPATIAL_CONTRACT; the guard rejected only 2 patches in the whole evaluation (both plain syntax errors, one per arm).
- **What did not.** (1) Told to abstain when the convention cannot be determined, it still patched 16 of 21 must-abstain runs (abstained 2, said "no bug" 3). (2) When told a script was reported as wrong, it modified 18 of 33 already-correct pipelines and broke 8 (it said "no bug" in 11). It follows the "reported as wrong" framing more than the "do not guess" rule. (3) It does not localise a one-token bug when the measurements only say "all landmarks off": on a Fortran-order reshape bug it rewrote correct code four times (one failed run in three; a second run repaired it on the third try).
- **Onboarding.** Zero friction beyond choosing it: the same chat and tool-call code that worked for Nano worked unchanged.
- **Would I use it again?** Yes, as a component. The evidence here is that its repairs are good when it is shown measurements, and that stopping it from guessing needed an external check, not a better prompt. My comparison is one model, one prompt, 31 cases and a unit-test baseline that I wrote, so it is evidence about this task, not a general ranking.

## nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B ($0.06 in / $0.24 out per M tokens, 800k tokens/min)

- **What for.** First-choice repair model because it is the cheapest; replaced by Super after a three-case trial.
- **What worked.** Hello and a tool call worked first time. On three trial bugs (stale affine, a crop-index bug, a units bug) with default reasoning it repaired two (the crop-index bug in 2 calls, the units bug in 1).
- **What did not.** The stale-affine bug failed after 4 calls. It hallucinated an API (`nibabel.orientations.ornt_from_axcodes` does not exist) and kept using it for three iterations. With thinking off, one call produced a 23,496-character "patched source" that failed to parse (repetition), and thinking-off repaired 1 of 3 (plus a budget stop). With `reasoning_effort=low`, 7 of 12 calls ended on `finish_reason=length` and 1 of 3 was repaired. Latency was 13 s for a one-word answer on the first call and 40-100 s when it reasoned at length.
- **Onboarding.** Same code path as Super; the only extra work was learning the reasoning switches.
- **Would I use it again?** Not for this repair task: it used more tokens per solved case than Super (46.6k tokens for the three trial cases against 10.8k for Super on the same three, which it solved 3 of 3). It may be fine for simpler jobs; I did not test any.
