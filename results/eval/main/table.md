### Repair on the 31 seeded bugs (mean over repetitions per case; 95% bootstrap CI over cases)

| arm | repaired (visible + held-out) | passed visible oracle | overfit among visible passes | tokens / case | LLM calls / case | oracle screenings / case |
|---|---|---|---|---|---|---|
| 1. oracle only (no LLM) | 77% [61%, 90%] | 77% [61%, 90%] | 0/24 (0%-14%, Wilson) | 0 | 0.0 | 163 |
| 2. LLM + repo tests | 86% [75%, 95%] | 86% [75%, 95%] | 0/80 (0%-5%, Wilson) | 1891 | 1.1 | 0 |
| 3. LLM + oracle feedback | 98% [95%, 100%] | 98% [95%, 100%] | 0/91 (0%-4%, Wilson) | 3213 | 1.2 | 0 |

Paired difference, arm 3 - arm 2, repaired rate: **+12 points** (95% CI +3 to +22).
Paired difference, arm 3 - arm 1: +20 points (95% CI +6 to +35).

### Arm comparison split by whether the repo tests flag the bug at the start (19 flagged / 12 not flagged)

| arm | repo tests flag it (n=19) | repo tests do not flag it (n=12) |
|---|---|---|
| 1. oracle only (no LLM) | 79% [58%, 95%] | 75% [50%, 100%] |
| 2. LLM + repo tests | 95% [89%, 100%] | 72% [47%, 94%] |
| 3. LLM + oracle feedback | 100% [100%, 100%] | 94% [86%, 100%] |

### Repaired rate by bug class (case means; small n per class)

| class | n | 1. oracle only (no LLM) | 2. LLM + repo tests | 3. LLM + oracle feedback |
|---|---|---|---|---|
| axis_swap | 5 | 80% | 100% | 93% |
| flipped_axis | 6 | 83% | 94% | 100% |
| stale_affine | 5 | 60% | 87% | 100% |
| voxel_vs_world | 5 | 40% | 47% | 93% |
| wrong_units | 7 | 100% | 90% | 100% |
| transposed_affine | 3 | 100% | 100% | 100% |

### Correct controls (11): did the arm touch or break code that was already right?

| arm | modified a correct pipeline | broke it (fails oracle or held-out afterwards) |
|---|---|---|
| 1. oracle only (no LLM) | 0/11 (0%-26%) | 0/11 (0%-26%) |
| 2. LLM + repo tests | 18/33 (38%-70%) | 8/33 (13%-41%) |  (said 'no bug': 11/33)
| 3. LLM + oracle feedback | 0/33 (0%-10%) | 0/33 (0%-10%) |

### Abstention (the 7 must-abstain cases; abstentions elsewhere count against precision)

| arm | abstained on must-abstain | patched instead (guessed a convention) | abstained on cases it should have handled | precision | recall |
|---|---|---|---|---|---|
| 1. oracle only (no LLM) | 7/7 | 0/7 | 0/42 | 100% | 100% |
| 2. LLM + repo tests | 2/21 | 16/21 | 4/126 | 33% | 10% |
| 3. LLM + oracle feedback | 21/21 | 0/21 | 0/126 | 100% | 100% |

Arms 1 and 3 abstain through the same deterministic gate on the contract, so their abstention numbers measure the gate and are 100% by construction.

Rows: 343. Harness errors: none.