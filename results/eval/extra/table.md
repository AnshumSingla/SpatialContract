Unit tests flag 19 of 31 bugs (weak suite) and 31 of 31 (stronger suite).

### Repair on the 31 seeded bugs (mean over repetitions per case; 95% bootstrap CI over cases)

| arm | repaired (visible + held-out) | tokens / case | LLM calls / case |
|---|---|---|---|
| 0. LLM, no feedback | 86% [75%, 95%] | 1714 | 1.0 |
| 2. LLM + weak unit tests | 86% [75%, 95%] | 1891 | 1.1 |
| 2s. LLM + stronger unit tests | 97% [91%, 100%] | 2572 | 1.2 |
| 3. LLM + oracle feedback | 98% [95%, 100%] | 3213 | 1.2 |

Paired difference, arm 3 - arm0: **+12 points** (95% CI +3 to +22).

Paired difference, arm 3 - arm2: **+12 points** (95% CI +3 to +22).

Paired difference, arm 3 - arm2s: **+1 points** (95% CI +0 to +3).

### Split by whether the WEAK unit tests flag the bug at the start

| arm | weak tests flag it (n=19) | weak tests do not flag it (n=12) |
|---|---|---|
| 0. LLM, no feedback | 98% [95%, 100%] | 67% [44%, 86%] |
| 2. LLM + weak unit tests | 95% [89%, 100%] | 72% [47%, 94%] |
| 2s. LLM + stronger unit tests | 100% [100%, 100%] | 92% [78%, 100%] |
| 3. LLM + oracle feedback | 100% [100%, 100%] | 94% [86%, 100%] |

### Correct controls (11 cases x repetitions): modified correct code, and broke it

| arm | modified | broke it |
|---|---|---|
| 0. LLM, no feedback | 16/33 | 10/33 |
| 2. LLM + weak unit tests | 18/33 | 8/33 |
| 2s. LLM + stronger unit tests | 26/33 | 4/33 |
| 3. LLM + oracle feedback | 0/33 | 0/33 |

### Must-abstain cases (7 x repetitions)

| arm | abstained | patched instead |
|---|---|---|
| 0. LLM, no feedback | 6/21 | 9/21 |
| 2. LLM + weak unit tests | 2/21 | 16/21 |
| 2s. LLM + stronger unit tests | 3/21 | 14/21 |
| 3. LLM + oracle feedback | 21/21 | 0/21 |

Harness errors: none.