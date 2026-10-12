# Filler in LLM answers on synthetic reports — t4_n20 (package v1.0.0)

Readers: main = Qwen/Qwen2.5-3B-Instruct; phi3.5-mini = microsoft/Phi-3.5-mini-instruct.

## T1. Filler map on simulated responses with known filler

Synthetic reports; each response = the requested facts + planted filler of known type (none / 2 / 5 fragments, or 2 with one fact left out). Recall counts planted filler fragments (95% CI, bootstrap over items); a repeat is resolved when no fact is stated twice after deletion; W error = measured − true filler share. 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader, tolerance | responses (with filler) | no fact lost | precision | recall, planted fragments | recall per response with filler | repeats resolved | MAE of W | mean error of W |
|---|---|---|---|---|---|---|---|---|
| main, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 0.953 [0.903, 0.989] | 0.953 | 0.846 | 0.036 | -0.036 |
| main, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 0.984 [0.965, 1.000] | 0.983 | 0.846 | 0.022 | -0.022 |
| main, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.926 [0.878, 0.964] | 0.926 | 0.846 | 0.041 | -0.041 |
| phi3.5-mini, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.923 | 0.013 | -0.013 |
| phi3.5-mini, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.962 | 0.012 | -0.012 |
| phi3.5-mini, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.958 [0.921, 0.985] | 0.941 | 0.885 | 0.026 | -0.026 |

Deleted / planted by type (τ = 0.5): main: closing 21/23, distractor 9/9, elaboration 20/21, enumeration 20/21, hedge 29/31, preamble 21/22, restatement 24/25, unrequested 37/38; phi3.5-mini: closing 23/23, distractor 9/9, elaboration 21/21, enumeration 21/21, hedge 31/31, preamble 22/22, restatement 25/25, unrequested 38/38.

Planted filler left in (τ = 0.5):
- main: of 9 planted filler fragments left in, 56% cost more than the tolerance to delete even alone (the reader uses them); 44% could go alone but not together with the fragments deleted before them.

| reader check | AUC present vs absent | gold unreadable | wrong-statement detection | nats per fact |
|---|---|---|---|---|
| main | 0.977 | 0.000 | 1.000 | 13.811 |
| phi3.5-mini | 1.000 | 0.000 | 1.000 | 13.815 |

Baselines on the same simulated responses. Every method on the same responses: the filler map of each reader, the LLM judge and Selective Context when they were run (drybench baselines), and three cheap metrics. Fragment level, removable (filler or wrong statement) vs needed: F1, Cohen's κ, and the AUC of the method's score (the judge's p(filler) + p(wrong); Selective Context: lower self-information = more removable). Response level, removable share by characters: Spearman with the true share; 'shorter but more removable' = pairs whose true share differs by ≥ 0.2 where the more removable response is the shorter one, share the method ranks higher (ties apart); partial Spearman with the true share given length [95% CI, bootstrap over items]. Selective Context deletes, least informative fragment first, as many characters as the first reader's filler map (equal budget); the judge's share is its expected removable share. Planted responses: the true share is the planted filler and wrong statements (a summary that repeats the facts is planted filler); a fact stated twice may go either way, so such fragments are left out of the fragment-level scores. First table: the responses every method scores; second: also the answers from which a reader reads none of the requested facts — the filler map's verdict on such an answer is that it delivers nothing, so all of it counts as removable (as in the results tables: a failed answer); answers whose gold the reader cannot read stay out.

96 of 96 responses scored by every method (filler map (reader main) 96; filler map (reader phi3.5-mini) 96); 347 fragments with a truth label; 288 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.976 | 0.948 | 0.976 | 0.952 | 0.962 | 0.000 | +0.93 [+0.86, +0.98] |
| filler map (reader phi3.5-mini) | 1.000 | 1.000 | 1.000 | 0.974 | 1.000 | 0.000 | +0.96 [+0.93, +0.99] |
| length (characters) |  |  |  | 0.630 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.747 | 0.469 | 0.003 | +0.59 [+0.46, +0.71] |
| stop-word share |  |  |  | 0.067 | 0.625 | 0.014 | +0.14 [-0.09, +0.34] |

## T2. Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 260 | 27.312 | 0.105 | 0.060 | 0.045 | 0.720 | 0.474 | 0.024 | 0.764 | 1.211 | 1.151 | 1.052 |
| qwen2.5-0.5b \| default | 260 | 57.935 | 0.269 | 0.214 | 0.055 | 0.704 | 0.474 | 0.203 | 0.738 | 2.584 | 1.543 | 1.675 |
| qwen2.5-1.5b \| concise | 260 | 33.673 | 0.069 | 0.063 | 0.005 | 0.912 | 0.819 | 0.061 | 0.936 | 1.323 | 1.086 | 1.217 |
| qwen2.5-1.5b \| default | 260 | 40.396 | 0.136 | 0.134 | 0.003 | 0.897 | 0.803 | 0.137 | 0.923 | 1.544 | 1.210 | 1.276 |
| smollm2-1.7b \| concise | 260 | 32.554 | 0.055 | 0.032 | 0.023 | 0.854 | 0.699 | 0.024 | 0.865 | 1.483 | 1.078 | 1.375 |
| smollm2-1.7b \| default | 260 | 40.842 | 0.036 | 0.027 | 0.010 | 0.835 | 0.687 | 0.015 | 0.868 | 1.765 | 1.052 | 1.677 |

Same, reader phi3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 260 | 32.262 | 0.120 | 0.058 | 0.062 | 0.735 | 0.473 | 0.036 | 0.747 | 1.269 | 1.179 | 1.076 |
| qwen2.5-0.5b \| default | 260 | 68.277 | 0.298 | 0.194 | 0.104 | 0.710 | 0.492 | 0.217 | 0.723 | 2.617 | 1.652 | 1.584 |
| qwen2.5-1.5b \| concise | 260 | 39.027 | 0.082 | 0.075 | 0.007 | 0.900 | 0.765 | 0.074 | 0.913 | 1.355 | 1.106 | 1.225 |
| qwen2.5-1.5b \| default | 260 | 46.931 | 0.149 | 0.132 | 0.017 | 0.885 | 0.758 | 0.151 | 0.902 | 1.569 | 1.257 | 1.248 |
| smollm2-1.7b \| concise | 260 | 37.546 | 0.060 | 0.025 | 0.035 | 0.836 | 0.677 | 0.024 | 0.851 | 1.505 | 1.089 | 1.383 |
| smollm2-1.7b \| default | 260 | 46.185 | 0.056 | 0.041 | 0.014 | 0.828 | 0.681 | 0.027 | 0.844 | 1.784 | 1.084 | 1.647 |

## T3. 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -30.623 [-35.694, -25.654] | -0.149 [-0.187, -0.107] | -0.145 [-0.179, -0.109] | -0.004 [-0.021, +0.014] | +0.017 [-0.024, +0.057] | ×0.51 [0.46, 0.56] |
| qwen2.5-1.5b | -6.723 [-9.324, -4.300] | -0.068 [-0.088, -0.047] | -0.070 [-0.090, -0.052] | +0.003 [-0.003, +0.009] | +0.015 [-0.011, +0.041] | ×0.86 [0.82, 0.90] |
| smollm2-1.7b | -8.288 [-10.058, -6.558] | +0.017 [+0.004, +0.032] | +0.007 [-0.004, +0.019] | +0.010 [+0.003, +0.018] | +0.019 [-0.012, +0.049] | ×0.83 [0.79, 0.86] |

Same, reader phi3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -36.015 [-41.898, -30.342] | -0.158 [-0.197, -0.116] | -0.126 [-0.161, -0.091] | -0.032 [-0.055, -0.009] | +0.024 [-0.014, +0.064] | ×0.53 [0.48, 0.57] |
| qwen2.5-1.5b | -7.904 [-10.931, -5.146] | -0.067 [-0.087, -0.047] | -0.056 [-0.073, -0.041] | -0.010 [-0.020, -0.002] | +0.015 [-0.008, +0.037] | ×0.86 [0.82, 0.90] |
| smollm2-1.7b | -8.638 [-10.546, -6.750] | +0.008 [-0.006, +0.023] | -0.007 [-0.020, +0.006] | +0.015 [+0.005, +0.025] | +0.009 [-0.021, +0.040] | ×0.84 [0.80, 0.87] |

## T4. Agreement between readers

1463 responses scored by both readers.

| main vs phi3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking |
|---|---|---|---|
| removable | 0.905 | 0.886 | 1.000 |
| filler | 0.827 | 0.799 | 0.867 |
| wrong | 0.698 | 0.718 |  |
| κ | 0.603 | 0.579 |  |
| facts | 0.769 | 0.730 | 1.000 |
| overhead | 0.958 | 0.951 | 1.000 |

Span level (4341 spans): deleted or kept — same decision 95.0%, Cohen's κ = 0.84; filler or not — 92.9%, κ = 0.71; wrong statement or not — 96.1%, κ = 0.61.

## T5. Removable share by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.093 | 0.163 | 0.000 | 0.000 | 0.346 |
| qwen2.5-0.5b \| default | 0.189 | 0.173 | 0.291 | 0.559 | 0.000 | 0.612 |
| qwen2.5-1.5b \| concise | 0.000 | 0.025 | 0.131 | 0.000 | 0.000 | 0.250 |
| qwen2.5-1.5b \| default | 0.000 | 0.078 | 0.183 | 0.015 | 0.000 | 0.659 |
| smollm2-1.7b \| concise | 0.000 | 0.020 | 0.061 | 0.000 | 0.000 | 0.413 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.040 | 0.027 | 0.036 | 0.300 |

## T5. Filler by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.060 | 0.058 | 0.000 | 0.000 | 0.346 |
| qwen2.5-0.5b \| default | 0.167 | 0.122 | 0.203 | 0.521 | 0.000 | 0.591 |
| qwen2.5-1.5b \| concise | 0.000 | 0.014 | 0.126 | 0.000 | 0.000 | 0.250 |
| qwen2.5-1.5b \| default | 0.000 | 0.076 | 0.177 | 0.015 | 0.000 | 0.659 |
| smollm2-1.7b \| concise | 0.000 | 0.006 | 0.018 | 0.000 | 0.000 | 0.348 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.014 | 0.027 | 0.036 | 0.282 |

## T5. Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.033 | 0.104 | 0.000 | 0.000 | 0.000 |
| qwen2.5-0.5b \| default | 0.021 | 0.051 | 0.088 | 0.038 | 0.000 | 0.020 |
| qwen2.5-1.5b \| concise | 0.000 | 0.011 | 0.005 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.002 | 0.006 | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.013 | 0.043 | 0.000 | 0.000 | 0.065 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 0.018 |

## T5. Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.872 | 0.719 | 0.744 | 0.409 | 0.600 | 0.625 |
| qwen2.5-0.5b \| default | 0.821 | 0.749 | 0.738 | 0.182 | 0.350 | 0.800 |
| qwen2.5-1.5b \| concise | 0.974 | 0.962 | 0.939 | 0.682 | 0.600 | 0.925 |
| qwen2.5-1.5b \| default | 0.897 | 0.958 | 0.940 | 0.500 | 0.650 | 0.950 |
| smollm2-1.7b \| concise | 0.974 | 0.909 | 0.896 | 0.500 | 0.650 | 0.625 |
| smollm2-1.7b \| default | 0.949 | 0.914 | 0.890 | 0.318 | 0.550 | 0.650 |

## T5. Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.125 | 0.062 | 0.000 | 0.050 | 0.200 | 0.250 |
| qwen2.5-0.5b \| default | 0.150 | 0.025 | 0.000 | 0.300 | 0.650 | 0.100 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 | 0.000 |
| qwen2.5-1.5b \| default | 0.050 | 0.000 | 0.000 | 0.100 | 0.200 | 0.000 |
| smollm2-1.7b \| concise | 0.025 | 0.000 | 0.000 | 0.150 | 0.350 | 0.150 |
| smollm2-1.7b \| default | 0.050 | 0.000 | 0.000 | 0.050 | 0.450 | 0.200 |

## T5. Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing fact, k=1 | missing fact, k=3 | all | n |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.100 | 0.000 | 0.050 | 40 |
| qwen2.5-0.5b \| default | 0.200 | 0.050 | 0.125 | 40 |
| qwen2.5-1.5b \| concise | 0.200 | 0.400 | 0.300 | 40 |
| qwen2.5-1.5b \| default | 0.350 | 0.400 | 0.375 | 40 |
| smollm2-1.7b \| concise | 0.150 | 0.100 | 0.125 | 40 |
| smollm2-1.7b \| default | 0.300 | 0.150 | 0.225 | 40 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.54; read as 'not stated' (p ≥ 0.5): 0.23 of the answers that say it, 0.19 of the others
- reader phi3.5-mini: AUC 0.93; read as 'not stated' (p ≥ 0.5): 0.12 of the answers that say it, 0.00 of the others

## T6. Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 57 | 0.755 | 0.820 | 0.309 | 0.055 | 0.442 | 0.772 |
| qwen2.5-0.5b \| default | 315 | 0.686 | 0.742 | 0.452 | 0.167 | 0.454 | 0.709 |
| qwen2.5-1.5b \| concise | 70 | 0.751 | 0.824 | 0.635 | 0.000 |  | 0.805 |
| qwen2.5-1.5b \| default | 176 | 0.800 | 0.768 | 0.697 | 0.012 | 1.000 | 0.934 |
| smollm2-1.7b \| concise | 24 | 0.895 | 0.845 | 0.761 | 0.059 | 0.853 | 0.897 |
| smollm2-1.7b \| default | 21 | 0.919 | 0.836 | 0.610 | 0.150 | 0.757 | 0.956 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.17 | 1.84 | 0.35 | 1.25 |
| qwen2.5-0.5b \| default | 0.73 | 1.23 | 0.57 | 1.31 |
| qwen2.5-1.5b \| concise | 0.40 | 1.62 | 0.37 | 1.25 |
| qwen2.5-1.5b \| default | 0.19 | 1.79 | 0.21 | 1.31 |
| smollm2-1.7b \| concise | 0.00 | 1.24 | 0.10 | 1.87 |
| smollm2-1.7b \| default | 0.00 | 1.31 | 0.00 | 2.70 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.086 [-0.154, -0.018] | +5.17 [+4.28, +6.06] | +0.50 [+0.30, +0.73] |
| qwen2.5-0.5b \| default | -0.063 [-0.205, +0.066] | +4.43 [+3.20, +5.55] | +0.72 [-1.26, +2.68] |
| qwen2.5-1.5b \| concise | -0.010 [-0.057, +0.039] | +5.75 [+5.18, +6.33] | +1.72 [+1.42, +2.02] |
| qwen2.5-1.5b \| default | -0.062 [-0.143, +0.012] | +5.81 [+5.33, +6.27] | +2.12 [+1.85, +2.44] |
| smollm2-1.7b \| concise | -0.032 [-0.083, +0.009] | +5.79 [+5.09, +6.71] | +0.23 [+0.07, +0.40] |
| smollm2-1.7b \| default | -0.039 [-0.095, +0.009] | +10.32 [+9.51, +11.28] | +0.12 [+0.00, +0.25] |

## T7. Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.105 | 0.112 |
| main — qwen2.5-0.5b \| default | 0.269 | 0.297 |
| main — qwen2.5-1.5b \| concise | 0.069 | 0.082 |
| main — qwen2.5-1.5b \| default | 0.136 | 0.148 |
| main — smollm2-1.7b \| concise | 0.055 | 0.058 |
| main — smollm2-1.7b \| default | 0.036 | 0.045 |
| phi3.5-mini — qwen2.5-0.5b \| concise | 0.120 | 0.129 |
| phi3.5-mini — qwen2.5-0.5b \| default | 0.298 | 0.310 |
| phi3.5-mini — qwen2.5-1.5b \| concise | 0.082 | 0.084 |
| phi3.5-mini — qwen2.5-1.5b \| default | 0.149 | 0.158 |
| phi3.5-mini — smollm2-1.7b \| concise | 0.060 | 0.066 |
| phi3.5-mini — smollm2-1.7b \| default | 0.056 | 0.058 |

## Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-0.5b, c12-0016] request: What caused the delay of the shipment that departed from Port Ultavi? — span: «2. **Delay Information**:»
- **uninformative** [qwen2.5-0.5b, c12-0016] request: What caused the delay of the shipment that departed from Port Ultavi? — span: «- The delay for shipment JD-2862 was due to a berth reassignment.»
- **uninformative** [qwen2.5-1.5b, c14-0018] request: What is the home port of the carrier that handled shipment ZB-6725? Explain how you found it. — span: «Step 2: Finding the home port of Zanob Cargo»
- **uninformative** [qwen2.5-0.5b, c03-0008] request: For shipment KZ-8470, what were the number of containers and the inspector? — span: «Based on the information provided in the report:»
- **duplicate** [qwen2.5-0.5b, c03-0009] request: For shipment SC-4779, what were the cause of the delay and the carrier? — span: «2. **Carrier**: The carrier responsible for this shipment is Marzanyor Shipping.»
- **duplicate** [qwen2.5-0.5b, c14-0011] request: What is the home port of the carrier that handled shipment RF-3147? Explain how you found it. — span: «The home port of the carrier that handled shipment RF-3147 is Relkabex Harbor.»
- **duplicate** [qwen2.5-0.5b, c04-0000] request: For shipment CM-5244, what were the number of containers, the destination port and the departure dat… — span: «- Destination port: Ulvivan Harbor»
- **duplicate** [qwen2.5-0.5b, c14-0005] request: What is the home port of the carrier that handled shipment BH-6911? Explain how you found it. — span: «3. **Conclusion**: Since Fenulyor Cargo is the carrier responsible for handling shipment BH-6911, and we have established that it originates from Eskmar Harbor, the home port of th»
- **misleading** [qwen2.5-0.5b, c09-0004] request: For shipment HC-2729, what were the cargo, the departure date, the number of containers, the inspect… — span: «**Origin Port:** Pelquiros Docks»
- **misleading** [qwen2.5-0.5b, c09-0018] request: For shipment VN-4782, what were the cargo, the inspector, the departure date, the destination port, … — span: «Shipment VN-4782 was handled by Zannalbex Shipping.»
- **misleading** [qwen2.5-0.5b, c08-0019] request: For shipment KJ-3299, what were the carrier, the cargo, the number of containers, the origin port, t… — span: «Inspector: Eskmaresk Lines»
- **misleading** [qwen2.5-0.5b, c12-0007] request: What caused the delay of the shipment that departed from Port Ulpel? — span: «- The delay for this shipment was also due to a berth reassignment.»
