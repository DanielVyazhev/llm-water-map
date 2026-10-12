# Filler in LLM answers on real texts — real_t4_n15 (package v1.0.0)

Reader: Qwen/Qwen2.5-3B-Instruct; other readers: second reader = microsoft/Phi-3.5-mini-instruct; models: qwen2.5-0.5b = Qwen/Qwen2.5-0.5B-Instruct, qwen2.5-1.5b = Qwen/Qwen2.5-1.5B-Instruct, smollm2-1.7b = HuggingFaceTB/SmolLM2-1.7B-Instruct.

## R0. Items after the leakage filter

An item is dropped when the reader answers one of its facts from an empty message with p ≥ 0.05 (it knows the fact), or reads a fact from the gold answer with p < 0.5.

| cell | pool | checked | reader knew a fact | gold unreadable | kept | kept from before |
|---|---|---|---|---|---|---|
| SQuAD 2.0: k=1 | 60 | 0 | 0 | 0 | 15 | 15 |
| SQuAD 2.0: k=2 | 60 | 0 | 0 | 0 | 15 | 15 |
| SQuAD 2.0: k=3 | 60 | 0 | 0 | 0 | 15 | 15 |
| SQuAD 2.0: k=4 | 60 | 0 | 0 | 0 | 15 | 15 |
| SQuAD 2.0: missing (k=1) | 60 | 0 | 0 | 0 | 15 | 15 |
| SQuAD 2.0: missing (k=3) | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 2-hop | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 2-hop+explain | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 3-hop | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 3-hop+explain | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 4-hop | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: 4-hop+explain | 60 | 0 | 0 | 0 | 15 | 15 |
| MuSiQue: missing (2-hop) | 60 | 0 | 0 | 0 | 15 | 15 |
| CUAD (contracts): k=1 | 60 | 0 | 0 | 0 | 15 | 15 |
| CUAD (contracts): k=2 | 60 | 0 | 0 | 0 | 15 | 15 |
| CUAD (contracts): k=3 | 60 | 0 | 0 | 0 | 15 | 15 |
| CUAD (contracts): k=1, long | 60 | 0 | 0 | 0 | 15 | 15 |
| CUAD (contracts): missing law (k=2) | 60 | 0 | 0 | 0 | 15 | 15 |
| Poddar et al. (passages): NarrativeQA | 59 | 0 | 0 | 0 | 15 | 15 |
| Poddar et al. (passages): TweetQA | 60 | 0 | 0 | 0 | 15 | 15 |
| Poddar et al. (passages): MS MARCO | 17 | 5 | 0 | 5 | 12 | 12 |

## R1. Reader check on the real items

| reader check (real items) | AUC present vs absent | gold unreadable | wrong-statement detection | nats per fact |
|---|---|---|---|---|
| main | 0.997 | 0.000 | 0.931 | 13.815 |
| phi-3.5-mini | 0.989 | 0.000 | 0.944 | 13.815 |

## R2. Filler map on simulated responses with known filler

Synthetic reports; each response = the requested facts + planted filler of known type (none / 2 / 5 fragments, or 2 with one fact left out). Recall counts planted filler fragments (95% CI, bootstrap over items); a repeat is resolved when no fact is stated twice after deletion; W error = measured − true filler share. 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader, tolerance | responses (with filler) | no fact lost | precision | recall, planted fragments | recall per response with filler | repeats resolved | MAE of W | mean error of W |
|---|---|---|---|---|---|---|---|---|
| main, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 0.953 [0.903, 0.989] | 0.953 | 0.846 | 0.036 | -0.036 |
| main, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 0.984 [0.965, 1.000] | 0.983 | 0.846 | 0.022 | -0.022 |
| main, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.926 [0.878, 0.964] | 0.926 | 0.846 | 0.041 | -0.041 |
| phi-3.5-mini, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.923 | 0.013 | -0.013 |
| phi-3.5-mini, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 0.962 | 0.012 | -0.012 |
| phi-3.5-mini, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.958 [0.921, 0.985] | 0.941 | 0.885 | 0.026 | -0.026 |

Deleted / planted by type (τ = 0.5): main: closing 21/23, distractor 9/9, elaboration 20/21, enumeration 20/21, hedge 29/31, preamble 21/22, restatement 24/25, unrequested 37/38; phi-3.5-mini: closing 23/23, distractor 9/9, elaboration 21/21, enumeration 21/21, hedge 31/31, preamble 22/22, restatement 25/25, unrequested 38/38.

Planted filler left in (τ = 0.5):
- main: of 9 planted filler fragments left in, 56% cost more than the tolerance to delete even alone (the reader uses them); 44% could go alone but not together with the fragments deleted before them.

## R3. Filler map on real-text responses with planted filler

Responses built from each item's own document: the sentence that states each requested fact, plus planted filler — formulaic (preamble, restatement, hedge, generic remark, closing), real document sentences that answer nothing asked (elaboration: same paragraph as a fact; unrequested: another paragraph) or a summary repeating the facts. Conditions: none / 2 / 5 filler fragments, 2 with one fact left out, and 1 filler fragment + 1 wrong statement (another item's answer given as the answer). Truth is known per fragment; recall counts planted filler fragments (95% CI, bootstrap over items). Items with a missing fact are left out. 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader — suite | responses (with filler) | no fact lost | precision | recall, planted filler [95% CI] | repeats resolved | wrong statements deleted | … and labelled wrong | MAE of removable share |
|---|---|---|---|---|---|---|---|---|
| main — all suites | 268 (195) | 0.974 | 0.987 | 0.782 [0.720, 0.844] | 0.688 | 0.935 | 0.652 | 0.154 |
| main — CUAD (contracts) | 70 (49) | 1.000 | 1.000 | 0.842 [0.750, 0.928] | 0.682 | 0.857 | 0.857 | 0.086 |
| main — MuSiQue | 91 (68) | 0.945 | 0.966 | 0.639 [0.526, 0.755] | 0.500 | 1.000 | 0.643 | 0.277 |
| main — Poddar et al. (passages) | 40 (29) | 1.000 | 1.000 | 0.987 [0.958, 1.000] | 0.923 | 1.000 | 0.667 | 0.026 |
| main — SQuAD 2.0 | 67 (49) | 0.970 | 0.985 | 0.795 [0.656, 0.920] | 0.778 | 0.917 | 0.417 | 0.133 |
| main — all suites, prefix ablation | 268 (195) | 0.974 | 0.986 | 0.734 [0.669, 0.798] | 0.649 |  |  | 0.159 |
| phi-3.5-mini — all suites | 280 (206) | 0.954 | 0.977 | 0.832 [0.786, 0.876] | 0.753 | 0.725 | 0.569 | 0.119 |
| phi-3.5-mini — CUAD (contracts) | 69 (48) | 1.000 | 1.000 | 0.882 [0.828, 0.938] | 0.818 | 0.923 | 0.923 | 0.083 |
| phi-3.5-mini — MuSiQue | 98 (74) | 0.939 | 0.967 | 0.797 [0.718, 0.877] | 0.640 | 0.533 | 0.333 | 0.167 |
| phi-3.5-mini — Poddar et al. (passages) | 40 (30) | 0.975 | 0.988 | 0.904 [0.821, 0.985] | 0.917 | 0.875 | 0.625 | 0.053 |
| phi-3.5-mini — SQuAD 2.0 | 73 (54) | 0.918 | 0.960 | 0.795 [0.692, 0.895] | 0.722 | 0.667 | 0.467 | 0.126 |
| phi-3.5-mini — all suites, prefix ablation | 283 (208) | 0.954 | 0.976 | 0.793 [0.743, 0.843] | 0.756 |  |  | 0.129 |

Deleted / planted by type (τ = 0.5 nats): main: closing 58/70, elaboration 63/73, generic 50/64, hedge 57/73, preamble 55/77, restatement 45/62, unrequested 46/59; phi-3.5-mini: closing 63/74, elaboration 64/80, generic 53/68, hedge 66/78, preamble 65/76, restatement 57/64, unrequested 49/61.

- main: of 104 planted filler fragments left in, 91% cost more than the tolerance to delete even alone (the reader uses them); 9% could go alone but not together with the fragments deleted before them.
- main, prefix ablation: planted filler deleted by suite — CUAD (contracts) 0.742; MuSiQue 0.620; Poddar et al. (passages) 0.947; SQuAD 2.0 0.752.
- main, responses whose facts the reader all reads (p ≥ 0.5): 150; no fact lost 1.000, precision 1.000, planted filler deleted 0.933.
- main: planted filler labelled as a wrong statement in 0.040 of 555 fragments; κ 0.88 with every fact, 0.55 with one fact left out; at τ = 2 nats no fact lost in 0.933, planted filler deleted 0.910.
- phi-3.5-mini: of 84 planted filler fragments left in, 83% cost more than the tolerance to delete even alone (the reader uses them); 17% could go alone but not together with the fragments deleted before them.
- phi-3.5-mini, prefix ablation: planted filler deleted by suite — CUAD (contracts) 0.840; MuSiQue 0.780; Poddar et al. (passages) 0.810; SQuAD 2.0 0.756.
- phi-3.5-mini, responses whose facts the reader all reads (p ≥ 0.5): 152; no fact lost 1.000, precision 1.000, planted filler deleted 0.979.
- phi-3.5-mini: planted filler labelled as a wrong statement in 0.066 of 578 fragments; κ 0.88 with every fact, 0.55 with one fact left out; at τ = 2 nats no fact lost in 0.933, planted filler deleted 0.964.

## R4. Filler map vs human span labels (Poddar et al., 2025)

Responses of 10 LLMs (GPT-3.5/4o-mini, Llama, Gemma, Mistral, Phi) to the passage questions of Poddar et al.'s data (NarrativeQA, TweetQA, MS MARCO; reference answer of at most 6 words), with human span labels. A fragment is 'not minimal' when most of its characters carry a label other than MINANS; precision and recall of deletion are measured against that (unlabelled fragments left out). The fact is the response's own answer: the reference answer, or what the reader reads from the fragments marked MINANS (annotators label minimal vs extra text whether or not the answer is right); responses with neither are not scored. 'reference answer only' = responses that state the reference answer; 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it.

| reader | responses scored | precision | recall | F1 | Cohen's κ | F1 by characters | minimal answer kept | Spearman, share per response | deleted share (mean) | human share (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 125 | 0.939 | 0.805 | 0.867 | 0.618 | 0.891 | 0.930 | 0.697 | 0.340 | 0.454 |
| main, reference answer only | 84 | 0.948 | 0.743 | 0.833 | 0.578 | 0.854 | 0.921 | 0.700 | 0.274 | 0.410 |
| main, prefix ablation | 125 | 0.938 | 0.792 | 0.859 | 0.601 | 0.883 | 0.930 | 0.693 | 0.336 | 0.454 |
| phi-3.5-mini | 131 | 0.951 | 0.831 | 0.887 | 0.647 | 0.880 | 0.949 | 0.710 | 0.345 | 0.459 |
| phi-3.5-mini, reference answer only | 95 | 0.976 | 0.805 | 0.882 | 0.678 | 0.869 | 0.965 | 0.718 | 0.294 | 0.425 |
| phi-3.5-mini, prefix ablation | 131 | 0.949 | 0.802 | 0.870 | 0.608 | 0.867 | 0.949 | 0.705 | 0.339 | 0.459 |

- main: 125 of 189 labelled responses scored (84 by the reference, 41 by the own answer).
- main: of 60 kept fragments the annotators marked as not minimal, 82% cost more than the tolerance to delete alone, 18% were blocked in the queue.
- phi-3.5-mini: 131 of 189 labelled responses scored (95 by the reference, 35 by the own answer, 1 by the none).
- phi-3.5-mini: of 59 kept fragments the annotators marked as not minimal, 97% cost more than the tolerance to delete alone, 3% were blocked in the queue.

| human label: share deleted | main | phi-3.5-mini |
|---|---|---|
| minimal answer (MINANS) | 0.12 (n=128) | 0.12 (n=130) |
| additional information (ADDINFO) | 0.86 (n=77) | 0.85 (n=84) |
| explanation (EXPLAIN) | 0.85 (n=120) | 0.82 (n=142) |
| conversational (CONVERS) | 0.83 (n=12) | 0.92 (n=12) |
| redundant (REDINFO) | 1.00 (n=2) | 0.33 (n=3) |
| irrelevant (IRREL) | 0.70 (n=97) | 0.83 (n=108) |
| unlabelled (NONE) | 1.00 (n=2) | 1.00 (n=2) |

| subset | main | phi-3.5-mini |
|---|---|---|
| MS MARCO | F1 0.92, κ 0.57 (n=17) | F1 0.93, κ 0.66 (n=15) |
| NarrativeQA | F1 0.86, κ 0.61 (n=50) | F1 0.90, κ 0.66 (n=55) |
| TweetQA | F1 0.82, κ 0.61 (n=58) | F1 0.83, κ 0.60 (n=61) |

Ranking of the 10 LLMs by mean deleted share vs by mean human non-minimal share: main: Kendall τ = 0.69 (p = 0.005, 10 LLMs); phi-3.5-mini: Kendall τ = 0.83 (p = 0.001, 9 LLMs).

## R5. Baselines on the same responses

Every method on the same responses: the filler map of each reader, the LLM judge and Selective Context when they were run (drybench baselines), and three cheap metrics. Fragment level, removable (filler or wrong statement) vs needed: F1, Cohen's κ, and the AUC of the method's score (the judge's p(filler) + p(wrong); Selective Context: lower self-information = more removable). Response level, removable share by characters: Spearman with the true share; 'shorter but more removable' = pairs whose true share differs by ≥ 0.2 where the more removable response is the shorter one, share the method ranks higher (ties apart); partial Spearman with the true share given length [95% CI, bootstrap over items]. Selective Context deletes, least informative fragment first, as many characters as the first reader's filler map (equal budget); the judge's share is its expected removable share. Planted responses: the true share is the planted filler and wrong statements (a summary that repeats the facts is planted filler); a fact stated twice may go either way, so such fragments are left out of the fragment-level scores. First table: the responses every method scores; second: also the answers from which a reader reads none of the requested facts — the filler map's verdict on such an answer is that it delivers nothing, so all of it counts as removable (as in the results tables: a failed answer); answers whose gold the reader cannot read stay out.

### Planted filler in real texts (the R3 responses)

259 of 302 responses scored by every method (filler map (reader main) 268; filler map (reader phi-3.5-mini) 280); 844 fragments with a truth label; 10295 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.861 | 0.691 | 0.859 | 0.766 | 0.833 | 0.016 | +0.77 [+0.69, +0.83] |
| filler map (reader phi-3.5-mini) | 0.869 | 0.693 | 0.854 | 0.696 | 0.797 | 0.004 | +0.70 [+0.61, +0.78] |
| length (characters) |  |  |  | 0.039 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.138 | 0.119 | 0.002 | +0.36 [+0.27, +0.44] |
| stop-word share |  |  |  | 0.385 | 0.759 | 0.002 | +0.40 [+0.29, +0.51] |

With the answers that deliver nothing to a reader (298 responses; all of such an answer counts as removable):

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.844 | 0.628 | 0.819 | 0.673 | 0.797 | 0.024 | +0.67 [+0.58, +0.76] |
| filler map (reader phi-3.5-mini) | 0.854 | 0.647 | 0.828 | 0.651 | 0.794 | 0.008 | +0.65 [+0.55, +0.74] |
| length (characters) |  |  |  | -0.006 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.099 | 0.117 | 0.002 | +0.37 [+0.28, +0.47] |
| stop-word share |  |  |  | 0.415 | 0.775 | 0.001 | +0.43 [+0.30, +0.53] |

### Human span labels (the R4 responses, Poddar et al.)

111 of 189 responses scored by every method (filler map (reader main) 125; filler map (reader phi-3.5-mini) 131); 396 fragments with a truth label; 287 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.871 | 0.617 | 0.838 | 0.704 | 0.341 | 0.223 | +0.25 [+0.00, +0.53] |
| filler map (reader phi-3.5-mini) | 0.889 | 0.664 | 0.859 | 0.733 | 0.383 | 0.216 | +0.32 [+0.06, +0.60] |
| length (characters) |  |  |  | 0.747 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.755 | 0.094 | 0.003 | +0.16 [+0.03, +0.29] |
| stop-word share |  |  |  | 0.154 | 0.488 | 0.003 | +0.04 [-0.14, +0.21] |

With the answers that deliver nothing to a reader (168 responses; all of such an answer counts as removable):

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.901 | 0.602 | 0.819 | 0.713 | 0.626 | 0.127 | +0.53 [+0.33, +0.72] |
| filler map (reader phi-3.5-mini) | 0.899 | 0.590 | 0.810 | 0.664 | 0.560 | 0.166 | +0.47 [+0.29, +0.66] |
| length (characters) |  |  |  | 0.597 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.600 | 0.044 | 0.008 | +0.07 [-0.06, +0.18] |
| stop-word share |  |  |  | 0.065 | 0.404 | 0.018 | -0.00 [-0.16, +0.17] |

Responses with a fragment labelled minimal answer (149 of 189; in the others no fragment is mostly minimal answer — the answer, if any, sits inside a fragment that is mostly something else, and the filler map does not delete a fragment that carries the answer):

102 of 149 responses scored by every method (filler map (reader main) 115; filler map (reader phi-3.5-mini) 118); 382 fragments with a truth label; 117 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | responses: Spearman | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.886 | 0.660 | 0.851 | 0.912 | 0.812 | 0.017 | +0.71 [+0.48, +0.88] |
| filler map (reader phi-3.5-mini) | 0.907 | 0.714 | 0.875 | 0.953 | 0.923 | 0.000 | +0.85 [+0.73, +0.93] |
| length (characters) |  |  |  | 0.853 | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.854 | 0.103 | 0.009 | +0.14 [+0.02, +0.25] |
| stop-word share |  |  |  | 0.143 | 0.547 | 0.000 | -0.01 [-0.21, +0.19] |

## R6. Across suites

Removable share (reader main):

| model \| prompt | SQuAD 2.0 | MuSiQue | CUAD (contracts) | Poddar et al. (passages) |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.155 | 0.310 | 0.101 | 0.028 |
| qwen2.5-0.5b \| default | 0.140 | 0.407 | 0.076 | 0.100 |
| qwen2.5-1.5b \| concise | 0.175 | 0.367 | 0.024 | 0.043 |
| qwen2.5-1.5b \| default | 0.188 | 0.457 | 0.020 | 0.114 |
| smollm2-1.7b \| concise | 0.174 | 0.233 | 0.066 | 0.036 |
| smollm2-1.7b \| default | 0.153 | 0.354 | 0.082 | 0.138 |

Facts delivered (reader main):

| model \| prompt | SQuAD 2.0 | MuSiQue | CUAD (contracts) | Poddar et al. (passages) |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.596 | 0.158 | 0.686 | 0.405 |
| qwen2.5-0.5b \| default | 0.599 | 0.156 | 0.686 | 0.310 |
| qwen2.5-1.5b \| concise | 0.685 | 0.267 | 0.769 | 0.405 |
| qwen2.5-1.5b \| default | 0.669 | 0.286 | 0.733 | 0.357 |
| smollm2-1.7b \| concise | 0.661 | 0.300 | 0.667 | 0.548 |
| smollm2-1.7b \| default | 0.590 | 0.325 | 0.683 | 0.548 |

Kendall τ between suites of the model ranking by removable share (default prompt; 3 models): SQuAD 2.0 vs MuSiQue +0.33; SQuAD 2.0 vs CUAD (contracts) -0.33; SQuAD 2.0 vs Poddar et al. (passages) +0.33; MuSiQue vs CUAD (contracts) -1.00; MuSiQue vs Poddar et al. (passages) -0.33; CUAD (contracts) vs Poddar et al. (passages) +0.33.

## SQuAD 2.0

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 44.933 | 0.155 | 0.096 | 0.060 | 0.596 | 0.350 | 0.000 | 0.661 | 2.907 | 1.254 | 2.318 |
| qwen2.5-0.5b \| default | 60 | 52.133 | 0.140 | 0.106 | 0.034 | 0.599 | 0.350 | 0.000 | 0.655 | 3.064 | 1.224 | 2.503 |
| qwen2.5-1.5b \| concise | 60 | 39.700 | 0.175 | 0.110 | 0.065 | 0.685 | 0.400 | 0.010 | 0.750 | 3.071 | 1.325 | 2.319 |
| qwen2.5-1.5b \| default | 60 | 49.733 | 0.188 | 0.123 | 0.065 | 0.669 | 0.383 | 0.008 | 0.753 | 3.839 | 1.328 | 2.892 |
| smollm2-1.7b \| concise | 60 | 49.167 | 0.174 | 0.082 | 0.093 | 0.661 | 0.383 | 0.010 | 0.728 | 3.850 | 1.300 | 2.961 |
| smollm2-1.7b \| default | 60 | 54.067 | 0.153 | 0.065 | 0.088 | 0.590 | 0.333 | 0.010 | 0.680 | 4.753 | 1.235 | 3.848 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 52.600 | 0.221 | 0.076 | 0.146 | 0.571 | 0.317 | 0.000 | 0.665 | 3.022 | 1.419 | 2.130 |
| qwen2.5-0.5b \| default | 60 | 60.300 | 0.220 | 0.081 | 0.139 | 0.556 | 0.300 | 0.027 | 0.656 | 3.587 | 1.391 | 2.578 |
| qwen2.5-1.5b \| concise | 60 | 46.783 | 0.184 | 0.036 | 0.148 | 0.675 | 0.417 | 0.023 | 0.756 | 2.897 | 1.320 | 2.194 |
| qwen2.5-1.5b \| default | 60 | 57.600 | 0.165 | 0.042 | 0.123 | 0.651 | 0.400 | 0.014 | 0.766 | 3.646 | 1.273 | 2.864 |
| smollm2-1.7b \| concise | 60 | 56.967 | 0.199 | 0.037 | 0.162 | 0.644 | 0.400 | 0.021 | 0.733 | 4.146 | 1.351 | 3.069 |
| smollm2-1.7b \| default | 60 | 62.483 | 0.215 | 0.059 | 0.156 | 0.613 | 0.383 | 0.026 | 0.691 | 4.393 | 1.364 | 3.220 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -7.200 [-13.817, -0.883] | +0.018 [-0.028, +0.066] | -0.002 [-0.048, +0.045] | +0.019 [-0.019, +0.058] | -0.003 [-0.074, +0.072] | ×0.92 [0.82, 1.04] |
| qwen2.5-1.5b | -10.033 [-15.484, -5.150] | -0.024 [-0.073, +0.027] | -0.013 [-0.063, +0.039] | -0.011 [-0.033, +0.007] | +0.015 [-0.036, +0.062] | ×0.77 [0.67, 0.89] |
| smollm2-1.7b | -4.900 [-9.018, -1.183] | +0.011 [-0.031, +0.056] | +0.012 [-0.017, +0.050] | -0.001 [-0.034, +0.029] | +0.071 [+0.013, +0.136] | ×0.85 [0.75, 0.95] |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -7.700 [-15.367, -0.400] | +0.000 [-0.039, +0.040] | -0.010 [-0.050, +0.032] | +0.010 [-0.029, +0.053] | +0.015 [-0.051, +0.094] | ×0.88 [0.75, 1.00] |
| qwen2.5-1.5b | -10.817 [-16.702, -5.483] | +0.016 [-0.022, +0.060] | -0.007 [-0.032, +0.021] | +0.023 [-0.016, +0.069] | +0.024 [-0.029, +0.075] | ×0.77 [0.66, 0.88] |
| smollm2-1.7b | -5.517 [-10.084, -1.316] | -0.010 [-0.065, +0.047] | -0.020 [-0.039, -0.003] | +0.010 [-0.041, +0.065] | +0.032 [-0.008, +0.076] | ×0.89 [0.81, 0.97] |

### Removable share by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.125 | 0.297 | 0.165 |
| qwen2.5-0.5b \| default | 0.000 | 0.160 | 0.208 | 0.168 |
| qwen2.5-1.5b \| concise | 0.000 | 0.269 | 0.253 | 0.149 |
| qwen2.5-1.5b \| default | 0.000 | 0.262 | 0.244 | 0.231 |
| smollm2-1.7b \| concise | 0.000 | 0.230 | 0.278 | 0.173 |
| smollm2-1.7b \| default | 0.017 | 0.153 | 0.234 | 0.195 |

### Filler by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.080 | 0.161 | 0.129 |
| qwen2.5-0.5b \| default | 0.000 | 0.122 | 0.144 | 0.142 |
| qwen2.5-1.5b \| concise | 0.000 | 0.145 | 0.212 | 0.065 |
| qwen2.5-1.5b \| default | 0.000 | 0.177 | 0.158 | 0.149 |
| smollm2-1.7b \| concise | 0.000 | 0.091 | 0.133 | 0.093 |
| smollm2-1.7b \| default | 0.000 | 0.055 | 0.072 | 0.124 |

### Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.045 | 0.136 | 0.037 |
| qwen2.5-0.5b \| default | 0.000 | 0.037 | 0.064 | 0.025 |
| qwen2.5-1.5b \| concise | 0.000 | 0.124 | 0.041 | 0.084 |
| qwen2.5-1.5b \| default | 0.000 | 0.085 | 0.087 | 0.082 |
| smollm2-1.7b \| concise | 0.000 | 0.140 | 0.145 | 0.080 |
| smollm2-1.7b \| default | 0.017 | 0.099 | 0.161 | 0.071 |

### Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.733 | 0.500 | 0.600 | 0.550 |
| qwen2.5-0.5b \| default | 0.667 | 0.567 | 0.578 | 0.583 |
| qwen2.5-1.5b \| concise | 0.733 | 0.600 | 0.689 | 0.717 |
| qwen2.5-1.5b \| default | 0.800 | 0.567 | 0.644 | 0.667 |
| smollm2-1.7b \| concise | 0.800 | 0.533 | 0.644 | 0.667 |
| smollm2-1.7b \| default | 0.600 | 0.533 | 0.578 | 0.650 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.200 | 0.200 | 0.000 | 0.267 |
| qwen2.5-0.5b \| default | 0.267 | 0.200 | 0.067 | 0.200 |
| qwen2.5-1.5b \| concise | 0.200 | 0.067 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.133 | 0.133 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.200 | 0.200 | 0.067 | 0.067 |
| smollm2-1.7b \| default | 0.200 | 0.267 | 0.133 | 0.067 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing (k=1) | missing (k=3) | all | n |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.000 | 30 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.000 | 30 |
| qwen2.5-1.5b \| concise | 0.000 | 0.067 | 0.033 | 30 |
| qwen2.5-1.5b \| default | 0.067 | 0.067 | 0.067 | 30 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.000 | 30 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.000 | 30 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.70; read as 'not stated' (p ≥ 0.5): 0.67 of the answers that say it, 0.15 of the others
- reader phi-3.5-mini: AUC 0.79; read as 'not stated' (p ≥ 0.5): 0.00 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 15 | 0.626 | 0.726 | 0.596 | 0.571 | 0.513 | 0.761 |
| qwen2.5-0.5b \| default | 17 | 0.453 | 0.625 | 0.632 | 0.688 | 0.421 | 0.586 |
| qwen2.5-1.5b \| concise | 17 | 0.670 | 0.464 | 0.621 | 0.133 | 0.649 | 0.689 |
| qwen2.5-1.5b \| default | 23 | 0.635 | 0.510 | 0.496 | 0.238 | 0.495 | 0.634 |
| smollm2-1.7b \| concise | 13 | 0.753 | 0.533 | 0.513 | 0.167 | 0.638 | 0.742 |
| smollm2-1.7b \| default | 12 | 0.730 | 0.617 | 0.413 | 0.333 | 0.746 | 0.696 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.50 | 0.71 | 1.07 | 1.67 |
| qwen2.5-0.5b \| default | 0.43 | 1.02 | 1.40 | 0.73 |
| qwen2.5-1.5b \| concise | 0.28 | 1.55 | 0.71 | 1.56 |
| qwen2.5-1.5b \| default | 0.43 | 0.48 | 1.49 | 1.59 |
| smollm2-1.7b \| concise | 0.21 | 0.78 | 0.52 | 2.12 |
| smollm2-1.7b \| default | 0.25 | 1.27 | 0.54 | 1.94 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | +0.060 [-0.065, +0.197] | +15.45 [+11.33, +19.80] | +3.70 [+1.07, +6.82] |
| qwen2.5-0.5b \| default | +0.099 [-0.055, +0.261] | +16.93 [+11.34, +22.68] | +3.87 [+0.76, +7.11] |
| qwen2.5-1.5b \| concise | +0.111 [-0.097, +0.324] | +12.29 [+8.39, +15.67] | +1.76 [-0.04, +3.92] |
| qwen2.5-1.5b \| default | +0.257 [+0.025, +0.443] | +14.28 [+10.73, +17.48] | +3.53 [+1.40, +6.00] |
| smollm2-1.7b \| concise | +0.108 [-0.042, +0.292] | +13.55 [+9.69, +16.95] | +2.59 [+0.49, +4.98] |
| smollm2-1.7b \| default | +0.086 [-0.022, +0.248] | +12.23 [+7.32, +16.27] | +3.08 [+1.24, +5.30] |

### Agreement between readers

307 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking |
|---|---|---|---|
| removable | 0.739 | 0.745 | -0.733 |
| filler | 0.567 | 0.562 | -0.067 |
| wrong | 0.555 | 0.554 |  |
| κ | 0.859 | 0.856 |  |
| facts | 0.774 | 0.773 | 0.600 |
| overhead | 0.916 | 0.912 | 0.733 |

Span level (913 spans): deleted or kept — same decision 88.3%, Cohen's κ = 0.63; filler or not — 90.8%, κ = 0.43; wrong statement or not — 90.3%, κ = 0.49.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.155 | 0.171 |
| main — qwen2.5-0.5b \| default | 0.140 | 0.175 |
| main — qwen2.5-1.5b \| concise | 0.175 | 0.193 |
| main — qwen2.5-1.5b \| default | 0.188 | 0.188 |
| main — smollm2-1.7b \| concise | 0.174 | 0.174 |
| main — smollm2-1.7b \| default | 0.153 | 0.182 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.221 | 0.229 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.220 | 0.257 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.184 | 0.214 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.165 | 0.204 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.199 | 0.249 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.215 | 0.240 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-1.5b, sq01-0008] request: Answer these questions about the text: 1. Focus on what is to ameliorate the many problems that aris… — span: «1. The focus on cooperation is to ameliorate the many problems that arise from the often highly competitive and adversarial practices within the construction industry.»
- **uninformative** [qwen2.5-1.5b, sq03-0009] request: Answer these questions about the text: 1. How many mechanisms does a typical steam engine have to ke… — span: «3. An illegal breakage of a seal would need to occur to allow an operator to tamper with the adjustable spring-loaded valve.»
- **uninformative** [qwen2.5-1.5b, sq02-0017] request: Answer these questions about the text: 1. Other than many sunny days, what characteristic is typical… — span: «1. The characteristic that is typically not mentioned but often associated with the climate in southern California is frequent rainfall.»
- **uninformative** [qwen2.5-0.5b, sq03-0001] request: Answer these questions about the text: 1. What video game series did Alex Seropian make? 2. What Ame… — span: «4. The University of Chicago is also an American university.»
- **duplicate** [smollm2-1.7b, sq03-0010] request: Answer these questions about the text: 1. How much of Afghanistan did the Taliban take over? 2. When… — span: «4. The civil war between political and tribal warlords was a vicious and destructive conflict.»
- **duplicate** [qwen2.5-0.5b, sq02-0014] request: Answer these questions about the text: 1. How many people lived in Fresno in 2000, according to the … — span: «2. The population density in Fresno in 2000 was 4,097.9 people per square mile (1,582.2/km²).»
- **duplicate** [smollm2-1.7b, sq03-0000] request: Answer these questions about the text: 1. What did Guo Shoujing do for calendars? 2. What was the En… — span: «2. The English name of Gou's calendar is the Shoushi Li (授時暦).»
- **duplicate** [qwen2.5-0.5b, sq02-0010] request: Answer these questions about the text: 1. Where did scientists find their Y. pestis sample? 2. When … — span: «1. Scientists found their Y. pestis sample in the tooth sockets in human skeletons from mass graves in northern, central and southern Europe.»
- **misleading** [qwen2.5-0.5b, sq03-0004] request: Answer these questions about the text: 1. Who mapped the Alta Vista Tract? 2. Which company provided… — span: «4. 1912»
- **misleading** [smollm2-1.7b, sq03-0013] request: Answer these questions about the text: 1. In what direction does the mountain system extend? 2. Wher… — span: «3. The temperature in the highest portion of the mountain range in winter is less than 9 °C (48 °F).»
- **misleading** [qwen2.5-1.5b, sq01-0003] request: Answer these questions about the text: 1. The inside of a ctenophore is lined with what? 2. What doe… — span: «2. The internal cavity contains a mouth that can usually be closed by muscles, a pharynx, a wider area in the center acting as a stomach, and a system of internal canals.»
- **misleading** [smollm2-1.7b, sq00-0013] request: How successful was the French revised efforts? — span: «The French revised efforts were unsuccessful in the aftermath of the Seven Years' War.»

## MuSiQue

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 90 | 62.322 | 0.310 | 0.290 | 0.020 | 0.158 | 0.078 | 0.241 | 0.220 | 6.694 | 1.753 | 3.819 |
| qwen2.5-0.5b \| default | 90 | 88.044 | 0.407 | 0.344 | 0.063 | 0.156 | 0.078 | 0.084 | 0.220 | 8.653 | 2.097 | 4.126 |
| qwen2.5-1.5b \| concise | 90 | 59.767 | 0.367 | 0.322 | 0.045 | 0.267 | 0.156 | 0.134 | 0.325 | 4.867 | 1.841 | 2.644 |
| qwen2.5-1.5b \| default | 90 | 102.322 | 0.457 | 0.430 | 0.027 | 0.286 | 0.167 | 0.266 | 0.328 | 8.091 | 2.325 | 3.480 |
| smollm2-1.7b \| concise | 90 | 47.656 | 0.233 | 0.197 | 0.036 | 0.300 | 0.211 | 0.141 | 0.332 | 4.922 | 1.417 | 3.474 |
| smollm2-1.7b \| default | 90 | 90.600 | 0.354 | 0.307 | 0.047 | 0.325 | 0.200 | 0.173 | 0.392 | 8.285 | 1.863 | 4.447 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 90 | 67.644 | 0.453 | 0.391 | 0.062 | 0.184 | 0.080 | 0.295 | 0.226 | 6.503 | 2.443 | 2.662 |
| qwen2.5-0.5b \| default | 90 | 96.822 | 0.458 | 0.343 | 0.114 | 0.183 | 0.080 | 0.288 | 0.231 | 8.109 | 2.485 | 3.264 |
| qwen2.5-1.5b \| concise | 90 | 65.167 | 0.343 | 0.296 | 0.047 | 0.326 | 0.193 | 0.158 | 0.401 | 3.678 | 1.746 | 2.106 |
| qwen2.5-1.5b \| default | 90 | 112.933 | 0.473 | 0.386 | 0.087 | 0.318 | 0.182 | 0.283 | 0.376 | 6.271 | 2.555 | 2.454 |
| smollm2-1.7b \| concise | 90 | 52.244 | 0.240 | 0.204 | 0.037 | 0.339 | 0.239 | 0.136 | 0.375 | 4.116 | 1.443 | 2.853 |
| smollm2-1.7b \| default | 90 | 99.611 | 0.436 | 0.372 | 0.064 | 0.366 | 0.227 | 0.226 | 0.416 | 7.835 | 2.249 | 3.484 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -25.722 [-39.781, -12.966] | -0.096 [-0.221, +0.023] | -0.065 [-0.178, +0.049] | -0.032 [-0.092, +0.020] | +0.002 [-0.051, +0.049] | ×0.73 [0.54, 0.97] |
| qwen2.5-1.5b | -42.556 [-55.446, -29.998] | -0.078 [-0.179, +0.026] | -0.101 [-0.192, -0.007] | +0.023 [-0.005, +0.056] | -0.019 [-0.078, +0.037] | ×0.62 [0.48, 0.78] |
| smollm2-1.7b | -42.944 [-57.746, -29.698] | -0.101 [-0.169, -0.033] | -0.077 [-0.156, -0.001] | -0.024 [-0.069, +0.019] | -0.025 [-0.082, +0.035] | ×0.63 [0.47, 0.84] |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -29.178 [-44.968, -14.819] | -0.063 [-0.172, +0.045] | -0.027 [-0.132, +0.070] | -0.036 [-0.098, +0.033] | +0.001 [-0.042, +0.042] | ×0.70 [0.52, 0.92] |
| qwen2.5-1.5b | -47.767 [-62.233, -33.797] | -0.111 [-0.190, -0.030] | -0.076 [-0.145, -0.011] | -0.035 [-0.075, +0.011] | +0.008 [-0.062, +0.076] | ×0.60 [0.47, 0.76] |
| smollm2-1.7b | -47.367 [-64.034, -32.441] | -0.180 [-0.269, -0.093] | -0.163 [-0.264, -0.057] | -0.016 [-0.069, +0.036] | -0.027 [-0.093, +0.040] | ×0.57 [0.43, 0.75] |

### Removable share by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.492 | 0.000 | 0.243 | 0.000 | 0.402 |
| qwen2.5-0.5b \| default | 0.000 | 0.579 | 0.000 | 0.455 | 0.000 | 0.461 |
| qwen2.5-1.5b \| concise | 0.000 | 0.596 | 0.000 | 0.442 | 0.000 | 0.413 |
| qwen2.5-1.5b \| default | 0.102 | 0.583 | 0.123 | 0.495 | 0.264 | 0.616 |
| smollm2-1.7b \| concise | 0.130 | 0.486 | 0.000 | 0.190 | 0.000 | 0.298 |
| smollm2-1.7b \| default | 0.275 | 0.651 | 0.026 | 0.358 | 0.000 | 0.453 |

### Filler by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.455 | 0.000 | 0.243 | 0.000 | 0.365 |
| qwen2.5-0.5b \| default | 0.000 | 0.544 | 0.000 | 0.355 | 0.000 | 0.346 |
| qwen2.5-1.5b \| concise | 0.000 | 0.573 | 0.000 | 0.317 | 0.000 | 0.380 |
| qwen2.5-1.5b \| default | 0.102 | 0.563 | 0.015 | 0.454 | 0.264 | 0.602 |
| smollm2-1.7b \| concise | 0.130 | 0.486 | 0.000 | 0.156 | 0.000 | 0.190 |
| smollm2-1.7b \| default | 0.275 | 0.575 | 0.026 | 0.264 | 0.000 | 0.407 |

### Wrong statements by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.036 | 0.000 | 0.000 | 0.000 | 0.037 |
| qwen2.5-0.5b \| default | 0.000 | 0.034 | 0.000 | 0.100 | 0.000 | 0.114 |
| qwen2.5-1.5b \| concise | 0.000 | 0.024 | 0.000 | 0.125 | 0.000 | 0.033 |
| qwen2.5-1.5b \| default | 0.000 | 0.021 | 0.108 | 0.041 | 0.000 | 0.013 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.000 | 0.034 | 0.000 | 0.108 |
| smollm2-1.7b \| default | 0.000 | 0.076 | 0.000 | 0.094 | 0.000 | 0.046 |

### Facts delivered by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.067 | 0.533 | 0.067 | 0.133 | 0.067 | 0.083 |
| qwen2.5-0.5b \| default | 0.133 | 0.433 | 0.133 | 0.089 | 0.067 | 0.083 |
| qwen2.5-1.5b \| concise | 0.333 | 0.400 | 0.267 | 0.200 | 0.133 | 0.267 |
| qwen2.5-1.5b \| default | 0.333 | 0.400 | 0.200 | 0.267 | 0.200 | 0.317 |
| smollm2-1.7b \| concise | 0.333 | 0.433 | 0.333 | 0.133 | 0.333 | 0.233 |
| smollm2-1.7b \| default | 0.267 | 0.500 | 0.400 | 0.133 | 0.333 | 0.317 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.800 | 0.133 | 0.800 | 0.400 | 0.867 | 0.467 |
| qwen2.5-0.5b \| default | 0.733 | 0.133 | 0.867 | 0.333 | 0.867 | 0.333 |
| qwen2.5-1.5b \| concise | 0.600 | 0.067 | 0.733 | 0.133 | 0.800 | 0.133 |
| qwen2.5-1.5b \| default | 0.667 | 0.267 | 0.733 | 0.200 | 0.733 | 0.133 |
| smollm2-1.7b \| concise | 0.667 | 0.333 | 0.667 | 0.533 | 0.600 | 0.133 |
| smollm2-1.7b \| default | 0.667 | 0.133 | 0.600 | 0.267 | 0.467 | 0.133 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing (2-hop) | all | n |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.200 | 0.200 | 15 |
| qwen2.5-0.5b \| default | 0.067 | 0.067 | 15 |
| qwen2.5-1.5b \| concise | 0.133 | 0.133 | 15 |
| qwen2.5-1.5b \| default | 0.200 | 0.200 | 15 |
| smollm2-1.7b \| concise | 0.067 | 0.067 | 15 |
| smollm2-1.7b \| default | 0.067 | 0.067 | 15 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.44; read as 'not stated' (p ≥ 0.5): 0.09 of the answers that say it, 0.18 of the others
- reader phi-3.5-mini: AUC 0.72; read as 'not stated' (p ≥ 0.5): 0.00 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 68 | 0.523 | 0.509 | 0.458 | 0.167 | 0.308 | 0.551 |
| qwen2.5-0.5b \| default | 118 | 0.550 | 0.621 | 0.454 | 0.124 | 0.393 | 0.565 |
| qwen2.5-1.5b \| concise | 78 | 0.537 | 0.644 | 0.552 | 0.134 | 0.286 | 0.612 |
| qwen2.5-1.5b \| default | 166 | 0.480 | 0.525 | 0.444 | 0.124 | 0.515 | 0.471 |
| smollm2-1.7b \| concise | 28 | 0.468 | 0.619 | 0.467 | 0.370 | 0.356 | 0.633 |
| smollm2-1.7b \| default | 88 | 0.561 | 0.476 | 0.510 | 0.172 | 0.459 | 0.589 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.84 | 1.25 | 0.86 | 1.13 |
| qwen2.5-0.5b \| default | 0.75 | 1.48 | 0.75 | 1.03 |
| qwen2.5-1.5b \| concise | 0.74 | 1.35 | 0.82 | 1.07 |
| qwen2.5-1.5b \| default | 1.08 | 1.13 | 0.83 | 0.92 |
| smollm2-1.7b \| concise | 1.22 | 0.87 | 0.70 | 1.13 |
| smollm2-1.7b \| default | 0.83 | 0.86 | 1.12 | 1.33 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.139 [-0.552, +0.245] | n/a | n/a |
| qwen2.5-0.5b \| default | -0.165 [-0.543, +0.178] | n/a | n/a |
| qwen2.5-1.5b \| concise | -0.141 [-0.425, +0.140] | n/a | n/a |
| qwen2.5-1.5b \| default | -0.130 [-0.534, +0.218] | n/a | n/a |
| smollm2-1.7b \| concise | -0.110 [-0.337, +0.112] | n/a | n/a |
| smollm2-1.7b \| default | -0.041 [-0.305, +0.213] | n/a | n/a |

### Agreement between readers

266 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking |
|---|---|---|---|
| removable | 0.712 | 0.668 | 0.600 |
| filler | 0.656 | 0.610 | 0.200 |
| wrong | 0.347 | 0.315 |  |
| κ | 0.838 | 0.815 |  |
| facts | 0.651 | 0.625 | 0.867 |
| overhead | 0.887 | 0.865 | 0.867 |

Span level (1753 spans): deleted or kept — same decision 76.7%, Cohen's κ = 0.50; filler or not — 74.7%, κ = 0.40; wrong statement or not — 90.3%, κ = 0.14.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.310 | 0.375 |
| main — qwen2.5-0.5b \| default | 0.407 | 0.494 |
| main — qwen2.5-1.5b \| concise | 0.367 | 0.425 |
| main — qwen2.5-1.5b \| default | 0.457 | 0.510 |
| main — smollm2-1.7b \| concise | 0.233 | 0.288 |
| main — smollm2-1.7b \| default | 0.354 | 0.412 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.453 | 0.547 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.458 | 0.570 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.343 | 0.417 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.473 | 0.545 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.240 | 0.296 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.436 | 0.502 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [smollm2-1.7b, mq05-0018] request: When was the country established that lies north of the region that prevailed with the disgrace of t… — span: «Document 4 provides information about the location of Kolonia Wola Szydłowiecka, which is in the Gmina Bolimów district of Skierniewice County, Łódź Voivodeship, in central Poland.»
- **uninformative** [qwen2.5-1.5b, mq03-0032] request: When did the party that controls the House of Representatives take control of the branch that gives … — span: «Senate and House of Representatives.»
- **uninformative** [qwen2.5-0.5b, mq05-0001] request: The country that Eyal Golan is from lies within a certain region. When was the area that lies immedi… — span: «- Document 6 provides: "The Kingdom of Saudi Arabia is a country situated in Southwest Asia."»
- **uninformative** [qwen2.5-1.5b, mq03-0006] request: When was the Palau de la Generalitat built in the city in which died the man called Martin of the re… — span: «Additional research would be needed to find this specific historical detail.»
- **duplicate** [qwen2.5-1.5b, mq05-0038] request: Where does the city where Veoh's headquarters is located rank in the top five largest urban areas of… — span: «1. Veoh's headquarters are located in San Diego, California.»
- **duplicate** [smollm2-1.7b, mq03-0027] request: When did the location of the head of the catholic religion with a basilica named after the same sain… — span: «In summary, the location of the head of the Catholic religion with a basilica named after the same saint as Mantua Cathedral became its own country when the Papal States were estab»
- **duplicate** [qwen2.5-0.5b, mq03-0027] request: When did the location of the head of the catholic religion with a basilica named after the same sain… — span: «The establishment of the Vatican Republic followed his death in 1871, making it a unique entity in the history of the Catholic Church.»
- **duplicate** [qwen2.5-1.5b, mq05-0022] request: What is the seat of the county sharing a border with the county in which Amy McKenzie was born? Expl… — span: «- Amy McKenzie was born in Appleton, Wisconsin.»
- **misleading** [qwen2.5-0.5b, mq05-0032] request: Which county shares a border with the county where the city in which the original singer of All By M… — span: «Therefore, the county that shares a border with the county where the city in which the original singer of "All by My Own" was born is located is **Rowan County, North Carolina**.»
- **misleading** [smollm2-1.7b, mq01-0014] request: How many campuses are there in the institution that Desert Studies Center is a part of? Explain how … — span: «The Desert Studies Center is officially operated by the California Desert Studies Consortium, a consortium of 7 CSU campuses: Dominguez Hills, Fullerton, Long Beach, Los Angeles, N»
- **misleading** [qwen2.5-1.5b, mq03-0021] request: What is the name of the castle in the birth city of the performer on the album Maria? Explain how yo… — span: «This can be determined from Document 6, which states: "Annabichl Castle is a castle in Austria."»
- **misleading** [qwen2.5-1.5b, mq01-0014] request: How many campuses are there in the institution that Desert Studies Center is a part of? Explain how … — span: «Therefore, the institution that Desert Studies Center is a part of consists of 7 campuses.»

## CUAD (contracts)

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 25.400 | 0.101 | 0.022 | 0.079 | 0.686 | 0.483 | 0.000 | 0.804 | 2.206 | 1.152 | 1.914 |
| qwen2.5-0.5b \| default | 60 | 26.817 | 0.076 | 0.022 | 0.054 | 0.686 | 0.500 | 0.000 | 0.819 | 2.358 | 1.108 | 2.128 |
| qwen2.5-1.5b \| concise | 60 | 24.333 | 0.024 | 0.010 | 0.014 | 0.769 | 0.550 | 0.000 | 0.888 | 2.016 | 1.034 | 1.951 |
| qwen2.5-1.5b \| default | 60 | 27.717 | 0.020 | 0.006 | 0.014 | 0.733 | 0.533 | 0.000 | 0.876 | 2.451 | 1.026 | 2.389 |
| smollm2-1.7b \| concise | 60 | 32.983 | 0.066 | 0.026 | 0.040 | 0.667 | 0.483 | 0.000 | 0.820 | 3.371 | 1.099 | 3.067 |
| smollm2-1.7b \| default | 60 | 35.700 | 0.082 | 0.037 | 0.045 | 0.683 | 0.500 | 0.012 | 0.825 | 3.704 | 1.127 | 3.287 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 29.750 | 0.056 | 0.012 | 0.044 | 0.692 | 0.500 | 0.000 | 0.776 | 2.200 | 1.075 | 2.046 |
| qwen2.5-0.5b \| default | 60 | 31.267 | 0.055 | 0.012 | 0.043 | 0.692 | 0.517 | 0.000 | 0.782 | 2.376 | 1.074 | 2.213 |
| qwen2.5-1.5b \| concise | 60 | 27.917 | 0.014 | 0.000 | 0.014 | 0.719 | 0.500 | 0.000 | 0.844 | 2.026 | 1.019 | 1.987 |
| qwen2.5-1.5b \| default | 60 | 31.833 | 0.015 | 0.000 | 0.015 | 0.661 | 0.450 | 0.000 | 0.822 | 2.617 | 1.020 | 2.565 |
| smollm2-1.7b \| concise | 60 | 37.867 | 0.033 | 0.004 | 0.029 | 0.567 | 0.383 | 0.000 | 0.771 | 3.408 | 1.047 | 3.254 |
| smollm2-1.7b \| default | 60 | 40.900 | 0.059 | 0.030 | 0.029 | 0.575 | 0.383 | 0.014 | 0.779 | 3.718 | 1.088 | 3.417 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -1.417 [-2.733, -0.183] | +0.025 [+0.002, +0.056] | -0.001 [-0.029, +0.029] | +0.026 [+0.000, +0.061] | +0.000 [-0.067, +0.058] | ×0.94 [0.82, 1.12] |
| qwen2.5-1.5b | -3.383 [-4.934, -1.967] | +0.004 [-0.023, +0.035] | +0.004 [-0.019, +0.030] | +0.001 [-0.015, +0.017] | +0.036 [-0.022, +0.100] | ×0.83 [0.74, 0.92] |
| smollm2-1.7b | -2.717 [-5.750, -0.600] | -0.012 [-0.049, +0.018] | -0.012 [-0.050, +0.017] | +0.000 [+0.000, +0.001] | -0.017 [-0.050, +0.000] | ×0.93 [0.87, 0.98] |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -1.517 [-3.134, -0.033] | +0.001 [-0.016, +0.020] | +0.000 [+0.000, +0.000] | +0.001 [-0.016, +0.020] | +0.000 [-0.067, +0.058] | ×0.93 [0.81, 1.10] |
| qwen2.5-1.5b | -3.917 [-5.800, -2.266] | -0.001 [-0.016, +0.014] | +0.000 [+0.000, +0.000] | -0.001 [-0.016, +0.014] | +0.058 [+0.000, +0.125] | ×0.77 [0.68, 0.87] |
| smollm2-1.7b | -3.033 [-6.601, -0.617] | -0.021 [-0.054, +0.000] | -0.021 [-0.054, +0.000] | +0.000 [+0.000, +0.000] | -0.008 [-0.058, +0.039] | ×0.94 [0.88, 0.99] |

### Removable share by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.208 | 0.169 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.162 | 0.122 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.071 | 0.022 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.032 | 0.044 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.153 | 0.091 | 0.000 |
| smollm2-1.7b \| default | 0.048 | 0.153 | 0.058 | 0.065 |

### Filler by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.041 | 0.039 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.026 | 0.058 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.039 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.025 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.049 | 0.047 | 0.000 |
| smollm2-1.7b \| default | 0.048 | 0.049 | 0.015 | 0.037 |

### Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.166 | 0.130 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.136 | 0.064 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.032 | 0.022 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.032 | 0.020 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.104 | 0.045 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.104 | 0.043 | 0.028 |

### Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.800 | 0.633 | 0.644 | 0.667 |
| qwen2.5-0.5b \| default | 0.933 | 0.567 | 0.644 | 0.600 |
| qwen2.5-1.5b \| concise | 0.867 | 0.700 | 0.644 | 0.867 |
| qwen2.5-1.5b \| default | 0.800 | 0.733 | 0.667 | 0.733 |
| smollm2-1.7b \| concise | 0.800 | 0.667 | 0.600 | 0.600 |
| smollm2-1.7b \| default | 0.800 | 0.667 | 0.600 | 0.667 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.067 | 0.000 | 0.000 | 0.200 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.000 | 0.267 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.133 |
| qwen2.5-1.5b \| default | 0.067 | 0.000 | 0.000 | 0.133 |
| smollm2-1.7b \| concise | 0.067 | 0.000 | 0.000 | 0.200 |
| smollm2-1.7b \| default | 0.067 | 0.000 | 0.000 | 0.133 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing law (k=2) | all | n |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 15 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 15 |
| qwen2.5-1.5b \| concise | 0.733 | 0.733 | 15 |
| qwen2.5-1.5b \| default | 0.933 | 0.933 | 15 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 15 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 15 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 1.00; read as 'not stated' (p ≥ 0.5): 0.96 of the answers that say it, 0.00 of the others
- reader phi-3.5-mini: AUC 1.00; read as 'not stated' (p ≥ 0.5): 0.76 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 3 | 0.699 | 0.667 | 0.504 | 0.667 | 0.931 | 0.431 |
| qwen2.5-0.5b \| default | 4 | 0.833 | 1.000 | 0.312 | 0.500 | 0.948 | 0.915 |
| qwen2.5-1.5b \| concise | 1 | 0.942 | 1.000 | 0.277 | 0.000 |  | 0.947 |
| qwen2.5-1.5b \| default | 1 | 0.480 | 1.000 | 0.931 | 0.000 |  | 0.380 |
| smollm2-1.7b \| concise | 3 | 0.451 | 0.667 | 0.843 | 0.333 | 0.610 | 0.377 |
| smollm2-1.7b \| default | 4 | 0.430 | 0.500 | 0.864 | 0.500 | 0.452 | 0.456 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.00 | 1.89 | 1.50 | 1.19 |
| qwen2.5-0.5b \| default | 0.00 | 2.87 | 0.00 | 2.06 |
| qwen2.5-1.5b \| concise | 0.00 | 4.27 | 0.00 | 0.00 |
| qwen2.5-1.5b \| default | 4.33 | 0.00 | 0.00 | 0.00 |
| smollm2-1.7b \| concise | 0.71 | 1.69 | 0.00 | 1.62 |
| smollm2-1.7b \| default | 0.46 | 0.00 | 1.73 | 1.54 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.004 [-0.079, +0.063] | +15.71 [+11.95, +19.69] | +0.92 [+0.00, +2.20] |
| qwen2.5-0.5b \| default | -0.019 [-0.085, +0.031] | +17.35 [+13.55, +21.63] | +1.32 [+0.10, +2.72] |
| qwen2.5-1.5b \| concise | -0.008 [-0.034, +0.000] | +18.87 [+16.45, +20.78] | +0.07 [+0.00, +0.30] |
| qwen2.5-1.5b \| default | -0.012 [-0.047, +0.000] | +18.68 [+16.45, +21.27] | +0.68 [+0.00, +2.31] |
| smollm2-1.7b \| concise | -0.051 [-0.129, +0.002] | +16.04 [+13.68, +18.39] | +1.84 [+0.00, +4.28] |
| smollm2-1.7b \| default | +0.072 [-0.037, +0.220] | +17.62 [+14.79, +20.33] | -1.28 [-5.13, +1.55] |

### Agreement between readers

340 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking |
|---|---|---|---|
| removable | 0.655 | 0.638 | 0.733 |
| filler | 0.478 | 0.497 | 0.645 |
| wrong | 0.565 | 0.623 |  |
| κ | 0.867 | 0.843 |  |
| facts | 0.774 | 0.781 | 0.714 |
| overhead | 0.955 | 0.954 | 1.000 |

Span level (642 spans): deleted or kept — same decision 95.6%, Cohen's κ = 0.63; filler or not — 98.0%, κ = 0.47; wrong statement or not — 96.1%, κ = 0.53.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.101 | 0.115 |
| main — qwen2.5-0.5b \| default | 0.076 | 0.091 |
| main — qwen2.5-1.5b \| concise | 0.024 | 0.033 |
| main — qwen2.5-1.5b \| default | 0.020 | 0.027 |
| main — smollm2-1.7b \| concise | 0.066 | 0.071 |
| main — smollm2-1.7b \| default | 0.082 | 0.105 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.056 | 0.070 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.055 | 0.064 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.014 | 0.019 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.015 | 0.019 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.033 | 0.045 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.059 | 0.081 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [smollm2-1.7b, cu00-0014] request: Which jurisdiction's law governs this agreement? — span: «If any terms of this Agreement are invalid or unenforceable under any statute, regulation, ordinance, executive order or other rule or law, such term shall be deemed reformed or de»
- **duplicate** [smollm2-1.7b, cu02-0008] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «2. The date of this agreement is November 14, 2002.»
- **duplicate** [qwen2.5-0.5b, cu02-0000] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «1. The title of this agreement is "Co-branding Agreement."»
- **duplicate** [qwen2.5-0.5b, cu02-0003] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «1. Title of this agreement: Content Distribution and License Agreement»
- **duplicate** [smollm2-1.7b, cu03-0012] request: Which jurisdiction's law governs this agreement? — span: «Based on the contract excerpt, the jurisdiction's law that governs this agreement is the laws of the State of Alabama.»
- **misleading** [qwen2.5-0.5b, cu01-0001] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. Which jurisdic… — span: «2. The jurisdiction's law governing this agreement is the Internal Laws of the State of Delaware.»
- **misleading** [qwen2.5-0.5b, cu01-0014] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. Which jurisdic… — span: «1. The title of this agreement is "National Nationwide Mutual Funds Scheduling Agreement."»
- **misleading** [qwen2.5-1.5b, cu02-0003] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «1. The title of this agreement is "Content Distribution and License Agreement".»
- **misleading** [smollm2-1.7b, cu01-0005] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «1. The title of this agreement is "Second Amendment to Co-Branding Agreement."»

## Poddar et al. (passages)

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 42 | 11.429 | 0.028 | 0.028 | 0.000 | 0.405 | 0.405 | 0.032 | 0.433 | 1.797 | 1.041 | 1.727 |
| qwen2.5-0.5b \| default | 42 | 23.429 | 0.100 | 0.100 | 0.000 | 0.310 | 0.310 | 0.078 | 0.357 | 3.554 | 1.147 | 3.099 |
| qwen2.5-1.5b \| concise | 42 | 18.262 | 0.043 | 0.043 | 0.000 | 0.405 | 0.405 | 0.046 | 0.416 | 1.995 | 1.087 | 1.835 |
| qwen2.5-1.5b \| default | 42 | 30.000 | 0.114 | 0.080 | 0.035 | 0.357 | 0.357 | 0.057 | 0.395 | 3.990 | 1.222 | 3.265 |
| smollm2-1.7b \| concise | 42 | 27.119 | 0.036 | 0.036 | 0.000 | 0.548 | 0.548 | 0.042 | 0.594 | 3.308 | 1.071 | 3.088 |
| smollm2-1.7b \| default | 42 | 30.952 | 0.138 | 0.138 | 0.000 | 0.548 | 0.548 | 0.141 | 0.608 | 5.869 | 1.244 | 4.717 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 42 | 13.071 | 0.030 | 0.030 | 0.000 | 0.444 | 0.444 | 0.034 | 0.446 | 1.523 | 1.044 | 1.458 |
| qwen2.5-0.5b \| default | 42 | 26.476 | 0.105 | 0.081 | 0.025 | 0.361 | 0.361 | 0.138 | 0.387 | 2.337 | 1.199 | 1.948 |
| qwen2.5-1.5b \| concise | 42 | 20.810 | 0.043 | 0.043 | 0.000 | 0.472 | 0.472 | 0.046 | 0.418 | 1.998 | 1.087 | 1.837 |
| qwen2.5-1.5b \| default | 42 | 34.595 | 0.052 | 0.047 | 0.005 | 0.444 | 0.444 | 0.053 | 0.407 | 2.411 | 1.117 | 2.158 |
| smollm2-1.7b \| concise | 42 | 31.524 | 0.034 | 0.034 | 0.000 | 0.583 | 0.583 | 0.045 | 0.634 | 3.556 | 1.066 | 3.335 |
| smollm2-1.7b \| default | 42 | 35.119 | 0.155 | 0.145 | 0.010 | 0.583 | 0.583 | 0.128 | 0.648 | 5.620 | 1.287 | 4.366 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -12.000 [-20.858, -5.214] | -0.079 [-0.163, +0.000] | -0.079 [-0.163, +0.000] | +0.000 [+0.000, +0.000] | +0.095 [+0.024, +0.190] | ×0.60 [0.36, 0.91] |
| qwen2.5-1.5b | -11.738 [-21.813, -3.856] | -0.028 [-0.079, +0.000] | -0.018 [-0.051, +0.000] | -0.009 [-0.028, +0.000] | +0.048 [+0.000, +0.119] | ×0.84 [0.54, 1.24] |
| smollm2-1.7b | -3.833 [-13.690, +9.858] | -0.079 [-0.161, -0.017] | -0.079 [-0.161, -0.017] | +0.000 [+0.000, +0.000] | +0.000 [-0.071, +0.071] | ×0.63 [0.46, 0.82] |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -13.405 [-23.333, -5.736] | -0.083 [-0.195, +0.000] | -0.055 [-0.166, +0.000] | -0.028 [-0.084, +0.000] | +0.083 [+0.000, +0.194] | ×0.73 [0.52, 1.00] |
| qwen2.5-1.5b | -13.786 [-25.720, -4.499] | -0.005 [-0.014, +0.000] | -0.005 [-0.014, +0.000] | +0.000 [+0.000, +0.000] | +0.028 [-0.056, +0.139] | ×1.01 [0.71, 1.43] |
| smollm2-1.7b | -3.595 [-14.930, +12.605] | -0.106 [-0.198, -0.026] | -0.095 [-0.182, -0.026] | -0.010 [-0.031, +0.000] | +0.000 [-0.083, +0.083] | ×0.65 [0.48, 0.82] |

### Removable share by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.092 |
| qwen2.5-0.5b \| default | 0.000 | 0.083 | 0.280 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.194 |
| qwen2.5-1.5b \| default | 0.000 | 0.069 | 0.389 |
| smollm2-1.7b \| concise | 0.015 | 0.000 | 0.117 |
| smollm2-1.7b \| default | 0.107 | 0.071 | 0.292 |

### Filler by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.092 |
| qwen2.5-0.5b \| default | 0.000 | 0.083 | 0.280 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.194 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.318 |
| smollm2-1.7b \| concise | 0.015 | 0.000 | 0.117 |
| smollm2-1.7b \| default | 0.107 | 0.071 | 0.292 |

### Wrong statements by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.069 | 0.071 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.000 |

### Facts delivered by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.400 | 0.333 | 0.500 |
| qwen2.5-0.5b \| default | 0.400 | 0.267 | 0.250 |
| qwen2.5-1.5b \| concise | 0.667 | 0.267 | 0.250 |
| qwen2.5-1.5b \| default | 0.667 | 0.133 | 0.250 |
| smollm2-1.7b \| concise | 0.533 | 0.533 | 0.583 |
| smollm2-1.7b \| default | 0.600 | 0.533 | 0.500 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.467 | 0.600 | 0.500 |
| qwen2.5-0.5b \| default | 0.467 | 0.600 | 0.583 |
| qwen2.5-1.5b \| concise | 0.333 | 0.733 | 0.667 |
| qwen2.5-1.5b \| default | 0.333 | 0.667 | 0.583 |
| smollm2-1.7b \| concise | 0.333 | 0.333 | 0.417 |
| smollm2-1.7b \| default | 0.267 | 0.267 | 0.417 |

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 1 | 0.733 | 1.000 | 0.622 | 0.000 |  | 0.706 |
| qwen2.5-0.5b \| default | 4 | 0.566 | 0.750 | 0.492 | 0.250 | 0.636 | 0.577 |
| qwen2.5-1.5b \| concise | 2 | 0.870 | 1.000 | 0.616 | 0.000 |  | 0.828 |
| qwen2.5-1.5b \| default | 9 | 0.837 | 0.958 | 0.530 | 0.143 | 0.750 | 0.789 |
| smollm2-1.7b \| concise | 3 | 0.657 | 1.000 | 0.412 | 0.000 |  | 0.597 |
| smollm2-1.7b \| default | 10 | 0.540 | 0.375 | 0.578 | 0.333 | 0.593 | 0.598 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.00 | 4.07 | 0.00 | 0.00 |
| qwen2.5-0.5b \| default | 0.91 | 1.80 | 1.35 | 0.00 |
| qwen2.5-1.5b \| concise | 0.00 | 3.63 | 0.00 | 0.00 |
| qwen2.5-1.5b \| default | 0.00 | 3.33 | 0.40 | 1.03 |
| smollm2-1.7b \| concise | 0.00 | 0.82 | 0.00 | 2.24 |
| smollm2-1.7b \| default | 0.06 | 2.80 | 2.07 | 0.65 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.041 [-0.187, +0.001] | n/a | n/a |
| qwen2.5-0.5b \| default | -0.111 [-0.493, +0.220] | n/a | n/a |
| qwen2.5-1.5b \| concise | -0.159 [-0.577, +0.000] | n/a | n/a |
| qwen2.5-1.5b \| default | -0.089 [-0.604, +0.160] | n/a | n/a |
| smollm2-1.7b \| concise | +0.067 [-0.069, +0.331] | n/a | n/a |
| smollm2-1.7b \| default | +0.155 [-0.094, +0.413] | n/a | n/a |

### Agreement between readers

119 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking |
|---|---|---|---|
| removable | 0.999 | 0.998 | 0.867 |
| filler | 0.938 | 0.928 | 1.000 |
| wrong | 0.563 | 1.000 |  |
| κ | 0.198 | 0.160 |  |
| facts | 0.714 | 0.797 | 0.923 |
| overhead | 0.988 | 0.978 | 0.733 |

Span level (380 spans): deleted or kept — same decision 95.0%, Cohen's κ = 0.64; filler or not — 95.5%, κ = 0.64; wrong statement or not — 98.4%, κ = -0.01.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.028 | 0.028 |
| main — qwen2.5-0.5b \| default | 0.100 | 0.179 |
| main — qwen2.5-1.5b \| concise | 0.043 | 0.043 |
| main — qwen2.5-1.5b \| default | 0.114 | 0.184 |
| main — smollm2-1.7b \| concise | 0.036 | 0.036 |
| main — smollm2-1.7b \| default | 0.138 | 0.148 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.030 | 0.030 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.105 | 0.164 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.043 | 0.099 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.052 | 0.137 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.034 | 0.034 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.155 | 0.159 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [smollm2-1.7b, pd00-0003] request: Out of all the main characters, who is killed/turned into a vampire first? — span: «He is bitten by Santanico Pandemonium and bleeds to death.»
- **uninformative** [qwen2.5-0.5b, pd02-0008] request: Procedure code for whooping cough vaccine? — span: «This corresponds to the current medical coding system used by healthcare providers.»
- **uninformative** [smollm2-1.7b, pd00-0016] request: When was John F. Kennedy assasinated? — span: «John F.»
- **uninformative** [qwen2.5-1.5b, pd02-0016] request: How fast does a 150cc scooter go? — span: «For example:»
- **duplicate** [smollm2-1.7b, pd02-0009] request: Was amelia earhart a feminist? — span: «She believed in empowering women and advocated for gender equality.»
- **duplicate** [qwen2.5-1.5b, pd02-0009] request: Was amelia earhart a feminist? — span: «She challenged traditional gender roles and advocated for women's rights, including their right to vote and participate equally in society.»
- **duplicate** [qwen2.5-0.5b, pd01-0010] request: Who is the clock running forwhat kind of storyis being told? — span: «The clock is running for Donald Trump to announce Pizza Rat as his running mate.»
- **misleading** [qwen2.5-1.5b, pd02-0016] request: How fast does a 150cc scooter go? — span: «- An 86 Honda Elite 150 scooter, driven by the author, could reach about 63 mph.»
- **misleading** [qwen2.5-1.5b, pd02-0000] request: How long does sperm stay in your system? — span: «3. The sperm's viability decreases significantly after 72 hours due to the process called capacitation, which is crucial for fertilization.»
- **misleading** [qwen2.5-1.5b, pd01-0010] request: Who is the clock running forwhat kind of storyis being told? — span: «The "clock" mentioned refers to the time frame within which Trump needs to make this announcement.»
