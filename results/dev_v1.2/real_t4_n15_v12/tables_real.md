# Filler in LLM answers on real texts — real_t4_n15_v12 (package v1.2.5)

Reader: Qwen/Qwen2.5-3B-Instruct; other readers: phi-3.5-mini = microsoft/Phi-3.5-mini-instruct; models: qwen2.5-0.5b = Qwen/Qwen2.5-0.5B-Instruct, qwen2.5-1.5b = Qwen/Qwen2.5-1.5B-Instruct, smollm2-1.7b = HuggingFaceTB/SmolLM2-1.7B-Instruct.

## R0. Items after the leakage filter

An item is dropped when the reader answers one of its facts from an empty message with p ≥ 0.05 (it knows the fact), or reads a fact from the gold answer with p < 0.5. Counts over the first filter and its top-ups.

| cell | pool | checked | reader knew a fact | gold unreadable | kept |
|---|---|---|---|---|---|
| SQuAD 2.0: k=1 | 60 | 19 | 1 | 3 | 15 |
| SQuAD 2.0: k=2 | 60 | 20 | 2 | 3 | 15 |
| SQuAD 2.0: k=3 | 60 | 21 | 5 | 1 | 15 |
| SQuAD 2.0: k=4 | 60 | 19 | 2 | 2 | 15 |
| SQuAD 2.0: missing (k=1) | 60 | 31 | 0 | 16 | 15 |
| SQuAD 2.0: missing (k=3) | 60 | 18 | 2 | 1 | 15 |
| MuSiQue: 2-hop | 60 | 21 | 0 | 6 | 15 |
| MuSiQue: 2-hop+explain | 60 | 34 | 1 | 18 | 15 |
| MuSiQue: 3-hop | 60 | 18 | 0 | 3 | 15 |
| MuSiQue: 3-hop+explain | 60 | 44 | 2 | 27 | 15 |
| MuSiQue: 4-hop | 60 | 21 | 0 | 6 | 15 |
| MuSiQue: 4-hop+explain | 60 | 41 | 4 | 22 | 15 |
| MuSiQue: missing (2-hop) | 60 | 18 | 0 | 3 | 15 |
| CUAD (contracts): k=1 | 60 | 16 | 0 | 1 | 15 |
| CUAD (contracts): k=2 | 60 | 15 | 0 | 0 | 15 |
| CUAD (contracts): k=3 | 60 | 15 | 0 | 0 | 15 |
| CUAD (contracts): k=1, long | 60 | 15 | 0 | 0 | 15 |
| CUAD (contracts): missing law (k=2) | 60 | 15 | 0 | 0 | 15 |
| Poddar et al. (passages): NarrativeQA | 59 | 18 | 0 | 3 | 15 |
| Poddar et al. (passages): TweetQA | 60 | 20 | 0 | 5 | 15 |
| Poddar et al. (passages): MS MARCO | 17 | 17 | 0 | 5 | 12 |

## R1. Reader check on the real items

| reader check (real items) | AUC present vs absent | gold unreadable | wrong-statement detection | nats per fact |
|---|---|---|---|---|
| main | 0.991 | 0.000 | 0.944 | 13.815 |
| phi-3.5-mini | 0.989 | 0.000 | 0.958 | 13.815 |

## R2. Filler map on simulated responses with known filler

Synthetic reports; each response = the requested facts + planted filler of known type (none / 2 / 5 fragments, or 2 with one fact left out). Recall counts planted filler fragments (95% CI, bootstrap over items); a repeat is resolved when no fact is stated twice after deletion; error of the removable share = measured − true, by characters, true = the planted filler (a summary that repeats the facts included: of two statements of a fact the first stays; v1.2.2, up to v1.2.1 the shortest set of fragments stating every fact). 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader, tolerance | responses (with filler) | no fact lost | precision | recall, planted fragments | recall per response with filler | repeats resolved | MAE of removable share | mean error of removable share |
|---|---|---|---|---|---|---|---|---|
| main, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 0.979 [0.953, 1.000] | 0.985 | 0.808 | 0.015 | -0.007 |
| main, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 0.995 [0.984, 1.000] | 0.997 | 0.846 | 0.006 | 0.003 |
| main, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.963 [0.934, 0.985] | 0.974 | 0.808 | 0.018 | -0.010 |
| phi-3.5-mini, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 |
| phi-3.5-mini, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 0.000 | 0.000 |
| phi-3.5-mini, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.958 [0.923, 0.985] | 0.941 | 1.000 | 0.014 | -0.014 |

Deleted / planted by type (τ = 0.5): main: closing 22/23, distractor 9/9, elaboration 21/21, enumeration 20/21, hedge 30/31, preamble 21/22, restatement 25/25, unrequested 38/38; phi-3.5-mini: closing 23/23, distractor 9/9, elaboration 21/21, enumeration 21/21, hedge 31/31, preamble 22/22, restatement 25/25, unrequested 38/38.

Planted filler left in (τ = 0.5):
- main: of 4 planted filler fragments left in, 100% cost more than the tolerance to delete even alone (without them the reader's answer probabilities move by more than τ); 0% could go alone but not together with the fragments deleted before them.

## R3. Filler map on real-text responses with planted filler

Responses built from each item's own document: the sentence that states each requested fact, plus planted filler — formulaic (preamble, restatement, hedge, generic remark, closing), real document sentences that answer nothing asked (elaboration: same paragraph as a fact; unrequested: another paragraph) or a summary repeating the facts. Conditions: none / 2 / 5 filler fragments, 2 with one fact left out, and 1 filler fragment + 1 wrong statement (another item's answer given as the answer). Truth is known per fragment: a fact is stated by the sentence chosen for it and by the summary, not by a sentence that only mentions its answer (v1.2). Recall counts planted filler fragments (95% CI, bootstrap over items); '… counting unscored responses' also counts the planted filler of the responses from which the reader reads no fact (no filler map) as not deleted. Error of the removable share: measured − true, by characters, true = the planted filler and wrong statements (v1.2.2; up to v1.2.1 the shortest set of fragments stating every fact: in a response with a summary, almost always the summary). Items with a missing fact, and items whose facts the document does not state in a sentence, are left out. 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader — suite | responses scored (with filler) | no fact lost | precision | recall, planted filler [95% CI] | … counting unscored responses | repeats resolved | wrong statements deleted | … and labelled wrong | MAE of removable share |
|---|---|---|---|---|---|---|---|---|---|
| main — all suites | 271 (196) of 303 | 0.908 | 0.935 | 0.799 [0.741, 0.856] | 0.736 | 0.625 | 0.913 | 0.674 | 0.105 |
| main — CUAD (contracts) | 70 (49) of 72 | 0.929 | 0.955 | 0.917 [0.860, 0.966] | 0.902 | 0.682 | 0.857 | 0.857 | 0.102 |
| main — MuSiQue | 91 (68) of 108 | 0.879 | 0.907 | 0.687 [0.587, 0.786] | 0.597 | 0.417 | 1.000 | 0.643 | 0.127 |
| main — Poddar et al. (passages) | 41 (28) of 48 | 1.000 | 1.000 | 0.901 [0.806, 0.986] | 0.821 | 0.733 | 1.000 | 1.000 | 0.059 |
| main — SQuAD 2.0 | 69 (51) of 75 | 0.870 | 0.909 | 0.777 [0.631, 0.904] | 0.734 | 0.737 | 0.846 | 0.385 | 0.106 |
| main — all suites, prefix ablation | 271 (196) | 0.915 | 0.943 | 0.766 [0.704, 0.824] |  | 0.625 |  |  | 0.108 |
| phi-3.5-mini — all suites | 285 (208) of 303 | 0.860 | 0.923 | 0.857 [0.814, 0.896] | 0.829 | 0.778 | 0.804 | 0.588 | 0.141 |
| phi-3.5-mini — CUAD (contracts) | 69 (48) of 72 | 0.899 | 0.944 | 0.899 [0.844, 0.954] | 0.877 | 0.818 | 0.923 | 0.923 | 0.134 |
| phi-3.5-mini — MuSiQue | 98 (74) of 108 | 0.786 | 0.879 | 0.797 [0.718, 0.879] | 0.759 | 0.640 | 0.667 | 0.467 | 0.176 |
| phi-3.5-mini — Poddar et al. (passages) | 44 (31) of 48 | 1.000 | 1.000 | 0.932 [0.892, 0.973] | 0.885 | 0.933 | 1.000 | 0.625 | 0.045 |
| phi-3.5-mini — SQuAD 2.0 | 74 (55) of 75 | 0.838 | 0.918 | 0.858 [0.778, 0.929] | 0.852 | 0.789 | 0.733 | 0.400 | 0.159 |
| phi-3.5-mini — all suites, prefix ablation | 285 (208) | 0.863 | 0.925 | 0.821 [0.772, 0.868] |  | 0.753 |  |  | 0.141 |

Deleted / planted by type (τ = 0.5 nats): main: closing 62/73, elaboration 65/78, generic 52/65, hedge 59/70, preamble 57/75, restatement 46/64, unrequested 41/53; phi-3.5-mini: closing 70/76, elaboration 71/83, generic 59/69, hedge 60/76, preamble 67/76, restatement 61/67, unrequested 42/55.

- main: of 96 planted filler fragments left in, 93% cost more than the tolerance to delete even alone (without them the reader's answer probabilities move by more than τ); 7% could go alone but not together with the fragments deleted before them.
- main, prefix ablation: planted filler deleted by suite — CUAD (contracts) 0.833; MuSiQue 0.675; Poddar et al. (passages) 0.873; SQuAD 2.0 0.760.
- main, responses whose facts the reader all reads (p ≥ 0.5): 156; no fact lost 0.929, precision 0.954, planted filler deleted 0.941.
- main: planted filler labelled as a wrong statement in 0.065 of 558 fragments; κ 0.87 with every fact, 0.57 with one fact left out; at τ = 2 nats no fact lost in 0.845, planted filler deleted 0.897.
- phi-3.5-mini: of 72 planted filler fragments left in, 88% cost more than the tolerance to delete even alone (without them the reader's answer probabilities move by more than τ); 12% could go alone but not together with the fragments deleted before them.
- phi-3.5-mini, prefix ablation: planted filler deleted by suite — CUAD (contracts) 0.866; MuSiQue 0.753; Poddar et al. (passages) 0.919; SQuAD 2.0 0.819.
- phi-3.5-mini, responses whose facts the reader all reads (p ≥ 0.5): 169; no fact lost 0.882, precision 0.940, planted filler deleted 0.953.
- phi-3.5-mini: planted filler labelled as a wrong statement in 0.154 of 583 fragments; κ 0.89 with every fact, 0.57 with one fact left out; at τ = 2 nats no fact lost in 0.796, planted filler deleted 0.964.

## R4. Filler map vs human span labels (Poddar et al., 2025)

Responses of 10 LLMs (GPT-3.5/4o-mini, Llama, Gemma, Mistral, Phi) to the passage questions of Poddar et al.'s data (NarrativeQA, TweetQA, MS MARCO; reference answer of at most 6 words), with human span labels. A fragment is 'not minimal' when most of its characters carry a label other than MINANS; precision and recall of deletion are measured against that (unlabelled fragments left out). The fact is the reference answer; only responses that state it are scored (annotators label minimal vs extra text whether or not the answer is right, so for another answer the labels and the filler map measure different things; up to v1.1 the reader was given the fragments marked MINANS to find such a response's own answer, so the labels entered the method). 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it.

| reader | responses scored | precision | recall | F1 | Cohen's κ | F1 by characters | minimal answer kept | Spearman, share per response | deleted share (mean) | human share (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 74 | 0.981 | 0.765 | 0.860 | 0.657 | 0.870 | 0.970 | 0.682 | 0.275 | 0.416 |
| main, prefix ablation | 74 | 0.972 | 0.757 | 0.851 | 0.637 | 0.863 | 0.955 | 0.677 | 0.272 | 0.416 |
| phi-3.5-mini | 79 | 0.993 | 0.856 | 0.919 | 0.775 | 0.926 | 0.986 | 0.699 | 0.312 | 0.434 |
| phi-3.5-mini, prefix ablation | 79 | 0.992 | 0.819 | 0.897 | 0.725 | 0.906 | 0.986 | 0.696 | 0.303 | 0.434 |

- main: 74 of the 96 responses that state the reference answer scored; 93 labelled responses without it are left out.
- main: of 32 kept fragments the annotators marked as not minimal, 91% cost more than the tolerance to delete alone, 9% were blocked in the queue.
- phi-3.5-mini: 79 of the 96 responses that state the reference answer scored; 93 labelled responses without it are left out.
- phi-3.5-mini: of 23 kept fragments the annotators marked as not minimal, 100% cost more than the tolerance to delete alone, 0% were blocked in the queue.

| human label: share deleted | main | phi-3.5-mini |
|---|---|---|
| minimal answer (MINANS) | 0.03 (n=66) | 0.01 (n=71) |
| additional information (ADDINFO) | 0.88 (n=40) | 0.73 (n=45) |
| explanation (EXPLAIN) | 0.83 (n=36) | 0.96 (n=46) |
| conversational (CONVERS) | 0.80 (n=5) | 0.83 (n=6) |
| irrelevant (IRREL) | 0.64 (n=55) | 0.87 (n=63) |
| unlabelled (NONE) | 1.00 (n=2) | 1.00 (n=2) |

| subset | main | phi-3.5-mini |
|---|---|---|
| MS MARCO | F1 0.98, κ 0.92 (n=8) | F1 0.86, κ 0.64 (n=8) |
| NarrativeQA | F1 0.80, κ 0.55 (n=35) | F1 0.99, κ 0.96 (n=35) |
| TweetQA | F1 0.88, κ 0.72 (n=31) | F1 0.85, κ 0.63 (n=36) |

Ranking of the 10 LLMs by mean deleted share vs by mean human non-minimal share: main: Kendall τ = 0.81 (p = 0.011, 7 LLMs); phi-3.5-mini: Kendall τ = 0.79 (p = 0.006, 8 LLMs).

## R5. Baselines on the same responses

Every method on the same responses: the filler map of each reader, the LLM judge and Selective Context when they were run (drybench baselines), and three cheap metrics. Fragment level, removable (filler or wrong statement) vs needed: F1, Cohen's κ, and the AUC of the method's score (the filler map: minus the fragment's leave-one-out cost, or its yes/no decision where no cost is stored; the judge's p(filler) + p(wrong); Selective Context: lower self-information = more removable). Response level, removable share by characters (95% CIs: bootstrap over items): pairs ranked right = over pairs whose true share differs by ≥ 0.2, the share where the method scores the response with more filler higher (ties count half; 0.5 = chance), over all pairs and over pairs of responses to the same item; Spearman with the true share; 'shorter but more removable' = such pairs where the more removable response is the shorter one, share the method ranks higher (ties apart; length: 0 by construction); partial Spearman with the true share given length. Selective Context deletes, least informative fragment first, as many characters as the first reader's filler map (equal budget); the judge's share is its expected removable share. Planted responses: the true share is the planted filler and wrong statements (a summary that repeats the facts is planted filler); a fact stated twice may go either way, so such fragments are left out of the fragment-level scores. First table: the responses every method scores; second: also the answers from which a reader reads none of the requested facts — the filler map's verdict on such an answer is that it delivers nothing, so all of it counts as removable (as in the results tables: a failed answer); answers whose gold the reader cannot read stay out.

### Planted filler in real texts (the R3 responses)

266 of 303 responses scored by every method (filler map (reader main) 271; filler map (reader phi-3.5-mini) 285); 860 fragments with a truth label; 10045 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.873 | 0.717 | 0.924 | 0.88 [0.84, 0.92] | 0.92 [0.85, 0.97] | 0.769 | 0.77 [0.69, 0.85] | 0.831 | 0.019 | +0.77 [+0.69, +0.85] |
| filler map (reader phi-3.5-mini) | 0.892 | 0.745 | 0.908 | 0.85 [0.80, 0.90] | 0.93 [0.85, 0.98] | 0.699 | 0.70 [0.60, 0.80] | 0.788 | 0.000 | +0.70 [+0.60, +0.80] |
| length (characters) |  |  |  | 0.54 [0.49, 0.61] | 0.91 [0.81, 0.97] | 0.098 | 0.10 [-0.00, 0.22] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.59 [0.53, 0.66] | 0.91 [0.81, 0.97] | 0.197 | 0.20 [0.09, 0.34] | 0.127 | 0.002 | +0.38 [+0.28, +0.46] |
| stop-word share |  |  |  | 0.66 [0.61, 0.72] | 0.74 [0.59, 0.85] | 0.335 | 0.34 [0.21, 0.44] | 0.727 | 0.001 | +0.36 [+0.24, +0.47] |

With the answers that deliver nothing to a reader (303 responses; all of such an answer counts as removable):

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.857 | 0.658 | 0.853 | 0.84 [0.80, 0.89] | 0.89 [0.80, 0.95] | 0.696 | 0.70 [0.60, 0.78] | 0.809 | 0.025 | +0.70 [+0.60, +0.78] |
| filler map (reader phi-3.5-mini) | 0.878 | 0.704 | 0.876 | 0.83 [0.78, 0.88] | 0.90 [0.80, 0.96] | 0.669 | 0.67 [0.56, 0.76] | 0.792 | 0.004 | +0.67 [+0.56, +0.77] |
| length (characters) |  |  |  | 0.51 [0.46, 0.58] | 0.90 [0.82, 0.96] | 0.044 | 0.04 [-0.05, 0.17] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.57 [0.50, 0.64] | 0.90 [0.82, 0.96] | 0.149 | 0.15 [0.03, 0.29] | 0.124 | 0.002 | +0.39 [+0.30, +0.47] |
| stop-word share |  |  |  | 0.68 [0.63, 0.73] | 0.75 [0.62, 0.85] | 0.366 | 0.37 [0.26, 0.47] | 0.745 | 0.002 | +0.38 [+0.27, +0.50] |

### Human span labels (the R4 responses, Poddar et al.)

71 of 96 responses scored by every method (filler map (reader main) 74; filler map (reader phi-3.5-mini) 79); 194 fragments with a truth label; 121 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.871 | 0.684 | 0.921 | 0.87 [0.79, 0.94] | n/a | 0.725 | 0.72 [0.54, 0.88] | 0.264 | 0.413 | +0.36 [+0.09, +0.68] |
| filler map (reader phi-3.5-mini) | 0.925 | 0.805 | 0.921 | 0.88 [0.80, 0.95] | n/a | 0.728 | 0.73 [0.55, 0.91] | 0.331 | 0.405 | +0.37 [+0.10, +0.73] |
| length (characters) |  |  |  | 0.91 [0.84, 0.96] | n/a | 0.719 | 0.72 [0.58, 0.82] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.92 [0.85, 0.97] | n/a | 0.729 | 0.73 [0.60, 0.82] | 0.124 | 0.008 | +0.18 [+0.01, +0.33] |
| stop-word share |  |  |  | 0.57 [0.45, 0.68] | n/a | 0.110 | 0.11 [-0.12, 0.32] | 0.504 | 0.000 | +0.01 [-0.21, +0.23] |

With the answers that deliver nothing to a reader (90 responses; all of such an answer counts as removable):

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.897 | 0.634 | 0.834 | 0.85 [0.78, 0.93] | n/a | 0.646 | 0.65 [0.48, 0.81] | 0.348 | 0.355 | +0.27 [+0.06, +0.54] |
| filler map (reader phi-3.5-mini) | 0.925 | 0.725 | 0.851 | 0.87 [0.80, 0.94] | n/a | 0.654 | 0.65 [0.46, 0.82] | 0.348 | 0.355 | +0.23 [-0.02, +0.53] |
| length (characters) |  |  |  | 0.94 [0.89, 0.97] | n/a | 0.748 | 0.75 [0.63, 0.83] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.94 [0.90, 0.97] | n/a | 0.755 | 0.75 [0.65, 0.83] | 0.121 | 0.007 | +0.15 [+0.02, +0.30] |
| stop-word share |  |  |  | 0.54 [0.42, 0.64] | n/a | 0.062 | 0.06 [-0.16, 0.25] | 0.496 | 0.000 | +0.01 [-0.22, +0.20] |

Responses with a fragment labelled minimal answer (84 of 96; in the others no fragment is mostly minimal answer — the answer, if any, sits inside a fragment that is mostly something else, and the filler map does not delete a fragment that carries the answer):

65 of 84 responses scored by every method (filler map (reader main) 66; filler map (reader phi-3.5-mini) 71); 186 fragments with a truth label; 43 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.892 | 0.735 | 0.941 | 0.94 [0.89, 0.99] | n/a | 0.923 | 0.92 [0.84, 0.98] | 0.721 | 0.116 | +0.78 [+0.53, +0.96] |
| filler map (reader phi-3.5-mini) | 0.952 | 0.875 | 0.939 | 0.98 [0.94, 1.00] | n/a | 0.972 | 0.97 [0.93, 1.00] | 0.884 | 0.093 | +0.92 [+0.81, +1.00] |
| length (characters) |  |  |  | 0.96 [0.93, 0.99] | n/a | 0.829 | 0.83 [0.75, 0.87] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.96 [0.92, 0.99] | n/a | 0.823 | 0.82 [0.74, 0.87] | 0.047 | 0.023 | +0.11 [-0.06, +0.25] |
| stop-word share |  |  |  | 0.58 [0.45, 0.71] | n/a | 0.134 | 0.13 [-0.10, 0.36] | 0.581 | 0.000 | +0.02 [-0.23, +0.24] |

## R6. Across suites

Removable share (reader main):

| model \| prompt | SQuAD 2.0 | MuSiQue | CUAD (contracts) | Poddar et al. (passages) |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.169 | 0.344 | 0.023 | 0.024 |
| qwen2.5-0.5b \| default | 0.201 | 0.339 | 0.038 | 0.025 |
| qwen2.5-1.5b \| concise | 0.098 | 0.305 | 0.000 | 0.063 |
| qwen2.5-1.5b \| default | 0.098 | 0.369 | 0.000 | 0.098 |
| smollm2-1.7b \| concise | 0.196 | 0.229 | 0.014 | 0.029 |
| smollm2-1.7b \| default | 0.165 | 0.348 | 0.029 | 0.124 |

Facts delivered (reader main):

| model \| prompt | SQuAD 2.0 | MuSiQue | CUAD (contracts) | Poddar et al. (passages) |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.693 | 0.205 | 0.819 | 0.500 |
| qwen2.5-0.5b \| default | 0.669 | 0.254 | 0.772 | 0.476 |
| qwen2.5-1.5b \| concise | 0.843 | 0.311 | 0.925 | 0.524 |
| qwen2.5-1.5b \| default | 0.836 | 0.285 | 0.858 | 0.500 |
| smollm2-1.7b \| concise | 0.731 | 0.315 | 0.781 | 0.619 |
| smollm2-1.7b \| default | 0.690 | 0.340 | 0.814 | 0.643 |

Kendall τ between suites of the model ranking by removable share (default prompt; 3 models): SQuAD 2.0 vs MuSiQue -1.00; SQuAD 2.0 vs CUAD (contracts) +1.00; SQuAD 2.0 vs Poddar et al. (passages) -0.33; MuSiQue vs CUAD (contracts) -1.00; MuSiQue vs Poddar et al. (passages) +0.33; CUAD (contracts) vs Poddar et al. (passages) -0.33.

## R7. Cost per response

Wall-clock seconds per response (logged from v1.0). The filler map: the whole score of a model's answer (deletion search, kinds of deleted fragments, the gold answer); its reader never reads the source text. The LLM judge: one pass per fragment over the source text, the request and the response. Selective Context: one pass over the request and the response. The map is timed on the run's answers, the baselines on the checks' responses; `drybench annotation-report` times all of them on the same responses.

| method — GPU | responses timed | seconds per response (median) | seconds per response (mean) |
|---|---|---|---|
| filler map, reader main — Tesla T4 | 1872 | 0.265 | 1.175 |
| filler map, reader phi-3.5-mini — Tesla T4 | 1872 | 0.328 | 1.649 |

## SQuAD 2.0

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 47.417 | 0.169 | 0.120 | 0.049 | 0.693 | 0.450 | 0.026 | 0.729 | 2.947 | 1.299 | 2.269 | 0.000 |
| qwen2.5-0.5b \| default | 60 | 52.917 | 0.201 | 0.138 | 0.063 | 0.669 | 0.417 | 0.036 | 0.715 | 3.332 | 1.347 | 2.474 | 0.000 |
| qwen2.5-1.5b \| concise | 60 | 39.517 | 0.098 | 0.044 | 0.055 | 0.843 | 0.717 | 0.019 | 0.859 | 2.675 | 1.152 | 2.322 | 0.000 |
| qwen2.5-1.5b \| default | 60 | 47.400 | 0.098 | 0.061 | 0.037 | 0.836 | 0.667 | 0.014 | 0.869 | 3.293 | 1.144 | 2.878 | 0.000 |
| smollm2-1.7b \| concise | 60 | 48.783 | 0.196 | 0.089 | 0.107 | 0.731 | 0.533 | 0.025 | 0.757 | 3.853 | 1.359 | 2.836 | 0.000 |
| smollm2-1.7b \| default | 60 | 54.033 | 0.165 | 0.067 | 0.098 | 0.690 | 0.500 | 0.026 | 0.732 | 4.758 | 1.273 | 3.739 | 0.000 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 55.283 | 0.173 | 0.056 | 0.117 | 0.654 | 0.450 | 0.015 | 0.726 | 2.736 | 1.295 | 2.112 | 0.000 |
| qwen2.5-0.5b \| default | 60 | 61.283 | 0.203 | 0.071 | 0.131 | 0.613 | 0.367 | 0.043 | 0.700 | 3.147 | 1.342 | 2.346 | 0.000 |
| qwen2.5-1.5b \| concise | 60 | 46.333 | 0.116 | 0.036 | 0.080 | 0.793 | 0.633 | 0.015 | 0.852 | 2.683 | 1.183 | 2.267 | 0.000 |
| qwen2.5-1.5b \| default | 60 | 54.817 | 0.104 | 0.024 | 0.079 | 0.774 | 0.583 | 0.011 | 0.862 | 3.348 | 1.148 | 2.917 | 0.000 |
| smollm2-1.7b \| concise | 60 | 56.433 | 0.198 | 0.007 | 0.190 | 0.672 | 0.467 | 0.018 | 0.744 | 4.033 | 1.369 | 2.947 | 0.000 |
| smollm2-1.7b \| default | 60 | 62.467 | 0.170 | 0.014 | 0.156 | 0.662 | 0.467 | 0.021 | 0.730 | 4.191 | 1.283 | 3.267 | 0.000 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -5.500 [-11.100, -0.566] | -0.023 [-0.064, +0.017] | -0.017 [-0.055, +0.013] | -0.006 [-0.029, +0.019] | +0.024 [-0.017, +0.072] | ×0.92 [0.83, 1.00] | 52 of 60 |
| qwen2.5-1.5b | -7.883 [-11.283, -4.766] | +0.000 [-0.045, +0.044] | -0.018 [-0.052, +0.013] | +0.018 [-0.006, +0.047] | +0.007 [-0.028, +0.046] | ×0.81 [0.71, 0.91] | 58 of 60 |
| smollm2-1.7b | -5.250 [-9.700, -1.267] | +0.026 [-0.018, +0.071] | +0.024 [-0.003, +0.057] | +0.001 [-0.036, +0.040] | +0.040 [-0.024, +0.103] | ×0.89 [0.80, 0.98] | 53 of 60 |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -6.000 [-12.367, -0.466] | -0.021 [-0.072, +0.030] | -0.016 [-0.048, +0.011] | -0.005 [-0.048, +0.037] | +0.042 [-0.004, +0.092] | ×0.91 [0.81, 0.99] | 51 of 60 |
| qwen2.5-1.5b | -8.483 [-12.217, -5.083] | +0.017 [-0.029, +0.059] | +0.012 [-0.005, +0.035] | +0.006 [-0.038, +0.045] | +0.019 [-0.022, +0.071] | ×0.82 [0.72, 0.92] | 58 of 60 |
| smollm2-1.7b | -6.033 [-11.133, -1.515] | +0.025 [-0.018, +0.075] | -0.006 [-0.016, +0.000] | +0.031 [-0.012, +0.081] | +0.010 [-0.044, +0.062] | ×0.91 [0.82, 1.00] | 53 of 60 |

### Removable share by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.147 | 0.259 | 0.221 |
| qwen2.5-0.5b \| default | 0.000 | 0.268 | 0.213 | 0.265 |
| qwen2.5-1.5b \| concise | 0.000 | 0.150 | 0.102 | 0.138 |
| qwen2.5-1.5b \| default | 0.000 | 0.202 | 0.096 | 0.096 |
| smollm2-1.7b \| concise | 0.000 | 0.217 | 0.309 | 0.222 |
| smollm2-1.7b \| default | 0.017 | 0.145 | 0.219 | 0.246 |

### Filler by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.116 | 0.189 | 0.144 |
| qwen2.5-0.5b \| default | 0.000 | 0.176 | 0.168 | 0.166 |
| qwen2.5-1.5b \| concise | 0.000 | 0.073 | 0.042 | 0.059 |
| qwen2.5-1.5b \| default | 0.000 | 0.115 | 0.052 | 0.078 |
| smollm2-1.7b \| concise | 0.000 | 0.084 | 0.141 | 0.113 |
| smollm2-1.7b \| default | 0.000 | 0.050 | 0.104 | 0.097 |

### Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.031 | 0.070 | 0.077 |
| qwen2.5-0.5b \| default | 0.000 | 0.092 | 0.045 | 0.100 |
| qwen2.5-1.5b \| concise | 0.000 | 0.077 | 0.060 | 0.079 |
| qwen2.5-1.5b \| default | 0.000 | 0.087 | 0.044 | 0.018 |
| smollm2-1.7b \| concise | 0.000 | 0.133 | 0.168 | 0.109 |
| smollm2-1.7b \| default | 0.017 | 0.095 | 0.115 | 0.149 |

### Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.733 | 0.600 | 0.778 | 0.661 |
| qwen2.5-0.5b \| default | 0.667 | 0.567 | 0.800 | 0.644 |
| qwen2.5-1.5b \| concise | 0.933 | 0.733 | 0.867 | 0.839 |
| qwen2.5-1.5b \| default | 0.933 | 0.700 | 0.889 | 0.822 |
| smollm2-1.7b \| concise | 0.800 | 0.667 | 0.733 | 0.722 |
| smollm2-1.7b \| default | 0.667 | 0.700 | 0.689 | 0.706 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.267 | 0.200 | 0.000 | 0.000 |
| qwen2.5-0.5b \| default | 0.333 | 0.133 | 0.000 | 0.000 |
| qwen2.5-1.5b \| concise | 0.067 | 0.067 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.067 | 0.067 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.200 | 0.133 | 0.000 | 0.000 |
| smollm2-1.7b \| default | 0.200 | 0.200 | 0.000 | 0.000 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing (k=1) | missing (k=3) | all | n |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.000 | 30 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.000 | 30 |
| qwen2.5-1.5b \| concise | 0.067 | 0.000 | 0.033 | 30 |
| qwen2.5-1.5b \| default | 0.067 | 0.067 | 0.067 | 30 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.000 | 30 |
| smollm2-1.7b \| default | 0.067 | 0.000 | 0.033 | 30 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.31; read as 'not stated' (p ≥ 0.5): 0.25 of the answers that say it, 0.14 of the others
- reader phi-3.5-mini: AUC 0.71; read as 'not stated' (p ≥ 0.5): 0.00 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 19 | 0.627 | 0.528 | 0.620 | 0.684 | 0.631 | 0.589 |
| qwen2.5-0.5b \| default | 22 | 0.609 | 0.525 | 0.608 | 0.636 | 0.539 | 0.754 |
| qwen2.5-1.5b \| concise | 9 | 0.454 | 0.531 | 0.487 | 0.500 | 0.760 | 0.479 |
| qwen2.5-1.5b \| default | 11 | 0.589 | 0.500 | 0.502 | 0.300 | 0.662 | 0.606 |
| smollm2-1.7b \| concise | 15 | 0.769 | 0.621 | 0.549 | 0.143 | 0.649 | 0.769 |
| smollm2-1.7b \| default | 12 | 0.788 | 0.648 | 0.531 | 0.250 | 0.700 | 0.796 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.54 | 0.43 | 0.70 | 2.09 |
| qwen2.5-0.5b \| default | 0.33 | 0.55 | 1.23 | 1.45 |
| qwen2.5-1.5b \| concise | 0.63 | 1.59 | 1.32 | 0.78 |
| qwen2.5-1.5b \| default | 0.39 | 0.84 | 1.16 | 1.60 |
| smollm2-1.7b \| concise | 0.18 | 0.71 | 0.93 | 1.88 |
| smollm2-1.7b \| default | 0.00 | 0.69 | 0.76 | 2.50 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | +0.215 [+0.050, +0.427] | +14.99 [+9.23, +20.65] | +2.69 [+0.08, +6.00] |
| qwen2.5-0.5b \| default | +0.306 [+0.113, +0.545] | +15.38 [+7.94, +22.64] | +2.61 [-0.99, +5.50] |
| qwen2.5-1.5b \| concise | -0.035 [-0.155, +0.066] | +13.19 [+9.28, +16.32] | +1.14 [-0.13, +2.78] |
| qwen2.5-1.5b \| default | +0.095 [-0.048, +0.275] | +15.54 [+11.05, +19.05] | +1.46 [+0.01, +3.41] |
| smollm2-1.7b \| concise | +0.150 [-0.030, +0.356] | +12.43 [+8.10, +16.09] | +3.02 [+0.50, +6.24] |
| smollm2-1.7b \| default | +0.192 [+0.030, +0.405] | +11.32 [+6.68, +15.32] | +2.83 [+0.28, +6.29] |

### Agreement between readers

328 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking [95% CI] |
|---|---|---|---|
| removable | 0.784 | 0.769 | +0.73 [+0.47, +1.00] |
| filler | 0.405 | 0.349 | +0.20 [-0.20, +0.60] |
| wrong | 0.525 | 0.485 |  |
| κ | 0.803 | 0.797 |  |
| facts | 0.617 | 0.602 | +0.87 [+0.14, +1.00] |
| overhead | 0.942 | 0.937 | +1.00 [+0.73, +1.00] |

Span level (866 spans): deleted or kept — same decision 91.5%, Cohen's κ = 0.71; filler or not — 90.9%, κ = 0.33; wrong statement or not — 90.9%, κ = 0.53.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.169 | 0.171 |
| main — qwen2.5-0.5b \| default | 0.201 | 0.203 |
| main — qwen2.5-1.5b \| concise | 0.098 | 0.122 |
| main — qwen2.5-1.5b \| default | 0.098 | 0.113 |
| main — smollm2-1.7b \| concise | 0.196 | 0.196 |
| main — smollm2-1.7b \| default | 0.165 | 0.192 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.173 | 0.196 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.203 | 0.235 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.116 | 0.131 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.104 | 0.129 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.198 | 0.228 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.170 | 0.199 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-1.5b, sq01-0019] request: Answer these questions about the text: 1. Over how many studies have shown that violence is more com… — span: «2. About half of all variation in homicide rates can be accounted for by differences in the amount of inequality in each province or state.»
- **uninformative** [qwen2.5-1.5b, sq03-0011] request: Answer these questions about the text: 1. What year did the the case go before the supreme court? 2.… — span: «4. One way of digital civil disobedience that can have far reaching consequences is threatening government officials.»
- **uninformative** [qwen2.5-0.5b, sq02-0011] request: Answer these questions about the text: 1. In which case did the Court of Justice state that refusal … — span: «1. In Gebhard v Consiglio dell’Ordine degli Avvocati e Procuratori di Milano, the Court of Justice held that to be "established" means to participate in economic life "on a stable »
- **uninformative** [qwen2.5-0.5b, sq03-0011] request: Answer these questions about the text: 1. What year did the the case go before the supreme court? 2.… — span: «4. One way of digital civil disobedience that can have far reaching consequences is spreading false information.»
- **duplicate** [qwen2.5-1.5b, sq02-0018] request: Answer these questions about the text: 1. What rule did some native live under? 2. In between French… — span: «3. In Father Le Loutre's War, the Mi'kmaq and the Abenaki were engaged.»
- **duplicate** [qwen2.5-0.5b, sq02-0018] request: Answer these questions about the text: 1. What rule did some native live under? 2. In between French… — span: «3. The Iroquois Confederation dominated much of present-day Upstate New York and the Ohio Country, although the latter also included Algonquian-speaking populations of Delaware and»
- **duplicate** [qwen2.5-0.5b, sq02-0003] request: Answer these questions about the text: 1. In what century did missionaries notably establish church … — span: «2. The South African Schools Act was recognized under the South African Schools Act.»
- **duplicate** [smollm2-1.7b, sq02-0018] request: Answer these questions about the text: 1. What rule did some native live under? 2. In between French… — span: «3. The tribes in Father Le Loutre's War were the Mi'kmaq and the Abenaki.»
- **misleading** [smollm2-1.7b, sq03-0000] request: Answer these questions about the text: 1. What did Guo Shoujing do for calendars? 2. What was the En… — span: «3. The Chinese name of Gou's calendar is the Shoushi Li (授時暦).»
- **misleading** [smollm2-1.7b, sq00-0013] request: How successful was the French revised efforts? — span: «The French revised efforts were unsuccessful in the aftermath of the Seven Years' War.»
- **misleading** [qwen2.5-1.5b, sq01-0003] request: Answer these questions about the text: 1. The inside of a ctenophore is lined with what? 2. What doe… — span: «2. The internal cavity contains a mouth that can usually be closed by muscles, a pharynx, a wider area in the center that acts as a stomach, and a system of internal canals.»
- **misleading** [smollm2-1.7b, sq01-0004] request: Answer these questions about the text: 1. What is the name of another algorithm useful for convenien… — span: «1. The name of another algorithm useful for conveniently testing the primality of large numbers is the Miller-Rabin primality test.»

## MuSiQue

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 90 | 54.111 | 0.344 | 0.300 | 0.044 | 0.205 | 0.133 | 0.161 | 0.260 | 5.683 | 1.836 | 3.094 | 0.022 |
| qwen2.5-0.5b \| default | 90 | 69.078 | 0.339 | 0.282 | 0.057 | 0.254 | 0.178 | 0.130 | 0.295 | 6.794 | 1.744 | 3.896 | 0.044 |
| qwen2.5-1.5b \| concise | 90 | 46.533 | 0.305 | 0.237 | 0.069 | 0.311 | 0.200 | 0.141 | 0.362 | 3.162 | 1.636 | 1.932 | 0.000 |
| qwen2.5-1.5b \| default | 90 | 85.033 | 0.369 | 0.263 | 0.106 | 0.285 | 0.178 | 0.173 | 0.347 | 6.730 | 1.879 | 3.581 | 0.033 |
| smollm2-1.7b \| concise | 90 | 47.889 | 0.229 | 0.156 | 0.073 | 0.315 | 0.222 | 0.134 | 0.345 | 4.841 | 1.427 | 3.394 | 0.000 |
| smollm2-1.7b \| default | 90 | 90.256 | 0.348 | 0.238 | 0.110 | 0.340 | 0.211 | 0.164 | 0.406 | 8.249 | 1.825 | 4.519 | 0.022 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 90 | 59.789 | 0.330 | 0.200 | 0.130 | 0.218 | 0.148 | 0.278 | 0.292 | 5.234 | 1.770 | 2.957 | 0.022 |
| qwen2.5-0.5b \| default | 90 | 76.456 | 0.376 | 0.188 | 0.187 | 0.255 | 0.170 | 0.261 | 0.330 | 5.747 | 1.886 | 3.048 | 0.044 |
| qwen2.5-1.5b \| concise | 90 | 50.867 | 0.300 | 0.138 | 0.162 | 0.365 | 0.227 | 0.169 | 0.439 | 2.699 | 1.678 | 1.608 | 0.000 |
| qwen2.5-1.5b \| default | 90 | 94.100 | 0.390 | 0.147 | 0.243 | 0.358 | 0.227 | 0.252 | 0.412 | 5.925 | 1.987 | 2.982 | 0.033 |
| smollm2-1.7b \| concise | 90 | 52.511 | 0.237 | 0.060 | 0.177 | 0.353 | 0.261 | 0.124 | 0.389 | 4.075 | 1.432 | 2.846 | 0.000 |
| smollm2-1.7b \| default | 90 | 99.267 | 0.427 | 0.186 | 0.241 | 0.381 | 0.239 | 0.215 | 0.430 | 7.664 | 2.197 | 3.488 | 0.022 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -14.967 [-24.534, -5.854] | -0.058 [-0.144, +0.028] | -0.047 [-0.122, +0.030] | -0.011 [-0.073, +0.046] | -0.049 [-0.119, +0.009] | ×0.90 [0.62, 1.20] | 37 of 90 |
| qwen2.5-1.5b | -38.500 [-52.633, -26.133] | -0.045 [-0.137, +0.046] | -0.014 [-0.098, +0.068] | -0.030 [-0.072, +0.012] | +0.026 [-0.039, +0.091] | ×0.53 [0.40, 0.71] | 47 of 90 |
| smollm2-1.7b | -42.367 [-57.222, -29.218] | -0.110 [-0.182, -0.046] | -0.071 [-0.147, -0.005] | -0.039 [-0.097, +0.022] | -0.025 [-0.082, +0.035] | ×0.64 [0.48, 0.84] | 44 of 90 |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -16.667 [-27.623, -6.376] | -0.074 [-0.152, +0.001] | +0.006 [-0.068, +0.082] | -0.079 [-0.155, -0.008] | -0.037 [-0.099, +0.023] | ×0.91 [0.69, 1.14] | 44 of 90 |
| qwen2.5-1.5b | -43.233 [-59.191, -29.087] | -0.066 [-0.156, +0.017] | +0.002 [-0.068, +0.063] | -0.068 [-0.140, +0.003] | +0.007 [-0.065, +0.080] | ×0.51 [0.41, 0.64] | 54 of 90 |
| smollm2-1.7b | -46.756 [-63.513, -31.733] | -0.174 [-0.259, -0.096] | -0.121 [-0.202, -0.045] | -0.054 [-0.117, +0.007] | -0.027 [-0.092, +0.040] | ×0.58 [0.44, 0.75] | 49 of 90 |

### Removable share by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.465 | 0.000 | 0.551 | 0.000 | 0.344 |
| qwen2.5-0.5b \| default | 0.079 | 0.467 | 0.223 | 0.433 | 0.000 | 0.408 |
| qwen2.5-1.5b \| concise | 0.000 | 0.556 | 0.000 | 0.475 | 0.000 | 0.275 |
| qwen2.5-1.5b \| default | 0.088 | 0.506 | 0.119 | 0.430 | 0.000 | 0.472 |
| smollm2-1.7b \| concise | 0.108 | 0.479 | 0.000 | 0.218 | 0.000 | 0.294 |
| smollm2-1.7b \| default | 0.229 | 0.628 | 0.026 | 0.390 | 0.000 | 0.447 |

### Filler by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.435 | 0.000 | 0.499 | 0.000 | 0.252 |
| qwen2.5-0.5b \| default | 0.079 | 0.379 | 0.223 | 0.362 | 0.000 | 0.330 |
| qwen2.5-1.5b \| concise | 0.000 | 0.507 | 0.000 | 0.282 | 0.000 | 0.221 |
| qwen2.5-1.5b \| default | 0.088 | 0.365 | 0.119 | 0.243 | 0.000 | 0.372 |
| smollm2-1.7b \| concise | 0.108 | 0.365 | 0.000 | 0.150 | 0.000 | 0.154 |
| smollm2-1.7b \| default | 0.229 | 0.449 | 0.026 | 0.174 | 0.000 | 0.335 |

### Wrong statements by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.031 | 0.000 | 0.053 | 0.000 | 0.092 |
| qwen2.5-0.5b \| default | 0.000 | 0.088 | 0.000 | 0.071 | 0.000 | 0.078 |
| qwen2.5-1.5b \| concise | 0.000 | 0.048 | 0.000 | 0.193 | 0.000 | 0.054 |
| qwen2.5-1.5b \| default | 0.000 | 0.141 | 0.000 | 0.186 | 0.000 | 0.101 |
| smollm2-1.7b \| concise | 0.000 | 0.114 | 0.000 | 0.068 | 0.000 | 0.140 |
| smollm2-1.7b \| default | 0.000 | 0.179 | 0.000 | 0.215 | 0.000 | 0.112 |

### Facts delivered by item design — reader main

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.200 | 0.567 | 0.133 | 0.178 | 0.067 | 0.083 |
| qwen2.5-0.5b \| default | 0.333 | 0.600 | 0.133 | 0.156 | 0.200 | 0.100 |
| qwen2.5-1.5b \| concise | 0.333 | 0.433 | 0.333 | 0.200 | 0.200 | 0.367 |
| qwen2.5-1.5b \| default | 0.267 | 0.500 | 0.200 | 0.244 | 0.267 | 0.233 |
| smollm2-1.7b \| concise | 0.400 | 0.433 | 0.333 | 0.156 | 0.333 | 0.233 |
| smollm2-1.7b \| default | 0.333 | 0.500 | 0.400 | 0.156 | 0.333 | 0.317 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | 2-hop | 2-hop+explain | 3-hop | 3-hop+explain | 4-hop | 4-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.733 | 0.133 | 0.733 | 0.400 | 0.867 | 0.267 |
| qwen2.5-0.5b \| default | 0.600 | 0.133 | 0.800 | 0.333 | 0.733 | 0.333 |
| qwen2.5-1.5b \| concise | 0.533 | 0.200 | 0.667 | 0.200 | 0.800 | 0.000 |
| qwen2.5-1.5b \| default | 0.667 | 0.200 | 0.733 | 0.133 | 0.733 | 0.067 |
| smollm2-1.7b \| concise | 0.600 | 0.333 | 0.667 | 0.467 | 0.600 | 0.133 |
| smollm2-1.7b \| default | 0.600 | 0.133 | 0.600 | 0.200 | 0.467 | 0.133 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing (2-hop) | all | n |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.067 | 0.067 | 15 |
| qwen2.5-0.5b \| default | 0.067 | 0.067 | 15 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 15 |
| qwen2.5-1.5b \| default | 0.067 | 0.067 | 15 |
| smollm2-1.7b \| concise | 0.067 | 0.067 | 15 |
| smollm2-1.7b \| default | 0.067 | 0.067 | 15 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.58; read as 'not stated' (p ≥ 0.5): 0.00 of the answers that say it, 0.20 of the others
- reader phi-3.5-mini: AUC 0.63; read as 'not stated' (p ≥ 0.5): 0.00 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 57 | 0.641 | 0.562 | 0.562 | 0.246 | 0.214 | 0.797 |
| qwen2.5-0.5b \| default | 51 | 0.482 | 0.508 | 0.536 | 0.529 | 0.506 | 0.575 |
| qwen2.5-1.5b \| concise | 42 | 0.626 | 0.695 | 0.559 | 0.385 | 0.609 | 0.673 |
| qwen2.5-1.5b \| default | 78 | 0.470 | 0.575 | 0.379 | 0.208 | 0.443 | 0.490 |
| smollm2-1.7b \| concise | 23 | 0.519 | 0.625 | 0.412 | 0.435 | 0.360 | 0.754 |
| smollm2-1.7b \| default | 72 | 0.557 | 0.436 | 0.480 | 0.208 | 0.353 | 0.619 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.91 | 1.60 | 0.75 | 0.67 |
| qwen2.5-0.5b \| default | 1.13 | 0.82 | 0.94 | 1.05 |
| qwen2.5-1.5b \| concise | 0.78 | 1.25 | 0.76 | 1.21 |
| qwen2.5-1.5b \| default | 1.27 | 1.14 | 0.96 | 0.57 |
| smollm2-1.7b \| concise | 1.13 | 0.91 | 0.55 | 1.35 |
| smollm2-1.7b \| default | 0.90 | 0.79 | 0.95 | 1.48 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | +0.048 [-0.225, +0.299] | n/a | n/a |
| qwen2.5-0.5b \| default | +0.069 [-0.200, +0.344] | n/a | n/a |
| qwen2.5-1.5b \| concise | -0.006 [-0.238, +0.215] | n/a | n/a |
| qwen2.5-1.5b \| default | -0.163 [-0.440, +0.076] | n/a | n/a |
| smollm2-1.7b \| concise | -0.013 [-0.198, +0.165] | n/a | n/a |
| smollm2-1.7b \| default | +0.115 [-0.129, +0.371] | n/a | n/a |

### Agreement between readers

290 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking [95% CI] |
|---|---|---|---|
| removable | 0.699 | 0.678 | +0.47 [+0.07, +1.00] |
| filler | 0.465 | 0.440 | +0.73 [+0.07, +1.00] |
| wrong | 0.540 | 0.525 |  |
| κ | 0.848 | 0.845 |  |
| facts | 0.757 | 0.760 | +0.60 [-0.07, +1.00] |
| overhead | 0.895 | 0.876 | +1.00 [+0.73, +1.00] |

Span level (903 spans): deleted or kept — same decision 74.5%, Cohen's κ = 0.49; filler or not — 72.0%, κ = 0.32; wrong statement or not — 76.9%, κ = 0.27.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.344 | 0.385 |
| main — qwen2.5-0.5b \| default | 0.339 | 0.422 |
| main — qwen2.5-1.5b \| concise | 0.305 | 0.335 |
| main — qwen2.5-1.5b \| default | 0.369 | 0.423 |
| main — smollm2-1.7b \| concise | 0.229 | 0.283 |
| main — smollm2-1.7b \| default | 0.348 | 0.409 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.330 | 0.481 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.376 | 0.485 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.300 | 0.387 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.390 | 0.463 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.237 | 0.291 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.427 | 0.487 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-0.5b, mq05-0007] request: When did the city where Greenwood Laboratory School is located become capitol of the state where the… — span: «The state of Wisconsin was formed on March 19, 1839, and it was the first state to be incorporated as a territory.»
- **uninformative** [qwen2.5-1.5b, mq05-0007] request: When did the city where Greenwood Laboratory School is located become capitol of the state where the… — span: «Margaret Mayo was born in Brownsville, Illinois.»
- **uninformative** [smollm2-1.7b, mq03-0019] request: When was the last time the team that Malcolm Briggs was a member of beat the 1894-95 FA cup winner? … — span: «However, the document does not provide this information.»
- **uninformative** [smollm2-1.7b, mq03-0027] request: When did the location of the head of the catholic religion with a basilica named after the same sain… — span: «The Papal States were established in 1859, and they were eventually dissolved in 1870.»
- **duplicate** [qwen2.5-1.5b, mq03-0043] request: Who is the president of the newly declared independent country part of the Commission of Truth and F… — span: «This can be determined from Document 5, which states that Democratic Republic of Timor-Leste (East Timor) was declared independent on 20 May 2002.»
- **duplicate** [smollm2-1.7b, mq01-0018] request: What was the population of the European country the Huguenots felt kinship for emigration to before … — span: «This information is found in Document 1, which states: "After the revocation of the Edict of Nantes, the Dutch Republic received the largest group of Huguenot refugees, an estimate»
- **duplicate** [smollm2-1.7b, mq00-0016] request: As a result of the black death it is estimated that the population of the country for Bâton à feu wa… — span: «However, it is mentioned that the Black Death was one of the most devastating pandemics in human history, peaking in Europe between 1347 and 1350 with 30% to 65% of the population »
- **duplicate** [qwen2.5-0.5b, mq03-0027] request: When did the location of the head of the catholic religion with a basilica named after the same sain… — span: «The name is taken from Vatican Hill, the geographic location of the state.»
- **misleading** [smollm2-1.7b, mq03-0021] request: What is the name of the castle in the birth city of the performer on the album Maria? Explain how yo… — span: «The first document, Princess Maria Elizabeth of Sweden, mentions that she was born in Örebro Castle, which is located in Örebro, Sweden.»
- **misleading** [smollm2-1.7b, mq01-0031] request: What company is the record label for Oh Woman, Oh Why a part of? Explain how you found it. — span: «Based on the documents, the record label for "Oh Woman, Oh Why" is Apple Records.»
- **misleading** [smollm2-1.7b, mq01-0013] request: Who was picked before the player who has the most points in an NBA season in the NBA draft? Explain … — span: «By comparing the information from the different documents, I was able to determine that Kevin Durant was picked before the player with the most points in an NBA season, which is To»
- **misleading** [qwen2.5-1.5b, mq05-0007] request: When did the city where Greenwood Laboratory School is located become capitol of the state where the… — span: «The city of Greenwood is located in Vernon County, Wisconsin.»

## CUAD (contracts)

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 27.567 | 0.023 | 0.004 | 0.019 | 0.819 | 0.700 | 0.000 | 0.913 | 2.062 | 1.040 | 1.983 | 0.000 |
| qwen2.5-0.5b \| default | 60 | 33.317 | 0.038 | 0.019 | 0.019 | 0.772 | 0.617 | 0.024 | 0.894 | 2.491 | 1.074 | 2.320 | 0.000 |
| qwen2.5-1.5b \| concise | 60 | 25.917 | 0.000 | 0.000 | 0.000 | 0.925 | 0.867 | 0.000 | 0.975 | 1.900 | 1.000 | 1.900 | 0.000 |
| qwen2.5-1.5b \| default | 60 | 28.217 | 0.000 | 0.000 | 0.000 | 0.858 | 0.750 | 0.000 | 0.968 | 2.252 | 1.000 | 2.252 | 0.000 |
| smollm2-1.7b \| concise | 60 | 33.017 | 0.014 | 0.004 | 0.010 | 0.781 | 0.667 | 0.000 | 0.909 | 3.022 | 1.020 | 2.963 | 0.000 |
| smollm2-1.7b \| default | 60 | 35.717 | 0.029 | 0.013 | 0.016 | 0.814 | 0.700 | 0.021 | 0.921 | 3.288 | 1.041 | 3.160 | 0.000 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 60 | 32.467 | 0.047 | 0.006 | 0.041 | 0.733 | 0.550 | 0.000 | 0.824 | 2.297 | 1.074 | 2.139 | 0.000 |
| qwen2.5-0.5b \| default | 60 | 38.667 | 0.063 | 0.022 | 0.041 | 0.703 | 0.517 | 0.028 | 0.795 | 2.790 | 1.110 | 2.514 | 0.000 |
| qwen2.5-1.5b \| concise | 60 | 29.833 | 0.000 | 0.000 | 0.000 | 0.842 | 0.767 | 0.000 | 0.929 | 1.946 | 1.000 | 1.946 | 0.000 |
| qwen2.5-1.5b \| default | 60 | 32.533 | 0.000 | 0.000 | 0.000 | 0.756 | 0.600 | 0.000 | 0.887 | 2.450 | 1.000 | 2.450 | 0.000 |
| smollm2-1.7b \| concise | 60 | 37.900 | 0.022 | 0.004 | 0.019 | 0.669 | 0.483 | 0.000 | 0.832 | 3.164 | 1.032 | 3.066 | 0.000 |
| smollm2-1.7b \| default | 60 | 40.900 | 0.048 | 0.029 | 0.019 | 0.694 | 0.500 | 0.027 | 0.851 | 3.405 | 1.071 | 3.180 | 0.000 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -5.750 [-15.134, -0.733] | -0.014 [-0.046, +0.003] | -0.015 [-0.046, +0.000] | +0.001 [+0.000, +0.003] | +0.047 [+0.011, +0.094] | ×0.83 [0.69, 0.96] | 57 of 60 |
| qwen2.5-1.5b | -2.300 [-3.834, -0.450] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.067 [+0.019, +0.125] | ×0.84 [0.77, 0.92] | 59 of 60 |
| smollm2-1.7b | -2.700 [-5.784, -0.533] | -0.009 [-0.028, +0.000] | -0.009 [-0.028, +0.000] | +0.000 [+0.000, +0.000] | -0.033 [-0.100, +0.033] | ×0.94 [0.89, 0.99] | 57 of 60 |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -6.200 [-16.250, -0.817] | -0.014 [-0.046, +0.003] | -0.015 [-0.046, +0.000] | +0.001 [+0.000, +0.003] | +0.031 [+0.000, +0.072] | ×0.83 [0.70, 0.94] | 57 of 60 |
| qwen2.5-1.5b | -2.700 [-4.633, -0.416] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.086 [+0.036, +0.144] | ×0.79 [0.71, 0.88] | 59 of 60 |
| smollm2-1.7b | -3.000 [-6.634, -0.533] | -0.020 [-0.052, +0.000] | -0.020 [-0.052, +0.000] | +0.000 [+0.000, +0.000] | -0.025 [-0.083, +0.025] | ×0.95 [0.89, 1.00] | 57 of 60 |

### Removable share by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.054 | 0.035 | 0.000 |
| qwen2.5-0.5b \| default | 0.063 | 0.049 | 0.035 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.039 | 0.014 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.039 | 0.015 | 0.063 |

### Filler by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.014 | 0.000 |
| qwen2.5-0.5b \| default | 0.063 | 0.000 | 0.014 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.014 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.015 | 0.037 |

### Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.054 | 0.021 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.049 | 0.021 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.039 | 0.000 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.039 | 0.000 | 0.026 |

### Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.867 | 0.633 | 0.844 | 0.933 |
| qwen2.5-0.5b \| default | 0.867 | 0.600 | 0.756 | 0.867 |
| qwen2.5-1.5b \| concise | 0.933 | 0.900 | 0.933 | 0.933 |
| qwen2.5-1.5b \| default | 0.867 | 0.833 | 0.867 | 0.867 |
| smollm2-1.7b \| concise | 0.800 | 0.767 | 0.822 | 0.733 |
| smollm2-1.7b \| default | 0.867 | 0.767 | 0.822 | 0.800 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2 | k=3 | k=1, long |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.067 | 0.000 | 0.000 | 0.067 |
| qwen2.5-0.5b \| default | 0.067 | 0.000 | 0.000 | 0.133 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.067 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.000 | 0.067 |
| smollm2-1.7b \| concise | 0.067 | 0.000 | 0.000 | 0.133 |
| smollm2-1.7b \| default | 0.067 | 0.000 | 0.000 | 0.067 |

### Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing law (k=2) | all | n |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 15 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 15 |
| qwen2.5-1.5b \| concise | 0.467 | 0.467 | 15 |
| qwen2.5-1.5b \| default | 1.000 | 1.000 | 15 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 15 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 15 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 1.00; read as 'not stated' (p ≥ 0.5): 1.00 of the answers that say it, 0.00 of the others
- reader phi-3.5-mini: AUC 1.00; read as 'not stated' (p ≥ 0.5): 0.95 of the answers that say it, 0.00 of the others

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 1 | 0.099 | 0.000 | 0.515 | 1.000 | 0.133 |  |
| qwen2.5-0.5b \| default | 4 | 0.273 | 0.167 | 0.626 | 1.000 | 0.186 |  |
| qwen2.5-1.5b \| concise | 0 |  |  |  |  |  |  |
| qwen2.5-1.5b \| default | 0 |  |  |  |  |  |  |
| smollm2-1.7b \| concise | 1 | 0.000 | 0.000 | 0.540 | 0.000 |  | 0.000 |
| smollm2-1.7b \| default | 2 | 0.471 | 0.000 | 0.748 | 0.000 |  | 0.456 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | compensation |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 2.74 | 0.00 | 0.00 |
| qwen2.5-0.5b \| default | 2.48 | 0.11 | 0.00 |
| qwen2.5-1.5b \| concise |  |  |  |
| qwen2.5-1.5b \| default |  |  |  |
| smollm2-1.7b \| concise | 3.81 | 0.00 | 0.00 |
| smollm2-1.7b \| default | 1.55 | 0.00 | 2.12 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.005 [-0.020, +0.000] | +20.21 [+18.07, +22.42] | +0.51 [+0.00, +1.72] |
| qwen2.5-0.5b \| default | -0.065 [-0.215, +0.000] | +19.52 [+17.45, +21.60] | -4.51 [-16.80, +1.44] |
| qwen2.5-1.5b \| concise | +0.000 [+0.000, +0.000] | +21.08 [+18.71, +24.25] | +0.00 [+0.00, +0.00] |
| qwen2.5-1.5b \| default | +0.000 [+0.000, +0.000] | +19.98 [+17.96, +22.29] | +0.00 [+0.00, +0.00] |
| smollm2-1.7b \| concise | -0.015 [-0.049, +0.000] | +18.80 [+16.43, +21.46] | +0.62 [+0.00, +1.92] |
| smollm2-1.7b \| default | +0.003 [-0.042, +0.062] | +17.57 [+12.40, +21.26] | +0.09 [-1.69, +1.78] |

### Agreement between readers

348 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking [95% CI] |
|---|---|---|---|
| removable | 0.758 | 0.757 | +1.00 [+0.43, +1.00] |
| filler | 0.571 | 0.541 | +0.71 [+0.09, +1.00] |
| wrong | 0.403 | 0.457 |  |
| κ | 0.877 | 0.861 |  |
| facts | 0.688 | 0.687 | +0.73 [+0.07, +1.00] |
| overhead | 0.961 | 0.955 | +1.00 [+0.73, +1.00] |

Span level (626 spans): deleted or kept — same decision 97.9%, Cohen's κ = 0.66; filler or not — 99.0%, κ = 0.66; wrong statement or not — 97.9%, κ = 0.37.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.023 | 0.049 |
| main — qwen2.5-0.5b \| default | 0.038 | 0.064 |
| main — qwen2.5-1.5b \| concise | 0.000 | 0.000 |
| main — qwen2.5-1.5b \| default | 0.000 | 0.000 |
| main — smollm2-1.7b \| concise | 0.014 | 0.023 |
| main — smollm2-1.7b \| default | 0.029 | 0.059 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.047 | 0.051 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.063 | 0.066 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.000 | 0.000 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.000 | 0.000 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.022 | 0.022 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.048 | 0.058 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-0.5b, cu00-0003] request: What is the title of this agreement? — span: «ESPACIO INFORMATION TECHNOLOGY, SA., a public limited company with place of business in P° de la Castellana, N° 259, 28046, Madrid, SPAIN, and represented by Mr. Carlos Lafitte as »
- **uninformative** [qwen2.5-0.5b, cu02-0014] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «The omitted portions, marked by [****], have been separately filed with the Commission."»
- **uninformative** [qwen2.5-0.5b, cu00-0003] request: What is the title of this agreement? — span: «SILICON SMELTERS << Pty >> Ltd, registration nr 1998/019036/07, VAT n° 4310178506, Registered office at Beyersnek Road, PO Box 657 Polokwane - South Africa, represented by Benoit E»
- **uninformative** [qwen2.5-0.5b, cu00-0003] request: What is the title of this agreement? — span: «Parties to this agreement»
- **duplicate** [smollm2-1.7b, cu03-0012] request: Which jurisdiction's law governs this agreement? — span: «Specifically, the contract states: "The laws of the State of Alabama, without giving effect to its conflict of laws provisions."»
- **duplicate** [smollm2-1.7b, cu02-0008] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «2. The date of this agreement is November 14, 2002.»
- **misleading** [smollm2-1.7b, cu01-0006] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. Which jurisdic… — span: «1. The title of this agreement is "OUTSOURCING AGREEMENT."»
- **misleading** [smollm2-1.7b, cu03-0002] request: Which jurisdiction's law governs this agreement? — span: «This is stated in the 10th paragraph of the agreement.»
- **misleading** [qwen2.5-0.5b, cu01-0010] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. Which jurisdic… — span: «1. The title of this agreement is "Athens Bancshares Corporation up to Shares (subject to increase up to shares) COMMON SHARES ($.01 Par Value) Subscription Price $10.00 Per Share »
- **misleading** [qwen2.5-0.5b, cu02-0012] request: Answer these questions about the contract: 1. What is the title of this agreement? 2. What is the da… — span: «1. The title of this agreement is "Unconditional Capital Maintenance Agreement."»

## Poddar et al. (passages)

### Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 42 | 10.167 | 0.024 | 0.024 | 0.000 | 0.500 | 0.500 | 0.027 | 0.545 | 1.373 | 1.036 | 1.326 | 0.000 |
| qwen2.5-0.5b \| default | 42 | 11.786 | 0.025 | 0.025 | 0.000 | 0.476 | 0.476 | 0.030 | 0.540 | 2.047 | 1.040 | 1.968 | 0.000 |
| qwen2.5-1.5b \| concise | 42 | 15.119 | 0.063 | 0.047 | 0.016 | 0.524 | 0.524 | 0.051 | 0.548 | 2.379 | 1.093 | 2.176 | 0.000 |
| qwen2.5-1.5b \| default | 42 | 19.476 | 0.098 | 0.084 | 0.014 | 0.500 | 0.500 | 0.068 | 0.535 | 2.753 | 1.173 | 2.348 | 0.000 |
| smollm2-1.7b \| concise | 42 | 27.095 | 0.029 | 0.029 | 0.000 | 0.619 | 0.619 | 0.031 | 0.650 | 2.939 | 1.062 | 2.766 | 0.024 |
| smollm2-1.7b \| default | 42 | 30.929 | 0.124 | 0.124 | 0.000 | 0.643 | 0.643 | 0.115 | 0.700 | 4.989 | 1.220 | 4.088 | 0.000 |

Same, reader phi-3.5-mini:

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 42 | 11.548 | 0.024 | 0.000 | 0.024 | 0.526 | 0.526 | 0.028 | 0.546 | 1.212 | 1.036 | 1.169 | 0.000 |
| qwen2.5-0.5b \| default | 42 | 13.548 | 0.025 | 0.000 | 0.025 | 0.541 | 0.541 | 0.030 | 0.561 | 1.838 | 1.040 | 1.768 | 0.000 |
| qwen2.5-1.5b \| concise | 42 | 17.333 | 0.052 | 0.000 | 0.052 | 0.583 | 0.583 | 0.029 | 0.548 | 2.228 | 1.082 | 2.058 | 0.000 |
| qwen2.5-1.5b \| default | 42 | 22.214 | 0.099 | 0.046 | 0.053 | 0.556 | 0.556 | 0.035 | 0.538 | 2.536 | 1.208 | 2.099 | 0.000 |
| smollm2-1.7b \| concise | 42 | 31.524 | 0.027 | 0.000 | 0.027 | 0.639 | 0.639 | 0.035 | 0.697 | 3.302 | 1.056 | 3.127 | 0.024 |
| smollm2-1.7b \| default | 42 | 35.119 | 0.156 | 0.050 | 0.105 | 0.658 | 0.658 | 0.134 | 0.741 | 5.305 | 1.305 | 4.065 | 0.000 |

### 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -1.619 [-5.049, +2.476] | -0.002 [-0.005, +0.000] | -0.002 [-0.005, +0.000] | +0.000 [+0.000, +0.000] | +0.024 [-0.048, +0.095] | ×0.70 [0.51, 0.91] | 22 of 42 |
| qwen2.5-1.5b | -4.357 [-9.239, -0.690] | -0.012 [-0.033, +0.003] | -0.013 [-0.034, +0.000] | +0.001 [+0.000, +0.004] | +0.024 [+0.000, +0.071] | ×0.96 [0.75, 1.21] | 23 of 42 |
| smollm2-1.7b | -3.833 [-13.690, +9.858] | -0.076 [-0.153, -0.017] | -0.076 [-0.153, -0.017] | +0.000 [+0.000, +0.000] | -0.024 [-0.119, +0.048] | ×0.62 [0.45, 0.81] | 28 of 42 |

Same, reader phi-3.5-mini:

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -2.000 [-5.857, +2.572] | -0.002 [-0.007, +0.000] | +0.000 [+0.000, +0.000] | -0.002 [-0.007, +0.000] | -0.027 [-0.108, +0.054] | ×0.68 [0.50, 0.91] | 21 of 42 |
| qwen2.5-1.5b | -4.881 [-10.667, -0.499] | -0.012 [-0.031, +0.000] | -0.011 [-0.033, +0.000] | -0.001 [-0.022, +0.019] | +0.028 [+0.000, +0.083] | ×0.97 [0.74, 1.27] | 22 of 42 |
| smollm2-1.7b | -3.595 [-14.930, +12.605] | -0.099 [-0.182, -0.024] | -0.025 [-0.064, +0.000] | -0.074 [-0.148, -0.010] | +0.000 [-0.083, +0.083] | ×0.64 [0.49, 0.81] | 30 of 42 |

### Removable share by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.095 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.101 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.301 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.393 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.117 |
| smollm2-1.7b \| default | 0.094 | 0.060 | 0.292 |

### Filler by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.095 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.101 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.226 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.336 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.117 |
| smollm2-1.7b \| default | 0.094 | 0.060 | 0.292 |

### Wrong statements by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.000 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.075 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.057 |
| smollm2-1.7b \| concise | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.000 |

### Facts delivered by item design — reader main

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.400 | 0.600 | 0.500 |
| qwen2.5-0.5b \| default | 0.400 | 0.533 | 0.500 |
| qwen2.5-1.5b \| concise | 0.733 | 0.467 | 0.333 |
| qwen2.5-1.5b \| default | 0.733 | 0.400 | 0.333 |
| smollm2-1.7b \| concise | 0.533 | 0.733 | 0.583 |
| smollm2-1.7b \| default | 0.600 | 0.800 | 0.500 |

### Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | NarrativeQA | TweetQA | MS MARCO |
|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.467 | 0.333 | 0.500 |
| qwen2.5-0.5b \| default | 0.467 | 0.333 | 0.500 |
| qwen2.5-1.5b \| concise | 0.267 | 0.467 | 0.583 |
| qwen2.5-1.5b \| default | 0.267 | 0.533 | 0.500 |
| smollm2-1.7b \| concise | 0.333 | 0.267 | 0.417 |
| smollm2-1.7b \| default | 0.267 | 0.133 | 0.417 |

### Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 1 | 0.792 | 1.000 | 0.854 | 0.000 |  | 1.000 |
| qwen2.5-0.5b \| default | 1 | 0.958 | 1.000 | 0.729 | 0.000 |  | 0.875 |
| qwen2.5-1.5b \| concise | 2 | 0.958 | 1.000 | 0.833 | 0.000 |  | 1.000 |
| qwen2.5-1.5b \| default | 6 | 0.712 | 0.778 | 0.817 | 0.500 | 0.500 | 1.000 |
| smollm2-1.7b \| concise | 2 | 0.732 | 1.000 | 0.670 | 0.000 |  | 0.735 |
| smollm2-1.7b \| default | 9 | 0.582 | 0.286 | 0.675 | 0.333 | 0.630 | 0.604 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.00 | 7.73 | 0.00 | 0.00 |
| qwen2.5-0.5b \| default | 0.00 | 7.33 | 0.00 | 0.00 |
| qwen2.5-1.5b \| concise | 0.00 | 2.47 | 0.00 | 1.70 |
| qwen2.5-1.5b \| default | 0.00 | 1.30 | 1.38 | 1.14 |
| smollm2-1.7b \| concise | 0.00 | 0.00 | 0.00 | 2.56 |
| smollm2-1.7b \| default | 0.41 | 2.41 | 1.68 | 0.70 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.054 [-0.202, +0.000] | n/a | n/a |
| qwen2.5-0.5b \| default | -0.057 [-0.218, +0.000] | n/a | n/a |
| qwen2.5-1.5b \| concise | +0.022 [-0.221, +0.298] | n/a | n/a |
| qwen2.5-1.5b \| default | +0.160 [-0.239, +0.552] | n/a | n/a |
| smollm2-1.7b \| concise | +0.094 [+0.000, +0.347] | n/a | n/a |
| smollm2-1.7b \| default | +0.157 [-0.094, +0.419] | n/a | n/a |

### Agreement between readers

151 responses scored by both readers.

| main vs phi-3.5-mini | Spearman, pooled | Spearman, within model × prompt | Kendall τ, group ranking [95% CI] |
|---|---|---|---|
| removable | 0.998 | 0.996 | +1.00 [+0.83, +1.00] |
| filler | 0.590 | 0.694 | +0.77 [+0.60, +1.00] |
| wrong |  |  |  |
| κ | 0.193 | 0.180 |  |
| facts | 0.410 | 0.431 | +0.36 [-0.60, +0.97] |
| overhead | 0.986 | 0.974 | +1.00 [+0.87, +1.00] |

Span level (177 spans): deleted or kept — same decision 95.5%, Cohen's κ = 0.80; filler or not — 89.8%, κ = 0.35; wrong statement or not — 91.0%, κ = 0.00.

### Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.024 | 0.024 |
| main — qwen2.5-0.5b \| default | 0.025 | 0.025 |
| main — qwen2.5-1.5b \| concise | 0.063 | 0.063 |
| main — qwen2.5-1.5b \| default | 0.098 | 0.108 |
| main — smollm2-1.7b \| concise | 0.029 | 0.029 |
| main — smollm2-1.7b \| default | 0.124 | 0.134 |
| phi-3.5-mini — qwen2.5-0.5b \| concise | 0.024 | 0.024 |
| phi-3.5-mini — qwen2.5-0.5b \| default | 0.025 | 0.025 |
| phi-3.5-mini — qwen2.5-1.5b \| concise | 0.052 | 0.052 |
| phi-3.5-mini — qwen2.5-1.5b \| default | 0.099 | 0.099 |
| phi-3.5-mini — smollm2-1.7b \| concise | 0.027 | 0.027 |
| phi-3.5-mini — smollm2-1.7b \| default | 0.156 | 0.160 |

### Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-1.5b, pd02-0009] request: Was amelia earhart a feminist? — span: «She believed in empowering women and worked to challenge gender stereotypes.»
- **uninformative** [qwen2.5-1.5b, pd02-0000] request: How long does sperm stay in your system? — span: «However, the most important thing to remember is the capacitation, which is the sperm's ability to fertilize an egg.»
- **uninformative** [smollm2-1.7b, pd01-0009] request: Who is visiting these states? — span: «He has yet to visit South Carolina, South Dakota, and Utah.»
- **uninformative** [smollm2-1.7b, pd00-0017] request: After Hargrave's affairs husband finds out about their affair, what do the 2 men do? — span: «This causes Laura to faint, and she is found by her father, who then realizes that Hargrave has been threatening his daughter.»
- **duplicate** [smollm2-1.7b, pd02-0009] request: Was amelia earhart a feminist? — span: «She believed in empowering women and advocated for gender equality.»
- **misleading** [qwen2.5-1.5b, pd02-0016] request: How fast does a 150cc scooter go? — span: «A 150cc scooter will generally top out at about 60-65 mph.»
