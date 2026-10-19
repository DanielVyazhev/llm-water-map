# Filler in LLM answers on synthetic reports — t4_n20_v12 (package v1.2.5)

Readers: main = Qwen/Qwen2.5-3B-Instruct.

## T1. Filler map on simulated responses with known filler

Synthetic reports; each response = the requested facts + planted filler of known type (none / 2 / 5 fragments, or 2 with one fact left out). Recall counts planted filler fragments (95% CI, bootstrap over items); a repeat is resolved when no fact is stated twice after deletion; error of the removable share = measured − true, by characters, true = the planted filler (a summary that repeats the facts included: of two statements of a fact the first stays; v1.2.2, up to v1.2.1 the shortest set of fragments stating every fact). 'prefix ablation' = the deletion search stops at the first fragment that cannot go together with the ones before it (the default skips that fragment and continues).

| reader, tolerance | responses (with filler) | no fact lost | precision | recall, planted fragments | recall per response with filler | repeats resolved | MAE of removable share | mean error of removable share |
|---|---|---|---|---|---|---|---|---|
| main, τ = 0.5 nats | 96 (72) | 1.000 | 1.000 | 0.979 [0.953, 1.000] | 0.985 | 0.808 | 0.015 | -0.007 |
| main, τ = 2 nats | 96 (72) | 1.000 | 1.000 | 0.995 [0.984, 1.000] | 0.997 | 0.846 | 0.006 | 0.003 |
| main, τ = 0.5 nats, prefix ablation | 96 (72) | 1.000 | 1.000 | 0.963 [0.934, 0.985] | 0.974 | 0.808 | 0.018 | -0.010 |

Deleted / planted by type (τ = 0.5): main: closing 22/23, distractor 9/9, elaboration 21/21, enumeration 20/21, hedge 30/31, preamble 21/22, restatement 25/25, unrequested 38/38.

Planted filler left in (τ = 0.5):
- main: of 4 planted filler fragments left in, 100% cost more than the tolerance to delete even alone (without them the reader's answer probabilities move by more than τ); 0% could go alone but not together with the fragments deleted before them.

| reader check | AUC present vs absent | gold unreadable | wrong-statement detection | nats per fact |
|---|---|---|---|---|
| main | 0.977 | 0.000 | 1.000 | 13.811 |

Baselines on the same simulated responses. Every method on the same responses: the filler map of each reader, the LLM judge and Selective Context when they were run (drybench baselines), and three cheap metrics. Fragment level, removable (filler or wrong statement) vs needed: F1, Cohen's κ, and the AUC of the method's score (the filler map: minus the fragment's leave-one-out cost, or its yes/no decision where no cost is stored; the judge's p(filler) + p(wrong); Selective Context: lower self-information = more removable). Response level, removable share by characters (95% CIs: bootstrap over items): pairs ranked right = over pairs whose true share differs by ≥ 0.2, the share where the method scores the response with more filler higher (ties count half; 0.5 = chance), over all pairs and over pairs of responses to the same item; Spearman with the true share; 'shorter but more removable' = such pairs where the more removable response is the shorter one, share the method ranks higher (ties apart; length: 0 by construction); partial Spearman with the true share given length. Selective Context deletes, least informative fragment first, as many characters as the first reader's filler map (equal budget); the judge's share is its expected removable share. Planted responses: the true share is the planted filler and wrong statements (a summary that repeats the facts is planted filler); a fact stated twice may go either way, so such fragments are left out of the fragment-level scores. First table: the responses every method scores; second: also the answers from which a reader reads none of the requested facts — the filler map's verdict on such an answer is that it delivers nothing, so all of it counts as removable (as in the results tables: a failed answer); answers whose gold the reader cannot read stay out.

96 of 96 responses scored by every method (filler map (reader main) 96); 347 fragments with a truth label; 288 pairs.

| method | fragments: F1 | fragments: κ | fragments: AUC | pairs ranked right [95% CI] | same-item pairs ranked right [95% CI] | responses: Spearman | Spearman [95% CI] | shorter but more removable: ranked right | … tied | partial Spearman given length |
|---|---|---|---|---|---|---|---|---|---|---|
| filler map (reader main) | 0.989 | 0.977 | 1.000 | 0.99 [0.98, 1.00] | 1.00 [1.00, 1.00] | 0.954 | 0.95 [0.91, 0.99] | 0.951 | 0.000 | +0.93 [+0.85, +0.99] |
| length (characters) |  |  |  | 0.88 [0.84, 0.92] | 0.99 [0.95, 1.00] | 0.630 | 0.63 [0.56, 0.71] | 0.000 | 0.000 | n/a |
| Yap excess (characters) |  |  |  | 0.94 [0.91, 0.96] | 0.99 [0.95, 1.00] | 0.747 | 0.75 [0.69, 0.81] | 0.469 | 0.003 | +0.59 [+0.46, +0.71] |
| stop-word share |  |  |  | 0.51 [0.42, 0.61] | 0.36 [0.18, 0.57] | 0.067 | 0.07 [-0.10, 0.25] | 0.625 | 0.014 | +0.14 [-0.09, +0.34] |

## T2. Results per model × prompt — reader main

Items without a missing fact (those are reported under 'Missing-fact items'). removable = share of tokens the filler map deletes = filler (uninformative + repeated facts) + wrong statements; facts, κ and complete answers count every response the reader can judge, a response that delivers nothing included; complete answers = share of responses from which the reader reads every requested fact, and removable over those only — a failed answer makes almost everything removable; overhead = tokens per delivered bit vs the minimal correct answer (geometric mean) = ×removable × ×wordiness.

| model \| prompt | n | tokens | removable | filler | wrong | facts | complete answers | removable, complete answers | κ | overhead | ×removable | ×wordiness | cut at the token limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 260 | 28.342 | 0.058 | 0.033 | 0.025 | 0.766 | 0.546 | 0.009 | 0.781 | 1.224 | 1.084 | 1.129 | 0.000 |
| qwen2.5-0.5b \| default | 260 | 35.269 | 0.067 | 0.039 | 0.028 | 0.737 | 0.514 | 0.040 | 0.746 | 1.824 | 1.099 | 1.660 | 0.000 |
| qwen2.5-1.5b \| concise | 260 | 34.150 | 0.047 | 0.043 | 0.004 | 0.908 | 0.851 | 0.041 | 0.905 | 1.370 | 1.059 | 1.293 | 0.000 |
| qwen2.5-1.5b \| default | 260 | 38.785 | 0.044 | 0.044 | 0.000 | 0.927 | 0.876 | 0.048 | 0.921 | 1.651 | 1.058 | 1.560 | 0.000 |
| smollm2-1.7b \| concise | 260 | 32.554 | 0.057 | 0.034 | 0.023 | 0.854 | 0.699 | 0.023 | 0.844 | 1.530 | 1.083 | 1.413 | 0.000 |
| smollm2-1.7b \| default | 260 | 40.842 | 0.041 | 0.029 | 0.012 | 0.837 | 0.691 | 0.016 | 0.841 | 1.816 | 1.060 | 1.713 | 0.000 |

## T3. 'Be concise.' vs default, same items (concise − default, 95% CI) — reader main

| model | Δtokens | Δremovable | Δfiller | Δwrong | Δfacts | overhead | pairs with both maps |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | -6.927 [-8.389, -5.484] | -0.010 [-0.030, +0.011] | -0.003 [-0.017, +0.012] | -0.007 [-0.022, +0.008] | +0.029 [-0.013, +0.071] | ×0.70 [0.65, 0.74] | 230 of 260 |
| qwen2.5-1.5b | -4.635 [-5.973, -3.515] | +0.003 [-0.010, +0.016] | -0.001 [-0.012, +0.011] | +0.004 [+0.000, +0.009] | -0.019 [-0.042, +0.003] | ×0.83 [0.78, 0.87] | 249 of 260 |
| smollm2-1.7b | -8.288 [-10.058, -6.558] | +0.015 [+0.002, +0.029] | +0.006 [-0.006, +0.018] | +0.009 [+0.002, +0.017] | +0.016 [-0.014, +0.047] | ×0.83 [0.80, 0.87] | 237 of 260 |

## T5. Removable share by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.032 | 0.065 | 0.000 | 0.000 | 0.416 |
| qwen2.5-0.5b \| default | 0.000 | 0.033 | 0.085 | 0.000 | 0.000 | 0.406 |
| qwen2.5-1.5b \| concise | 0.000 | 0.020 | 0.070 | 0.000 | 0.000 | 0.231 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.057 | 0.000 | 0.000 | 0.326 |
| smollm2-1.7b \| concise | 0.000 | 0.020 | 0.061 | 0.037 | 0.000 | 0.412 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.046 | 0.060 | 0.036 | 0.300 |

## T5. Filler by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.010 | 0.012 | 0.000 | 0.000 | 0.416 |
| qwen2.5-0.5b \| default | 0.000 | 0.006 | 0.027 | 0.000 | 0.000 | 0.393 |
| qwen2.5-1.5b \| concise | 0.000 | 0.009 | 0.068 | 0.000 | 0.000 | 0.231 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.056 | 0.000 | 0.000 | 0.326 |
| smollm2-1.7b \| concise | 0.000 | 0.006 | 0.018 | 0.037 | 0.000 | 0.346 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.017 | 0.060 | 0.036 | 0.255 |

## T5. Wrong statements by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.000 | 0.022 | 0.054 | 0.000 | 0.000 | 0.000 |
| qwen2.5-0.5b \| default | 0.000 | 0.027 | 0.059 | 0.000 | 0.000 | 0.013 |
| qwen2.5-1.5b \| concise | 0.000 | 0.011 | 0.001 | 0.000 | 0.000 | 0.000 |
| qwen2.5-1.5b \| default | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |
| smollm2-1.7b \| concise | 0.000 | 0.013 | 0.043 | 0.000 | 0.000 | 0.065 |
| smollm2-1.7b \| default | 0.000 | 0.000 | 0.028 | 0.000 | 0.000 | 0.044 |

## T5. Facts delivered by item design — reader main

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.897 | 0.852 | 0.780 | 0.409 | 0.550 | 0.525 |
| qwen2.5-0.5b \| default | 1.000 | 0.753 | 0.718 | 0.500 | 0.400 | 0.700 |
| qwen2.5-1.5b \| concise | 0.974 | 0.962 | 0.971 | 0.591 | 0.450 | 0.950 |
| qwen2.5-1.5b \| default | 0.949 | 0.964 | 0.978 | 0.500 | 0.700 | 1.000 |
| smollm2-1.7b \| concise | 0.974 | 0.909 | 0.896 | 0.500 | 0.650 | 0.625 |
| smollm2-1.7b \| default | 0.949 | 0.914 | 0.898 | 0.318 | 0.550 | 0.650 |

## T5. Answers that deliver nothing by item design — reader main

Share of answers from which the reader reads less than 1 nat (a failed answer): they have no filler map, so the removable, filler and wrong tables above leave them out; facts counts them as failed.

| model \| prompt | k=1 | k=2-3 | k=4-6 | ambiguous | 2-hop | 2-hop+explain |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.075 | 0.025 | 0.000 | 0.100 | 0.350 | 0.250 |
| qwen2.5-0.5b \| default | 0.000 | 0.037 | 0.037 | 0.000 | 0.600 | 0.150 |
| qwen2.5-1.5b \| concise | 0.000 | 0.000 | 0.000 | 0.050 | 0.400 | 0.000 |
| qwen2.5-1.5b \| default | 0.025 | 0.000 | 0.000 | 0.100 | 0.250 | 0.000 |
| smollm2-1.7b \| concise | 0.025 | 0.000 | 0.000 | 0.150 | 0.350 | 0.150 |
| smollm2-1.7b \| default | 0.050 | 0.000 | 0.000 | 0.050 | 0.450 | 0.200 |

## T5. Missing-fact items: does the answer say the fact is not given?

Items whose source does not give one of the requested facts; the right answer says so. Whether an answer does is judged by a lexical rule (an abstention phrase — 'not stated / specified / mentioned / provided', 'unknown', 'no information', 'cannot determine', … — about the asked fact; drybench.abstain), not by the reader: the reader's reading of 'not stated' does not separate answers that say it from answers that make a value up (AUC below, from the stored scores). These items are left out of every other table.

| model \| prompt | missing fact, k=1 | missing fact, k=3 | all | n |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.050 | 0.000 | 0.025 | 40 |
| qwen2.5-0.5b \| default | 0.050 | 0.000 | 0.025 | 40 |
| qwen2.5-1.5b \| concise | 0.100 | 0.100 | 0.100 | 40 |
| qwen2.5-1.5b \| default | 0.250 | 0.250 | 0.250 | 40 |
| smollm2-1.7b \| concise | 0.150 | 0.100 | 0.125 | 40 |
| smollm2-1.7b \| default | 0.300 | 0.150 | 0.225 | 40 |

The reader's reading of 'not stated' on the same answers (its accepted reply to the missing fact's question):

- reader main: AUC 0.50; read as 'not stated' (p ≥ 0.5): 0.23 of the answers that say it, 0.18 of the others

## T6. Does the generating model's span entropy flag filler? — reader main

AUC of span entropy for filler vs kept spans (0.5 = no signal); span length as a baseline. 'copied' = spans whose content words (3 or more, 80% of them) occur in one sentence of the document; 'own words' = the other spans with 3+ content words; the AUC is computed within each (filler vs kept), and 'filler copied from the document' is the copied share of such filler spans.

| model \| prompt | filler spans | AUC span entropy | AUC within a response | AUC span length | filler copied from the document | AUC, copied | AUC, own words |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 22 | 0.827 | 0.829 | 0.624 | 0.182 | 0.467 | 0.888 |
| qwen2.5-0.5b \| default | 31 | 0.746 | 0.754 | 0.581 | 0.300 | 0.630 | 0.793 |
| qwen2.5-1.5b \| concise | 49 | 0.725 | 0.759 | 0.546 | 0.267 | 0.500 | 0.697 |
| qwen2.5-1.5b \| default | 56 | 0.837 | 0.853 | 0.447 | 0.107 | 0.571 | 0.960 |
| smollm2-1.7b \| concise | 25 | 0.915 | 0.864 | 0.764 | 0.111 | 0.943 | 0.914 |
| smollm2-1.7b \| default | 22 | 0.931 | 0.896 | 0.597 | 0.238 | 0.853 | 0.969 |

Filler tokens per entropy cell relative to the cell's share of all tokens (> 1: over-represented):

| model \| prompt | autopilot | over_elaboration | template_hedge | compensation |
|---|---|---|---|---|
| qwen2.5-0.5b \| concise | 0.28 | 1.92 | 0.16 | 1.41 |
| qwen2.5-0.5b \| default | 0.41 | 2.04 | 0.28 | 1.43 |
| qwen2.5-1.5b \| concise | 0.64 | 1.09 | 0.69 | 1.38 |
| qwen2.5-1.5b \| default | 0.11 | 2.46 | 0.18 | 1.19 |
| smollm2-1.7b \| concise | 0.00 | 1.20 | 0.10 | 1.90 |
| smollm2-1.7b \| default | 0.00 | 1.31 | 0.00 | 2.70 |

Profiles (item bootstrap 95% CI):

| model \| prompt | β_ctrl (filler vs uncertainty, same design) | γ (substance tokens per fact) | γ_filler (filler tokens per fact) |
|---|---|---|---|
| qwen2.5-0.5b \| concise | -0.001 [-0.035, +0.030] | +7.04 [+6.06, +7.93] | +0.26 [+0.03, +0.60] |
| qwen2.5-0.5b \| default | -0.010 [-0.053, +0.034] | +7.14 [+6.21, +8.19] | +0.56 [+0.17, +0.95] |
| qwen2.5-1.5b \| concise | +0.057 [+0.016, +0.104] | +7.13 [+6.54, +7.75] | +1.33 [+1.02, +1.62] |
| qwen2.5-1.5b \| default | -0.016 [-0.056, +0.018] | +7.49 [+6.97, +8.08] | +1.19 [+0.89, +1.47] |
| smollm2-1.7b \| concise | +0.014 [-0.020, +0.046] | +5.79 [+5.09, +6.71] | +0.23 [+0.07, +0.40] |
| smollm2-1.7b \| default | -0.012 [-0.058, +0.033] | +10.16 [+9.35, +11.16] | +0.19 [+0.03, +0.41] |

## T7. Tolerance: removable share at 0.5 vs 2 nats

| reader — model \| prompt | removable, 0.5 nats | removable, 2 nats |
|---|---|---|
| main — qwen2.5-0.5b \| concise | 0.058 | 0.062 |
| main — qwen2.5-0.5b \| default | 0.067 | 0.070 |
| main — qwen2.5-1.5b \| concise | 0.047 | 0.049 |
| main — qwen2.5-1.5b \| default | 0.044 | 0.045 |
| main — smollm2-1.7b \| concise | 0.057 | 0.060 |
| main — smollm2-1.7b \| default | 0.041 | 0.048 |

## Examples of deleted spans (default prompt) — reader main

- **uninformative** [qwen2.5-1.5b, c08-0012] request: For shipment NN-6432, what were the departure date, the carrier, the origin port, the cause of the d… — span: «For shipment NN-6432:»
- **uninformative** [smollm2-1.7b, c14-0011] request: What is the home port of the carrier that handled shipment RF-3147? Explain how you found it. — span: «This information was found in the operations report, which states: "Shipment RF-3147 left on 15 May.»
- **uninformative** [smollm2-1.7b, c09-0019] request: For shipment PZ-5047, what were the inspector, the cause of the delay, the carrier, the number of co… — span: «Shipment PZ-5047:»
- **uninformative** [smollm2-1.7b, c14-0002] request: What is the home port of the carrier that handled shipment DT-1126? Explain how you found it. — span: «This information was found by analyzing the shipment details provided in the report.»
- **duplicate** [qwen2.5-1.5b, c14-0012] request: What is the home port of the carrier that handled shipment KG-4249? Explain how you found it. — span: «The report also mentions that "Dunyorvi Lines is based in Port Vithive."»
- **duplicate** [qwen2.5-1.5b, c14-0004] request: What is the home port of the carrier that handled shipment VV-5374? Explain how you found it. — span: «Roseskros Freight is explicitly identified as being based in Rosnal Harbor.»
- **duplicate** [qwen2.5-1.5b, c14-0006] request: What is the home port of the carrier that handled shipment JK-1879? Explain how you found it. — span: «By connecting these two pieces of information, we can conclude that the carrier (Pelturel Freight) that handled shipment JK-1879 has its home port in Port Kaobdra.»
- **duplicate** [qwen2.5-0.5b, c14-0000] request: What is the home port of the carrier that handled shipment MR-2985? Explain how you found it. — span: «The shipment MR-2985 departed from Port Eskta, making it the carrier's home port.»
- **misleading** [smollm2-1.7b, c06-0018] request: For shipment VP-2278, what were the inspector, the number of containers, the cause of the delay and … — span: «The origin port for shipment VP-2278 was Vanbexdra Docks.»
- **misleading** [qwen2.5-0.5b, c09-0007] request: For shipment GZ-3874, what were the carrier, the departure date, the cargo, the destination port, th… — span: «The carrier for shipment GZ-3874 was Uldun Rostu.»
- **misleading** [qwen2.5-0.5b, c07-0009] request: For shipment PB-3476, what were the cargo, the cause of the delay, the number of containers and the … — span: «The origin port for shipment PB-3476 was Port Marobros.»
- **misleading** [qwen2.5-0.5b, c07-0001] request: For shipment XN-3656, what were the inspector, the destination port, the cargo and the number of con… — span: «The number of containers for shipment XN-3656 was 39.»

## T8. Cost per response

Wall-clock seconds per response (logged from v1.0). The filler map: the whole score of a model's answer (deletion search, kinds of deleted fragments, the gold answer); its reader never reads the source text. The LLM judge: one pass per fragment over the source text, the request and the response. Selective Context: one pass over the request and the response. The map is timed on the run's answers, the baselines on the checks' responses; `drybench annotation-report` times all of them on the same responses.

| method — GPU | responses timed | seconds per response (median) | seconds per response (mean) |
|---|---|---|---|
| filler map, reader main — Tesla T4 | 1800 | 0.287 | 0.950 |
