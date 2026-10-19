# Results of the development runs

Source: [results/dev_v1.2](../results/dev_v1.2/README.md), runs `t4_n20_v12` (tables `tables_synthetic.md`, T) and
`real_t4_n15_v12` (`tables_real.md`, R). All design decisions were made on the development runs; the confirmation
run ([PROTOCOL.md](PROTOCOL.md)) has not been made; the development values of its tests are in
[PROTOCOL.md](PROTOCOL.md#development-values).

**Status.** Scored by package v1.2.1 (scoring version 11, which v1.2.5 keeps), tables of v1.2.5; readers and models
in fp16 on a Colab T4. The LLM judge and Selective Context have not run on these runs yet. The v1.0 runs
([results/dev_v1.0](../results/dev_v1.0/README.md)) and what changed since: [below](#changes-since-v10).

| | Synthetic | Real texts |
|---|---|---|
| Items | 300: 15 cells × 20 | 312: 21 cells × 15 (MS MARCO 12) |
| Models | Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B (Instruct) | same |
| Prompts | default, "Be concise." | same |
| Decoding | plain greedy, at most 320 new tokens | same |
| Responses | 1800 | 1872 |
| Readers | Qwen2.5-3B-Instruct (main) | Qwen2.5-3B-Instruct (main), Phi-3.5-mini-instruct |
| GPU | Colab T4 | Colab T4 |

## Checks

Main reader; Phi-3.5-mini in brackets (real texts only: the synthetic v1.2 run has no second reader).

**Planted filler** (T1, R3). A fact is stated by the sentence chosen for it and by the summary, not by a sentence
that only mentions its answer; the error of the removable share is measured by characters against the planted
filler and planted wrong statements.

| | Synthetic | Real texts |
|---|---|---|
| Responses with a filler map (with filler) | 96 (72) of 96 | 271 (196) of 303 (285 (208)) |
| No fact lost | 1.00 | 0.91 (0.86) |
| Precision | 1.00 | 0.94 (0.92) |
| Recall of planted filler [95% CI] | 0.98 [0.95, 1.00] | 0.80 [0.74, 0.86] (0.86) |
| … counting the responses without a map as not deleted | – | 0.74 (0.83) |
| Repeats resolved | 0.81 | 0.63 (0.78) |
| Planted wrong statements: deleted / labelled wrong | – | 0.91 / 0.67 (0.80 / 0.59) |
| Mean absolute error of the removable share | 0.015 | 0.105 (0.141) |

Real texts by suite (main): no fact lost CUAD 0.93, MuSiQue 0.88, Poddar et al. 1.00, SQuAD 0.87; recall 0.92,
0.69, 0.90, 0.78. Of the planted filler left in, 93% costs more than the tolerance to delete even alone. On the 156
responses whose facts the reader all reads at p ≥ 0.5: no fact lost 0.93, precision 0.95, recall 0.94.

Lost facts (`scripts/r3_losses.py`): the main reader loses 29 facts in 25 responses. In 14 (12 of them MuSiQue) it
reads less than 0.5 nats of the fact from the whole response, so the sentence stating it is not protected. In 15 a
kept fragment mentions the answer and the reader reads the fact from it; by hand, in 6 of these the kept fragment
states the fact ("later proved by Leibniz and Euler"; the agreement's title in its opening sentence), which the
strict truth does not count, and in 9 it only mentions it ("VerticalNet, Inc., a Pennsylvania corporation" for the
governing law). Phi-3.5-mini loses a fact in 40 responses (14%); not read 12, mention 29, other 3.

**Human labels, Poddar et al.** (R4). Only the responses that state the reference answer are scored (96 of 189;
the main reader scores 74, Phi-3.5-mini 79):

| | Main | Phi-3.5-mini |
|---|---|---|
| Precision / recall | 0.98 / 0.77 | 0.99 / 0.86 |
| F1 / Cohen's κ | 0.86 / 0.66 | 0.92 / 0.78 |
| Minimal answer kept | 0.97 | 0.99 |
| Ranking of the LLMs vs people, Kendall τ | 0.81 (7 LLMs) | 0.79 (8 LLMs) |

Share deleted by human label (main): minimal answer 0.03, additional information 0.88, explanation 0.83,
conversational 0.80, irrelevant 0.64.

**Baselines** (T1, R5), on the responses every method scores:

| | Synthetic, planted | Real, planted | Human labels |
|---|---|---|---|
| Responses | 96 | 266 | 71 |
| Spearman with the true share: filler map / length / Yap / stop words | 0.95 / 0.63 / 0.75 / 0.07 | 0.77 / 0.10 / 0.20 / 0.34 | 0.73 / 0.72 / 0.73 / 0.11 |
| Pairs "shorter but more removable", ranked right: filler map / length (0 by construction) / Yap / stop words | 0.95 / 0 / 0.47 / 0.63 | 0.83 / 0 / 0.13 / 0.73 | 0.26 (ties 0.41) / 0 / 0.12 / 0.50 |
| Partial Spearman given length, filler map | +0.93 [0.85, 0.99] | +0.77 [0.69, 0.85] | +0.36 [0.09, 0.68] |

By dataset (`scripts/readme_figure.py`, the chart in the README). Pairwise accuracy: over pairs of responses whose
true shares differ by ≥ 0.2, the share where the metric scores the response with more filler higher (ties count
half; 50% = chance). Fragment ROC-AUC: fragments ranked by minus their leave-one-out cost (the information lost by
deleting each alone), filler or wrong vs needed.

| Dataset | Responses | Pairs | Filler map | Length | Yap excess | Stop-word share | Filler map, fragment ROC-AUC |
|---|---|---|---|---|---|---|---|
| Synthetic reports, planted | 96 | 2380 | 99% | 88% | 94% | 51% | 1.00 |
| SQuAD 2.0, planted | 68 | 1167 | 87% | 57% | 57% | 68% | 0.95 |
| MuSiQue, planted | 89 | 2351 | 84% | 55% | 68% | 72% | 0.87 |
| CUAD, planted | 68 | 1483 | 88% | 50% | 51% | 59% | 0.96 |
| Poddar et al., planted | 41 | 505 | 95% | 90% | 91% | 62% | 1.00 |
| Poddar et al., human labels | 71 | 1416 | 87% | 91% | 92% | 57% | 0.92 |

- Filler map minus length, 95% CI (points): synthetic +6.8 to +15.2, SQuAD 2.0 +12.6 to +45.9, MuSiQue +15.7 to
  +39.4, CUAD +23.5 to +47.3, Poddar et al. planted −6.2 to +13.9, human labels −10.0 to +0.6.
- Responses from which a reader reads no fact are left out (real texts: 37 of 303). Counted as all removable, the
  filler map scores 84% on SQuAD 2.0, 79% on MuSiQue, 87% on CUAD, 90% on Poddar et al.
- Over pairs of responses to the same item (38–87 pairs per dataset) length does about as well as the filler map
  (length vs filler map): synthetic 99% vs 100%, SQuAD 2.0 92% vs 94%, MuSiQue 90% vs 85%, CUAD 85% vs 95%, Poddar et
  al. 100% vs 95%.
- Length works where the fact sentences are short (median 32 characters on synthetic reports, 94 on Poddar et al.),
  not where they are long (144–153 on SQuAD 2.0, MuSiQue, CUAD).

Human labels with the responses from which the reader reads nothing (90 responses, those all removable): Spearman
filler map 0.65, length 0.75; partial Spearman given length +0.27 [+0.06, +0.54]. Only the responses with a
fragment labelled minimal answer (65): 0.92 vs 0.83, +0.78 [+0.53, +0.96].

**Second reader** (real texts, per suite):

| | SQuAD 2.0 | MuSiQue | CUAD | Poddar et al. |
|---|---|---|---|---|
| Responses scored by both | 328 | 290 | 348 | 151 |
| Spearman, removable share per response | 0.78 | 0.70 | 0.76 | 1.00 |
| Kendall τ, ranking of model × prompt | +0.73 | +0.47 | +1.00 | +1.00 |
| Per fragment, deleted or kept: same decision, Cohen's κ | 91.5%, 0.71 | 74.5%, 0.49 | 97.9%, 0.66 | 95.5%, 0.80 |
| Per fragment, wrong statement or not: Cohen's κ | 0.53 | 0.27 | 0.37 | 0.00 |

## Models

Synthetic reports, main reader (T2):

| Model, prompt | Tokens | Removable | Filler | Wrong | Facts | Complete answers | Overhead = ×removable × ×wordiness |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, default | 35 | 0.07 | 0.04 | 0.03 | 0.74 | 0.51 | 1.82 = 1.10 × 1.66 |
| Qwen2.5-0.5B, concise | 28 | 0.06 | 0.03 | 0.03 | 0.77 | 0.55 | 1.22 = 1.08 × 1.13 |
| Qwen2.5-1.5B, default | 39 | 0.04 | 0.04 | 0.00 | 0.93 | 0.88 | 1.65 = 1.06 × 1.56 |
| Qwen2.5-1.5B, concise | 34 | 0.05 | 0.04 | 0.00 | 0.91 | 0.85 | 1.37 = 1.06 × 1.29 |
| SmolLM2-1.7B, default | 41 | 0.04 | 0.03 | 0.01 | 0.84 | 0.69 | 1.82 = 1.06 × 1.71 |
| SmolLM2-1.7B, concise | 33 | 0.06 | 0.03 | 0.02 | 0.85 | 0.70 | 1.53 = 1.08 × 1.41 |

Real texts, main reader (R6):

| Model, prompt | Removable: SQuAD / MuSiQue / CUAD / Poddar | Facts: SQuAD / MuSiQue / CUAD / Poddar |
|---|---|---|
| Qwen2.5-0.5B, default | 0.20 / 0.34 / 0.04 / 0.03 | 0.67 / 0.25 / 0.77 / 0.48 |
| Qwen2.5-0.5B, concise | 0.17 / 0.34 / 0.02 / 0.02 | 0.69 / 0.21 / 0.82 / 0.50 |
| Qwen2.5-1.5B, default | 0.10 / 0.37 / 0.00 / 0.10 | 0.84 / 0.29 / 0.86 / 0.50 |
| Qwen2.5-1.5B, concise | 0.10 / 0.31 / 0.00 / 0.06 | 0.84 / 0.31 / 0.93 / 0.52 |
| SmolLM2-1.7B, default | 0.17 / 0.35 / 0.03 / 0.12 | 0.69 / 0.34 / 0.81 / 0.64 |
| SmolLM2-1.7B, concise | 0.20 / 0.23 / 0.01 / 0.03 | 0.73 / 0.32 / 0.78 / 0.62 |

The model ranking by removable share does not carry over between suites (Kendall τ from −1.00 to +1.00, 3 models).

**"Be concise."** (paired, same items, concise − default; 95% CIs):
- Synthetic: 5–8 tokens shorter; removable share unchanged (Qwen2.5-0.5B −0.01 [−0.03, +0.01], Qwen2.5-1.5B +0.00,
  SmolLM2 +0.02 [+0.00, +0.03]); overhead ×0.70, ×0.83, ×0.83.
- MuSiQue: 15–42 tokens shorter; removable −0.06, −0.05, −0.11 (the CI excludes 0 for SmolLM2 only); overhead
  ×0.90, ×0.53, ×0.64.
- Poddar et al.: removable −0.00, −0.01, −0.08 (CI excludes 0 for SmolLM2); overhead ×0.70, ×0.96, ×0.62.
- SQuAD 2.0, CUAD: removable within ±0.03; overhead ×0.81–0.94. Facts up on CUAD for Qwen2.5-0.5B (+0.05) and
  Qwen2.5-1.5B (+0.07).
- Removable share and overhead count only the pairs with a filler map for both answers: MuSiQue 37–47 of 90,
  Poddar et al. 22–28 of 42, SQuAD and CUAD 52–59 of 60.

**Item design.** Synthetic (T5): the removable share is near 0 for one-fact, ambiguous and 2-hop requests and largest
with "Explain how you found it." (0.23–0.42); with 4–6 facts 0.05–0.09, for Qwen2.5-0.5B mostly wrong statements.
MuSiQue: 0.22–0.63 with "Explain how you found it.", 0–0.23 without. SQuAD 2.0: near 0 with one question, 0.10–0.31
with 2–4. CUAD at most 0.06. Poddar et al.: MS MARCO 0.10–0.39, NarrativeQA and TweetQA 0 (SmolLM2 default 0.09,
0.06).

**Missing facts** (lexical rule): share of answers to missing-fact items that say the fact is not given. Synthetic
2.5–25% (Qwen2.5-1.5B default 25%); SQuAD 2.0 and MuSiQue 0–7%; CUAD's missing governing law: Qwen2.5-1.5B 47%
(concise) and 100% (default), the others 0%. The rule was built on hand labels of the 600 v1.0 answers to such
items (precision 0.99, recall 1.00 there); the confirmation run labels 150 answers it has not seen.

**Entropy** (T6 and per suite). The generating model's span entropy separates filler from kept fragments with AUC
0.73–0.93 on synthetic reports, above span length (0.45–0.76) in every group; SQuAD 2.0 0.45–0.79, MuSiQue
0.47–0.64; CUAD and Poddar et al. have at most 9 filler fragments per group. Filler growing with answer uncertainty
given the item design (β_ctrl): synthetic −0.02 to +0.06 (CI above 0 for Qwen2.5-1.5B concise only); SQuAD 2.0 in 3
of 6 groups (Qwen2.5-0.5B +0.22 and +0.31, SmolLM2 default +0.19); MuSiQue every CI includes 0.

## Changes since v1.0

The v1.0 runs ([results/dev_v1.0](../results/dev_v1.0/README.md)) were scored by v1.0.0 (scoring versions 7 and 8).
What differs in v1.2:
- Scoring: a fact stated in the response's own words is credited (v1.1; [below](#answer-matching-in-v100)); of two
  statements of a fact the first stays; completeness is capped at the gold's information; abbreviations do not end a
  sentence.
- Decoding: plain greedy (v1.0 applied the models' generation settings, such as Qwen2.5's repetition penalty). The
  answers are shorter and less removable: synthetic, Qwen2.5-0.5B default, 35 tokens and 0.07 removable (v1.0: 58
  and 0.27).
- Items: SQuAD 2.0 accepts the canonical span, overlapping spans and spans two annotators chose; no accepted form of
  one question appears in another question of the request.
- Checks: R3 counts a fact as kept only by its chosen sentence and the summary (v1.0 also by any fragment that
  mentions its answer: no fact lost 0.97, precision 0.99); the error of the removable share is measured against the
  planted filler (v1.0: against the shortest set of fragments stating every fact); R4 scores only the responses that
  state the reference answer (v1.0: 125 responses, whose own answer was partly found from the human labels).

## Pilot (CPU, GPT-2 reader)

`results/pilot`: 48 synthetic items × 4 simulated responses (no filler, 2, 5, 2 with a fact left out), GPT-2 124M
as the reader. Spearman with the true share: filler map 0.79, length 0.66, Yap 0.79, stop words 0.09; pairs ranked
right: 0.88, 0, 0.55, 0.46. At τ = 0.5 nats: no fact lost 98.4%, precision 0.99, recall 0.93.

## Answer matching in v1.0.0

The reader copies the response's wording; v1.0.0 accepted only the gold's forms. Examples from the development run:

| Suite | Response (part) | Accepted forms | p |
|---|---|---|---|
| CUAD | …is the Internal Laws of the State of Delaware. | Delaware; State of Delaware; the laws of the State of Delaware; … | 0.00, deleted as wrong |
| CUAD | The title of this agreement is "Second Amendment to Co-Branding Agreement". | SECOND AMENDMENT TO CO-BRANDING AGREEMENT; … | 0.00 |
| Poddar (TweetQA) | Matthew Perry | matthew perry; Matthew perry | 0.00 |
| SQuAD 2.0 | New York-based filmmakers were drawn to Jacksonville… | filmmakers; New York–based filmmakers | 0.00 |
| SQuAD 2.0 | USC Trojans | Trojans | 0.001 |
| MuSiQue | …played the role of the future King George VI. | George VI | 0.00 |

Scale (facts scored as not delivered whose gold value occurs in the response; an upper bound):

| Suite | Not credited | Gold value in the response | Share of all facts |
|---|---|---|---|
| CUAD | 203 | 169 | 27% |
| SQuAD 2.0 | 338 | 156 | 17% |
| MuSiQue | 818 | 216 | 20% |
| Poddar et al. | 144 | 34 | 13.5% |

A manual check of 10 such cases per suite found real misses in almost all SQuAD cases, 8–9 of 10 on CUAD and
Poddar, about half on MuSiQue (there the value often stands in another context). Rough estimate of facts wrongly
not credited: 1 in 4–5 on CUAD, 1 in 6 on SQuAD, 1 in 10 on MuSiQue and Poddar.

Affected: facts, removable and wrong-statement shares on real texts; "Be concise." on real texts; the human-label
comparison may be partly affected (responses "from which the reader reads nothing"). Not affected: synthetic
reports, planted filler (fact sentences come from the source), the Poddar check (it already accepts the response's
own answer). Known misses that v1.1.0 still does not credit: a response shorter than the gold ("Euler" for
"Leonhard Euler"), number words above twenty, "the first day of July", "1.7.2020".


## Open questions

- Human labels: no advantage over length (pairwise accuracy 87% vs 91%, 95% CI of the difference −10.0 to +0.6
  points; Spearman 0.73 vs 0.72). The annotation of the confirmation run's answers decides it.
- Planted filler: a gold-string match finds the fact sentences of these responses by construction, and within one
  item length ranks responses about as well as the filler map. The planted checks show that the deletion search
  works, not that the reader beats simpler methods.
- R3 loses a fact in 9% of the responses (MuSiQue 12%): the reader does not read a multi-hop answer from the
  response, or reads a fact from a fragment that only mentions its answer. The confirmation run tests H1 on R3
  against 15% and 0.85 ([PROTOCOL.md](PROTOCOL.md#confirmatory-tests)).
- The readers agree on the removable share per response (Spearman 0.70–1.00) more than on single fragments on
  MuSiQue (κ 0.49) and on the wrong-statement label (κ 0.00–0.53).
- Models up to 1.7B and readers of 3–4B so far; the confirmation run has models of 3–14B and a 14B reader.
- Scope: answers with checkable facts (QA, retrieval, extraction). Reasoning, open-ended explanations and code are
  not covered.
