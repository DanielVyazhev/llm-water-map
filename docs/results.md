# Results of the development runs

Source: [results/dev_v1.0](../results/dev_v1.0/README.md), tables `tables_synthetic.md` (T) and `tables_real.md`
(R). All design decisions were made on these runs; the confirmation run ([PROTOCOL.md](PROTOCOL.md)) has not been
made.

**Status.** Scored by v1.0.0 (scoring version 8 for the synthetic run, 7 for the real-text run; the two differ only
on missing-fact items). On real texts v1.0.0 did not credit part of the correct answers
([below](#answer-matching-in-v100)); v1.1.0 fixes it, the runs are to be rescored. Synthetic results should change
little (models copy invented names verbatim); not checked yet.

| | Synthetic | Real texts |
|---|---|---|
| Items | 300: 15 cells × 20 | 312: 21 cells × 15 (MS MARCO 12) |
| Models | Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B (Instruct) | same |
| Prompts | default, "Be concise." | same |
| Responses | 1800 | 1872 |
| Readers | Qwen2.5-3B-Instruct (main), Phi-3.5-mini-instruct | same |
| GPU | Colab T4 | Colab T4 |

## Checks

Main reader; Phi-3.5-mini in brackets.

**Planted filler** (T1, R3):

| | Synthetic | Real texts |
|---|---|---|
| Responses (with filler) | 96 (72) | 268 (195) |
| No fact lost | 1.00 (1.00) | 0.97 (0.95) |
| Precision | 1.00 (1.00) | 0.99 (0.98) |
| Recall of planted filler [95% CI] | 0.95 [0.90, 0.99] (1.00) | 0.78 [0.72, 0.84] (0.83) |
| Repeats resolved | 0.85 (0.92) | 0.69 (0.75) |
| Planted wrong statements: deleted / labelled wrong | — | 0.93 / 0.65 (0.73 / 0.57) |
| Mean absolute error of the share | 0.036 (0.013) | 0.154 (0.119) |

Real texts by suite, recall: CUAD 0.84, MuSiQue 0.64, Poddar et al. 0.99, SQuAD 0.79. Of the planted filler left
in, 91% costs more than the tolerance to delete even alone: without it the reader's answer probabilities move by
more than τ. On the 150 responses whose facts the reader all reads at p ≥ 0.5: no fact lost 1.00, precision 1.00,
recall 0.93.

The truth of R3 counts a fact as kept while any kept fragment mentions its answer. Where that fragment only
mentions it ("VerticalNet, Inc., a Pennsylvania corporation" for the governing law), the fact is lost. Counting a
fact as kept only by the sentence chosen to state it (and the summary): no fact lost 0.90 (Phi-3.5-mini 0.84),
precision 0.95 (0.91); [`scripts/truth_variants.py`](../scripts/truth_variants.py). Part of the 20 (36) responses
that change are true restatements, so the value lies between.

**Human labels, Poddar et al.** (R4; 125 of 189 responses scored by the main reader, 131 by Phi):

| | Main | Phi-3.5-mini |
|---|---|---|
| Precision / recall | 0.94 / 0.81 | 0.95 / 0.83 |
| F1 / Cohen's κ | 0.87 / 0.62 | 0.89 / 0.65 |
| Minimal answer kept | 0.93 | 0.95 |
| Ranking of the LLMs vs people, Kendall τ | 0.69 (10 LLMs) | 0.83 (9 LLMs) |

The fact of a response is its own answer. For 41 of the 125 (Phi: 35 of 131) it is what the reader reads from the
fragments people labelled minimal answer, so these labels enter the method. Responses whose fact is the reference
answer (84; Phi 95): F1 0.83, κ 0.58 (Phi 0.88, 0.68).

Share deleted by human label (main): minimal answer 0.12, additional information 0.86, explanation 0.85,
conversational 0.83, irrelevant 0.70.

**Baselines** (T1, R5). The LLM judge and Selective Context were not run on the development runs.

| | Synthetic, planted | Real, planted | Human labels |
|---|---|---|---|
| Responses | 96 | 259 | 111 |
| Spearman with the true share: filler map / length / Yap / stop words | 0.95 / 0.63 / 0.75 / 0.07 | 0.77 / 0.04 / 0.14 / 0.39 | 0.70 / 0.75 / 0.75 / 0.15 |
| Pairs "shorter but more removable", ranked right: filler map / length (0 by construction) / Yap / stop words | 0.96 / 0 / 0.47 / 0.63 | 0.83 / 0 / 0.12 / 0.76 | 0.34 / 0 / 0.09 / 0.49 |
| Partial Spearman given length, filler map | +0.93 [0.86, 0.98] | +0.77 [0.69, 0.83] | +0.25 [0.00, 0.53] |

By dataset (`scripts/readme_figure.py`, the chart in the README). Pairwise accuracy: over pairs of responses whose
true shares differ by ≥ 0.2, the share where the metric scores the response with more filler higher (ties count
half; 50% = chance). Fragment ROC-AUC: the filler map's yes/no decisions per fragment, filler or wrong vs needed
(equal to balanced accuracy). Responses from which the reader reads no fact are left out (real texts: 39 of 298;
counted as all removable, the filler map scores 81% on SQuAD 2.0, 83% on MuSiQue and CUAD, 74% on Poddar et al.).
Over pairs of responses to the same item (28–87 pairs per dataset) length does about as well as the filler map:
synthetic 99% vs 100%, SQuAD 2.0 95% vs 92%, MuSiQue 89% vs 86%, CUAD 83% vs 90%, Poddar et al. 100% vs 89%.

| Dataset | Responses | Pairs | Filler map | Length | Yap excess | Stop-word share | Filler map, fragment ROC-AUC |
|---|---|---|---|---|---|---|---|
| Synthetic reports, planted | 96 | 2380 | 99% | 88% | 94% | 51% | 0.98 |
| SQuAD 2.0, planted | 66 | 1141 | 86% | 51% | 53% | 68% | 0.86 |
| MuSiQue, planted | 89 | 2355 | 89% | 58% | 69% | 76% | 0.79 |
| CUAD, planted | 68 | 1467 | 84% | 46% | 47% | 60% | 0.88 |
| Poddar et al., planted | 36 | 349 | 84% | 95% | 94% | 69% | 0.99 |
| Poddar et al., human labels | 111 | 3639 | 88% | 92% | 92% | 58% | 0.84 |

On human labels the result depends on the responses from which the reader reads nothing:

| Human-labelled responses | Spearman: filler map / length | Pairs ranked right (ties) | Partial Spearman given length |
|---|---|---|---|
| without them (111) | 0.70 / 0.75 | 0.34 (0.22) | +0.25 [0.00, 0.53] |
| with them, all removable (168) | 0.71 / 0.60 | 0.63 (0.13) | +0.53 [0.33, 0.72] |
| with them, without answer source "none" (131) | 0.66 / 0.74 | 0.39 (0.19) | +0.19 [−0.03, 0.43] |
| only with a fragment labelled minimal answer (102) | 0.91 / 0.85 | 0.81 (0.02) | +0.71 [0.48, 0.88] |

The second row is not evidence for the filler map. Whether the reader gets a response's own answer depends on the
fragments people labelled minimal answer. A response without the reference answer and without such an answer
(answer source "none") delivers nothing and counts as all removable; most of these have no fragment labelled
minimal answer, so their true share is 1 by the same labels. Without them (third row) the advantage is gone. The
last row selects responses by the truth labels. Third row: [`scripts/truth_variants.py`](../scripts/truth_variants.py).

**Second reader** (T4 and per suite):

| | Synthetic | SQuAD 2.0 | MuSiQue | CUAD | Poddar et al. |
|---|---|---|---|---|---|
| Spearman, removable share per response | 0.90 | 0.74 | 0.71 | 0.66 | 1.00 |
| Kendall τ, ranking of model × prompt | 1.00 | −0.73 | 0.60 | 0.73 | 0.87 |

Synthetic, per fragment (4341): same decision 95.0%, Cohen's κ 0.84; filler or not κ 0.71; wrong statement or not
κ 0.61.

## Models

Synthetic reports, main reader (T2):

| Model, prompt | Tokens | Removable | Filler | Wrong | Facts | Complete answers | Overhead = ×removable × ×wordiness |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, default | 58 | 0.27 | 0.21 | 0.06 | 0.70 | 0.47 | 2.58 = 1.54 × 1.67 |
| Qwen2.5-0.5B, concise | 27 | 0.10 | 0.06 | 0.05 | 0.72 | 0.47 | 1.21 = 1.15 × 1.05 |
| Qwen2.5-1.5B, default | 40 | 0.14 | 0.13 | 0.00 | 0.90 | 0.80 | 1.54 = 1.21 × 1.28 |
| Qwen2.5-1.5B, concise | 34 | 0.07 | 0.06 | 0.01 | 0.91 | 0.82 | 1.32 = 1.09 × 1.22 |
| SmolLM2-1.7B, default | 41 | 0.04 | 0.03 | 0.01 | 0.83 | 0.69 | 1.76 = 1.05 × 1.68 |
| SmolLM2-1.7B, concise | 33 | 0.05 | 0.03 | 0.02 | 0.85 | 0.70 | 1.48 = 1.08 × 1.38 |

Real texts, main reader (R6):

| Model, prompt | Removable: SQuAD / MuSiQue / CUAD / Poddar | Facts: SQuAD / MuSiQue / CUAD / Poddar |
|---|---|---|
| Qwen2.5-0.5B, default | 0.14 / 0.41 / 0.08 / 0.10 | 0.60 / 0.16 / 0.69 / 0.31 |
| Qwen2.5-0.5B, concise | 0.16 / 0.31 / 0.10 / 0.03 | 0.60 / 0.16 / 0.69 / 0.40 |
| Qwen2.5-1.5B, default | 0.19 / 0.46 / 0.02 / 0.11 | 0.67 / 0.29 / 0.73 / 0.36 |
| Qwen2.5-1.5B, concise | 0.18 / 0.37 / 0.02 / 0.04 | 0.68 / 0.27 / 0.77 / 0.40 |
| SmolLM2-1.7B, default | 0.15 / 0.35 / 0.08 / 0.14 | 0.59 / 0.33 / 0.68 / 0.55 |
| SmolLM2-1.7B, concise | 0.17 / 0.23 / 0.07 / 0.04 | 0.66 / 0.30 / 0.67 / 0.55 |

Facts on real texts are understated by v1.0.0 answer matching. The model ranking by removable share does not carry
over between suites (Kendall τ from −1.00 to +0.33, 3 models).

**"Be concise."** (paired, same items, concise − default):
- Synthetic: Qwen2.5-0.5B removable −0.15 [−0.19, −0.11], overhead ×0.51; Qwen2.5-1.5B −0.07, ×0.86; SmolLM2
  +0.02 [+0.00, +0.03], ×0.83: shorter wording, slightly more removable.
- MuSiQue: removable −0.10, −0.08, −0.10 (CI excludes 0 for SmolLM2 only); overhead ×0.73, ×0.62, ×0.63.
- Poddar et al.: removable −0.08, −0.03, −0.08 (CI excludes 0 for SmolLM2 only); overhead ×0.60, ×0.84, ×0.63.
- SQuAD, CUAD: removable share within ±0.03; overhead ×0.77–0.94.
- Facts: no significant drop; up with SmolLM2 on SQuAD (+0.07) and Qwen2.5-0.5B on Poddar et al. (+0.10).
- Overhead and removable share count only the pairs where the reader reads a fact from both responses: MuSiQue
  31–44 of 90, Poddar et al. 17–27 of 42, SQuAD and CUAD 47–57 of 60.

**Item design** (synthetic, T5): the removable share is largest for 2-hop requests with "Explain how you found it."
(0.25–0.66) and ambiguous requests (Qwen2.5-0.5B default 0.56); near 0 for one-fact requests except Qwen2.5-0.5B
default (0.19). Wrong statements are most frequent with 4–6 facts (up to 0.10).

**Missing facts.** Hand labels of all 600 development answers to missing-fact items: synthetic, 20% say the fact is
not given, 67% give a value, 14% answer something else; real texts, 11% say it is not given. The lexical rule
agrees with these labels (precision 0.99, recall 1.00; tuned on them).

**Entropy** (T6 and per suite). The generating model's span entropy separates filler from kept fragments with AUC
0.69–0.92 on synthetic reports, above span length in every group. On real texts: SQuAD 0.45–0.75, MuSiQue 0.47–0.56,
CUAD and Poddar too few filler fragments. These AUCs count the fragments of responses without a filler map as
kept. On synthetic reports filler does not grow with answer uncertainty given k, missing fact, ambiguity and hops
(β_ctrl from −0.09 to −0.01; "explain" is not controlled for). On SQuAD it does in one of six groups (β_ctrl
+0.06 to +0.26; Qwen2.5-1.5B default +0.26 [+0.03, +0.44]); on MuSiQue every CI includes 0.

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

- Human labels: no advantage over length (pairwise accuracy 88% vs 92%; Spearman 0.70 vs 0.75). The advantage
  when the responses that deliver nothing are counted comes from how the response's own answer is found
  ([above](#checks)). The annotation of the confirmation run's answers decides it.
- Planted filler: a gold-string match finds the fact sentences of these responses by construction, and within one
  item length ranks responses about as well as the filler map. The planted checks show that the deletion search
  works, not that the reader beats simpler methods.
- The second reader reverses the SQuAD ranking of model × prompt; the wrong-statement label is less stable between
  readers than the removable share.
- Models up to 1.7B and readers of 3–4B so far; the confirmation run adds models up to 14B and Phi-4 as a reader.
- Scope: answers with checkable facts (QA, retrieval, extraction). Reasoning, open-ended explanations and code are
  not covered.
