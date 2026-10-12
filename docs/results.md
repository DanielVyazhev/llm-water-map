# Results of the development runs

Source: [results/dev_v1.0](../results/dev_v1.0/README.md), tables `tables_synthetic.md` (T) and `tables_real.md`
(R). All design decisions were made on these runs; the confirmation run ([PROTOCOL.md](PROTOCOL.md)) has not been
made.

**Status.** Scored by v1.0.0 (scoring version 8 for the synthetic run, 7 for the real-text run; the two differ only
on missing-fact items). On real texts v1.0.0 did not credit part of the correct answers
([below](#answer-matching-in-v100)); v1.1.0 fixes it, the runs are to be rescored. Synthetic results are barely
affected: models copy invented names verbatim.

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
| Planted wrong statements: deleted / labelled wrong | — | 0.94 / 0.65 (0.73 / 0.57) |
| Mean absolute error of the share | 0.036 (0.013) | 0.154 (0.119) |

Real texts by suite, recall: CUAD 0.84, MuSiQue 0.64, Poddar et al. 0.99, SQuAD 0.80. Of the planted filler left
in, 91% costs more than the tolerance to delete even alone: the reader uses it. On the 150 responses whose facts
the reader all reads at p ≥ 0.5: no fact lost 1.00, precision 1.00, recall 0.93.

**Human labels, Poddar et al.** (R4; 125 of 189 responses scored by the main reader, 131 by Phi):

| | Main | Phi-3.5-mini |
|---|---|---|
| Precision / recall | 0.94 / 0.81 | 0.95 / 0.83 |
| F1 / Cohen's κ | 0.87 / 0.62 | 0.89 / 0.65 |
| Minimal answer kept | 0.93 | 0.95 |
| Ranking of the LLMs vs people, Kendall τ | 0.69 (10 LLMs) | 0.83 (9 LLMs) |

Share deleted by human label (main): minimal answer 0.12, additional information 0.86, explanation 0.85,
conversational 0.83, irrelevant 0.70.

**Baselines** (T1, R5). The LLM judge and Selective Context were not run on the development runs.

| | Synthetic, planted | Real, planted | Human labels |
|---|---|---|---|
| Responses | 96 | 259 | 111 |
| Spearman with the true share: filler map / length / Yap / stop words | 0.95 / 0.63 / 0.75 / 0.07 | 0.77 / 0.04 / 0.14 / 0.39 | 0.70 / 0.75 / 0.76 / 0.15 |
| Pairs "shorter but more removable", ranked right: filler map / length / Yap / stop words | 0.96 / 0 / 0.47 / 0.63 | 0.83 / 0 / 0.12 / 0.76 | 0.34 / 0 / 0.09 / 0.49 |
| Partial Spearman given length, filler map | +0.93 [0.86, 0.98] | +0.77 [0.69, 0.83] | +0.25 [0.00, 0.53] |

Planted responses by dataset, Spearman with the true share (`scripts/readme_figure.py`, the chart in the README):

| Dataset | Responses | Filler map | Length | Yap excess | Stop-word share |
|---|---|---|---|---|---|
| Synthetic reports | 96 | 0.95 | 0.63 | 0.75 | 0.07 |
| SQuAD 2.0 | 66 | 0.65 | 0.03 | 0.06 | 0.35 |
| MuSiQue | 89 | 0.79 | 0.18 | 0.41 | 0.51 |
| CUAD | 68 | 0.70 | −0.05 | −0.02 | 0.22 |
| Poddar et al. | 36 | 0.66 | 0.79 | 0.77 | 0.42 |

On human labels the result depends on the responses from which the reader reads nothing:

| Human-labelled responses | Spearman: filler map / length | Pairs ranked right (ties) | Partial Spearman given length |
|---|---|---|---|
| without them (111) | 0.70 / 0.75 | 0.34 (0.22) | +0.25 [0.00, 0.53] |
| with them, all removable (168) | 0.71 / 0.60 | 0.63 (0.13) | +0.53 [0.33, 0.72] |
| only with a fragment labelled minimal answer (102) | 0.91 / 0.85 | 0.81 (0.02) | +0.71 [0.48, 0.88] |

**Second reader** (T4 and per suite):

| | Synthetic | SQuAD 2.0 | MuSiQue | CUAD | Poddar et al. |
|---|---|---|---|---|---|
| Spearman, removable share per response | 0.91 | 0.74 | 0.71 | 0.66 | 1.00 |
| Kendall τ, ranking of model × prompt | 1.00 | −0.73 | 0.60 | 0.73 | 0.87 |

Synthetic, per fragment (4341): same decision 95.0%, Cohen's κ 0.84; filler or not κ 0.71; wrong statement or not
κ 0.61.

## Models

Synthetic reports, main reader (T2):

| Model, prompt | Tokens | Removable | Filler | Wrong | Facts | Complete answers | Overhead = ×removable × ×wordiness |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, default | 58 | 0.27 | 0.21 | 0.06 | 0.70 | 0.47 | 2.58 = 1.54 × 1.68 |
| Qwen2.5-0.5B, concise | 27 | 0.11 | 0.06 | 0.05 | 0.72 | 0.47 | 1.21 = 1.15 × 1.05 |
| Qwen2.5-1.5B, default | 40 | 0.14 | 0.13 | 0.00 | 0.90 | 0.80 | 1.54 = 1.21 × 1.28 |
| Qwen2.5-1.5B, concise | 34 | 0.07 | 0.06 | 0.01 | 0.91 | 0.82 | 1.32 = 1.09 × 1.22 |
| SmolLM2-1.7B, default | 41 | 0.04 | 0.03 | 0.01 | 0.84 | 0.69 | 1.77 = 1.05 × 1.68 |
| SmolLM2-1.7B, concise | 33 | 0.06 | 0.03 | 0.02 | 0.85 | 0.70 | 1.48 = 1.08 × 1.38 |

Real texts, main reader (R6):

| Model, prompt | Removable: SQuAD / MuSiQue / CUAD / Poddar | Facts: SQuAD / MuSiQue / CUAD / Poddar |
|---|---|---|
| Qwen2.5-0.5B, default | 0.14 / 0.41 / 0.08 / 0.10 | 0.60 / 0.16 / 0.69 / 0.31 |
| Qwen2.5-0.5B, concise | 0.16 / 0.31 / 0.10 / 0.03 | 0.60 / 0.16 / 0.69 / 0.41 |
| Qwen2.5-1.5B, default | 0.19 / 0.46 / 0.02 / 0.11 | 0.67 / 0.29 / 0.73 / 0.36 |
| Qwen2.5-1.5B, concise | 0.18 / 0.37 / 0.02 / 0.04 | 0.69 / 0.27 / 0.77 / 0.41 |
| SmolLM2-1.7B, default | 0.15 / 0.35 / 0.08 / 0.14 | 0.59 / 0.33 / 0.68 / 0.55 |
| SmolLM2-1.7B, concise | 0.17 / 0.23 / 0.07 / 0.04 | 0.66 / 0.30 / 0.67 / 0.55 |

Facts on real texts are understated by v1.0.0 answer matching. The model ranking by removable share does not carry
over between suites (Kendall τ from −1.00 to +0.33, 3 models).

**"Be concise."** (paired, same items, concise − default):
- Synthetic: Qwen2.5-0.5B removable −0.15 [−0.19, −0.11], overhead ×0.51; Qwen2.5-1.5B −0.07, ×0.86; SmolLM2
  +0.02 [+0.00, +0.03], ×0.83: shorter wording, the same share of removable fragments.
- MuSiQue: removable −0.10, −0.08, −0.10 (CI excludes 0 for SmolLM2 only); overhead ×0.73, ×0.62, ×0.63.
- SQuAD, CUAD: removable share within ±0.03; overhead ×0.77–0.94.
- No significant drop in facts.

**Item design** (synthetic, T5): the removable share is largest for 2-hop requests with "Explain how you found it."
(0.25–0.66) and ambiguous requests (Qwen2.5-0.5B default 0.56); near 0 for one-fact requests except Qwen2.5-0.5B
default (0.19). Wrong statements are most frequent with 4–6 facts (up to 0.10).

**Missing facts.** Hand labels of all 600 development answers to missing-fact items: synthetic, 20% say the fact is
not given, 67% give a value, 14% answer something else; real texts, 11% say it is not given. The lexical rule
agrees with these labels (precision 0.99, recall 1.00; tuned on them).

**Entropy** (T6 and per suite). The generating model's span entropy separates filler from kept fragments with AUC
0.69–0.92 on synthetic reports, above span length in every group. On real texts: SQuAD 0.45–0.75, MuSiQue 0.47–0.56,
CUAD and Poddar too few filler fragments. Filler does not follow answer uncertainty once the design is fixed
(β_ctrl from −0.09 to −0.01).

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

- Human labels: the advantage over length is clear only when the responses that deliver nothing are counted. The
  annotation of the confirmation run's answers decides it.
- The second reader reverses the SQuAD ranking of model × prompt; the wrong-statement label is less stable between
  readers than the removable share.
- Models up to 1.7B and readers of 3–4B so far; the confirmation run adds models up to 14B and Phi-4 as a reader.
- Scope: answers with checkable facts (QA, retrieval, extraction). Reasoning, open-ended explanations and code are
  not covered.
