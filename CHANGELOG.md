# Changelog

A change of the scoring version makes a resumed run recompute its scores ([running](docs/running.md#rescoring)).

## 1.1.0 (scoring version 9)

- A fact the response states in its own words is credited: the response's wording joins the fact's accepted answers
  for that response, also when a deleted fragment is classified (`drybench/forms.py`). Every score is recomputed.
- `drybench score --gold-forms-only` scores as 1.0.0. Scored rows store `added_answers` and `own_wording`.
- `scripts/run_final.py` and `scripts/make_colab.py` use the package next to them, not an installed copy.

## 1.0.0 (scoring version 8)

- Prepared for the confirmation run (`scripts/run_final.py`, `colab/final_a100.ipynb`, `protocol.json`); replaced by
  1.1.0 before the run was made.
- Baselines on the same responses: LLM judge per fragment, Selective Context, length, Yap excess, stop-word share;
  F1, Cohen's κ, AUC, Spearman, pair test, partial Spearman given length.
- One set of labelling guidelines for people, chat LLMs and the judge. Samples and report for human labels.

## 0.8

- 0.8.2: missing facts judged by a lexical rule (hand labels of 600 development answers); missing-fact items left
  out of the filler tables.
- 0.8.1 (scoring 8): floor 0.1 for yes/no questions about a missing fact; partial rescoring.
- 0.8.0 (scoring 7): the question about a missing fact chosen per reader (`drybench missing-check`).

## 0.7

- 0.7.0: CUAD and Poddar et al.; planted filler in real texts (R3), human labels (R4), baselines (R5), rankings
  across suites (R6).
- 0.7.1–0.7.6: missing facts asked explicitly and read from any reply starting with "not"; a deleted true statement
  needs 80% of its words in one source sentence; the deletion search skips a failing fragment and continues;
  completeness counts a contradicted fact as 0; wordings of the missing-fact question compared.

## 0.6

- 0.6.0: SQuAD 2.0 and MuSiQue; leakage filter.
- 0.6.1–0.6.3: recall over planted fragments with CIs; `scripts/show_checks.py`; `scripts/trace_response.py`.

## 0.5

- Accepted answers include the forms readers reply with; reader probabilities floored at 10⁻⁶ (at most 13.8 nats per
  fact); tolerance in nats.
- 0.5.1: a deleted header, a fragment about another shipment or a true value next to its own attribute is not a
  wrong statement.

## 0.3–0.4

- List numbers stay with their line; deleted fragments split into uninformative, repeat and wrong; overhead split
  into ×removable and ×wordiness; absolute tolerance; facts metric.
