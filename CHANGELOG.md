# Changelog

A change of the scoring version makes a resumed run recompute its scores ([running](docs/running.md#rescoring)).

## 1.2.0 (scoring version 10, signals version 3)

Fixes after a review of 1.1.0. Every score is recomputed; the development runs are made again
([running](docs/running.md#rescoring)).

Scoring (`drybench/filler.py`, `forms.py`, `reader.py`, `segment.py`, `abstain.py`):
- Of two statements of a fact the first full one stays and the later one goes, as the annotation guidelines say;
  `--no-keep-first` turns this off.
- A deleted fragment that gives a wrong answer is a wrong statement even when it also repeats a fact.
- Completeness caps each fact at the gold's information. A response has no filler map when its positive
  information over all facts is below 1 nat (a contradicted fact no longer hides a delivered one).
- An answer that ends in a digit counts only if the reply does not go on with a digit ("70" is not "700").
- Own wording: an answer that is part of a longer word or name gets no lead ("twenty-five", "West Virginia"); a
  title keeps the name; CUAD dates and terms found by value may be bare numbers; seventy, eighty and
  "twenty-one-year" terms parse.
- Abbreviations and initials do not end a sentence. The gold answer is read as responses are.
- Missing facts: the lexical rule reads clause by clause and knows more phrases.
- "Another shipment" excuses a wrong-looking fragment only when it names another shipment and not the requested one.
- `drybench score` defaults to `--mode loo`, as every run uses.

Items and checks (`drybench/realdata.py`, `poddar.py`, `realcheck.py`):
- SQuAD 2.0 accepts the canonical span, spans overlapping it and spans two annotators chose; no accepted form of
  one question appears in another question of the request or starts another's form; the unanswerable question's
  plausible answer is no other question's answer.
- A held-out run also leaves out MuSiQue questions that share a single-hop step with a development item;
  `filter-items --keep` no longer bypasses `--exclude-items`; the filter report keeps the counts of every top-up.
- Poddar et al.: the closest n-gram must have every content word and the same numbers. R4 scores only responses
  that state the reference answer (the human labels no longer enter the method). Poddar et al. is not a suite of
  the confirmation run.
- R3: a fact is stated by its chosen sentence and the summary only; fact sentences that state no other requested
  answer are preferred; items without such sentences are left out; recall is also given counting the responses
  without a filler map.

Generation (`drybench/backends/hf.py`, `cli.py`): plain greedy decoding (the generation settings a model ships
with, such as Qwen2.5's repetition penalty, are dropped); 320 new tokens by default; responses store the token
count, whether they were cut at the limit, and how they were generated. Tables count cut responses.

Comparisons and protocol (`drybench/compare.py`, `confirm.py`, `scripts/run_final.py`, `docs/PROTOCOL.md`):
- Primary metrics: pairwise accuracy (all pairs and pairs of answers to the same item) and Spearman, with 95% CIs
  from bootstrap resamples of items. The filler map's fragment AUC ranks fragments by leave-one-out cost.
- `drybench confirm`: the eight confirmatory tests (H1–H3) with Holm adjustment. The protocol lists exclusions,
  tests and what is fixed before the run; `run_final.py --print-hash`; protocol changes are logged in
  `protocol.json`; the code hash covers the package and the four run scripts; development items and data files
  are hashed.
- `colab/dev_baselines_a100.ipynb`: the LLM judge and Selective Context on the development runs.
- Reader agreement (Kendall τ) on the responses both readers scored, with a CI. β_ctrl also controls for
  "Explain how you found it.".

Runs and annotation:
- A last JSONL line cut by an interrupted write is dropped before the next append. A resumed step with other
  generation or scoring settings stops.
- Response ids of the annotation samples carry the sample's id; the label report refuses labels of another sample
  and merges re-exports of one annotator. The offline labelling page asks the annotator's name, keeps each
  loaded sample and can load another one.

Repository: MIT license; licenses of the data per source (`results/dev_v1.0/README.md`); the notebooks embed a
byte-reproducible package; `scripts/trace_response.py` shows the status, the first-statement pass and each
deleted fragment's own wording; outdated `scripts/run_pipeline.sh` and `colab/missing_fact_check.py` removed.

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
