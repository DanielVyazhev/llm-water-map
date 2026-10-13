# Protocol of the confirmation run (package v1.2.0)

## Development, registration, confirmation

```mermaid
flowchart LR
    D1["Development v1.0<br>t4_n20, real_t4_n15<br>design decisions"] --> D2["Development v1.2<br>t4_n20_v12, real_t4_n15_v12<br>+ LLM judge, Selective Context"]
    D2 --> R["Registration (OSF)<br>this file, commit, code hash,<br>development items, development values"]
    R --> C["Confirmation run<br>new items, 7 models, 3 readers<br>human labels of its answers"]
```

**Development runs.** All design decisions were made on them.
- v1.0 (`results/dev_v1.0`, Colab T4): synthetic reports `t4_n20` (item seed 0, 20 items per cell) and real texts
  `real_t4_n15` (SQuAD 2.0, MuSiQue, CUAD, Poddar et al.; 15 items per cell); models Qwen2.5-0.5B/1.5B-Instruct and
  SmolLM2-1.7B-Instruct; readers Qwen2.5-3B-Instruct and Phi-3.5-mini-instruct; Poddar et al.'s 189
  human-labelled responses; 600 missing-fact answers labelled while the abstention rule was built.
- v1.2 (the same design, package v1.2.0 after review): `t4_n20_v12`, `real_t4_n15_v12` on a T4, then the LLM judge
  (Phi-4) and Selective Context (Qwen2.5-3B) on both on an A100 (`colab/dev_baselines_a100.ipynb`). Its values of
  the confirmatory statistics are the development values of the registration.

**Confirmation run.** `colab/final_a100.ipynb` or `scripts/run_final.py`, package v1.2.0:
- synthetic reports: new items (seed 1, 30 per cell, 450 items); planted-filler check T1 on items of seed 2027
  (development: 2026), 5 per cell;
- real texts: SQuAD 2.0, MuSiQue (MuSiQue-Full required), CUAD; 30 items per cell, built only from source texts the
  development items do not use (below); planted-filler check R3 on 4 items per cell;
- seven models: the three development models, Qwen2.5-7B/14B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-9B-it;
  prompts default and "Be concise."; plain greedy decoding (the models' own generation settings, such as a
  repetition penalty, are not applied), at most 400 new tokens (answers cut there are flagged);
- three readers: Qwen2.5-3B-Instruct (main), Phi-3.5-mini-instruct, Phi-4 (14B; answers to 10 items per cell and
  the annotation sample);
- LLM judge (Phi-4) and Selective Context (Qwen2.5-3B) on every response with known truth;
- the samples for human labels, drawn from this run's answers (seed 0): ≈190 answers to 24 items (6 per source),
  fragment labels; 150 answers to missing-fact items, abstention labels.

```mermaid
flowchart TD
    P["protocol.json: package version, code hash, settings,<br>development items (sha256)"] --> S["Synthetic reports → RUN/synthetic"]
    S --> R["Real texts → RUN/real"]
    R --> A["Samples for people → RUN/annotation"]
    A --> Z["RUN.zip, RUN_annotation.zip"]
    A --> L["Human labels (by hand)"]
    L --> T["drybench confirm → RUN/confirmatory.md"]
    P -.->|"rerun with other code or settings"| X["Stop<br>(--allow-protocol-change: run, log the change, report it)"]
```

## Exclusions

| What | Rule | Code |
|---|---|---|
| development source texts | no confirmation item uses a SQuAD 2.0 paragraph, a CUAD contract, a MuSiQue question, or a MuSiQue question that shares a single-hop sub-question with an item of either development run (`results/dev_v1.0/real/items.jsonl`, `real_t4_n15_v12/items.jsonl`) | `drybench.realdata.source_keys`, `filter-items --exclude-items` |
| Poddar et al. | not a suite of the confirmation run: the development human-label check used every question of it usable as an item | `run_final.DEFAULT_SUITES` |
| leaked items | the main reader answers a fact from an empty message (p ≥ 0.05) or cannot read it from the gold answer (p < 0.5) | `drybench.realdata.filter_items` |
| missing-fact items | out of the filler tables, R3 and H1–H3 (their answers are scored for abstention) | `drybench.rescore.has_missing` |
| unreadable gold | a response to an item whose gold answer the reader cannot read is out of that reader's tables (counted) | `drybench.filler` |
| annotation sample | missing-fact items out; items need ≥ 3 distinct answers; identical answers labelled once | `drybench.annotate.build_sample` |

## Frozen

| What | Value | Code |
|---|---|---|
| information a response delivers | I_j = max(ln p, ln ε) − max(ln p∅, ln ε), ε = 10⁻⁶; a reply that ends in a digit must not go on with a digit ("70" is not "700") | `drybench.reader` |
| accepted answers | the gold's forms; for a fact they leave undelivered (p < 0.5), also the response's own wording of it (same words up to case, dashes, quotes, apostrophes, digit separators, '&', accents and articles; CUAD dates and terms by value), with up to 8 words of its phrase in front (never across , ( ) : ; or words such as not / or / than / renew…; a name's modifier such as "North" or "Saint" joins the answer), kept when the reader replies with it at p ≥ 0.02 (best per place, at most 3); the same for each deleted fragment read alone when it is classified | `drybench.forms` |
| delivered fact | I_j ≥ 0.5 nats; a response with less than 1 nat of positive information over all facts has no filler map (its score: all removable, a failed answer) | `drybench.filler` |
| tolerance of the filler map | τ = 0.5 nats, slack min(τ, I_j / 2), every delivered fact separately | `drybench.filler` |
| deletion search | leave-one-out queue; a fragment that cannot go with the ones before it is skipped; then of two statements of the same fact the first full one stays and the later one goes | `drybench.filler` |
| fragments | sentences and list lines; abbreviations and initials do not end a sentence | `drybench.segment` |
| kinds of deleted fragments | wrong statement (p ≥ 0.5 on a wrong answer) before repeat (≥ half of a delivered fact's information) before uninformative | `drybench.filler` |
| completeness κ | per fact, the information delivered capped at the gold's | `drybench.filler` |
| missing facts | lexical rule, clause by clause; missing-fact items are left out of the filler tables | `drybench.abstain` |
| readers | chat template; the question about a missing fact per reader | `drybench.reader` |
| decoding | plain greedy; the model's generation settings dropped; 400 new tokens | `drybench.backends.hf` |
| baselines | length, Yap excess, stop-word share; LLM judge = fragment labels by the annotation guidelines, read off label probabilities; Selective Context = summed token self-information per fragment, as many characters deleted as the main reader's filler map | `drybench.judge`, `drybench.selective` |
| comparisons | the same responses for every method; true share = planted filler and wrong statements, or the fragments labelled filler or wrong (without "+ needed fact"), by characters; primary metrics: pairwise accuracy (pairs whose true share differs by ≥ 0.2, ties half) and Spearman with the true share; 95% CIs from bootstrap resamples of items | `drybench.compare`, `drybench.confirm` |
| human labels | guidelines of v1.0; item-wise sample, seed 0; truth = the annotator with the most complete labels; response ids carry the sample's id and exports are checked against the sample | `drybench.guidelines`, `drybench.annotate` |

## Confirmatory tests

Eight one-sided tests on the main reader (`drybench confirm`, written to `RUN/confirmatory.md`); Holm over the
eight, family-wise α = 0.05. p-values from 2000 bootstrap resamples of items (seed 0): p = (1 + number of
resamples at or beyond the null) / 2001.

| Test | Truth set | Hypothesis | Decision |
|---|---|---|---|
| H1 T1, H1 R3 | planted filler: synthetic reports (T1), real texts (R3) | on the responses with a filler map, a requested fact is lost in fewer than 5% of them, and more than 95% of the deleted fragments are planted filler or planted wrong statements | both: p = the larger of the two p-values (intersection-union) |
| H2 T1, H2 R3, H2 HL | T1, R3, the human labels of the annotation sample (HL) | pairwise accuracy of the filler map minus that of length > 0 | p of the difference |
| H3 T1, H3 R3, H3 HL | T1, R3, HL | Spearman of the filler map with the true share minus that of length > 0 | p of the difference |

H2 and H3 count every response the truth covers; a response from which the reader reads no requested fact counts
as all removable (the score the benchmark gives it), so a reader failure counts against the filler map. H1 is
about deletions, which only a filler map makes: its share of responses with a map is reported next to it.
A test is confirmed when its Holm-adjusted p ≤ 0.05; each is reported with its estimate and 95% CI whichever way it
comes out. No formal power analysis: the sizes are set by one A100 session (~10 h) and 2–3 h of labelling per
annotator; the development intervals of T1 and R3 are far from the nulls, HL has no development counterpart
(on Poddar et al.'s labels length ranked pairs better than the filler map).

## Secondary, with 95% CIs, no tests

1. The other two readers on H1–H3; same-item pairs; pairs where the response with more filler is the shorter one;
   partial Spearman given length; Yap excess and stop-word share; per suite.
2. The LLM judge and Selective Context on the same responses (the judge is the same model as the 14B reader); cost
   per response.
3. Agreement of the filler map with the primary annotator against the agreement of the two annotators (Cohen's κ).
4. Models: filler by item design; "Be concise." (paired CIs per model); abstention on missing facts per model;
   agreement of model rankings and "Be concise." effects between the three readers.
5. Entropy: whether the generating model's span entropy flags filler above chance.
6. The abstention rule: precision and recall on the held-out labels of 150 answers.

## Registration

Before the confirmation run starts, on OSF:
1. the commit of the repository that runs, and this file at that commit;
2. `python scripts/run_final.py --print-hash`: package version and code hash (sha1 of `drybench/` and the four run
   scripts; `protocol.json` records the same and stops a rerun with other code);
3. sha256 of both development items files (`results/dev_v1.0/real/items.jsonl`:
   `bbda8f899a2c5ec8abc319f20021f7c3856a610d147f4283ac0434421b02078c`; `real_t4_n15_v12/items.jsonl`);
4. the development values of the eight statistics (`drybench confirm` on the v1.2 development runs, written by
   `colab/dev_baselines_a100.ipynb`; T1 and R3 only);
5. the annotators: who they are and whether they took part in the development.

A change after registration is a deviation: `--allow-protocol-change` logs it in `protocol.json` under "changes",
and the paper reports it.

## After the run, by hand

1. `annotation/sample.jsonl`: one annotator labels every answer, a second one at least 60
   ([ANNOTATION.md](ANNOTATION.md)).
2. `annotation/abstain_sample.csv`: the label column, without looking at the key.
3. `drybench annotation-report --sample annotation --labels … --abstain-csv …` → `annotation/labels_report.md`.
4. `drybench confirm --run RUN --labels …` → `RUN/confirmatory.md`: the eight tests, Holm-adjusted.
