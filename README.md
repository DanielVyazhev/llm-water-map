# Filler in LLM answers

Code and results for a paper on filler in LLM answers: text that adds length but no requested information.

A reader model sees only the request and the response and answers a question per requested fact. A fragment of the
response is removable if every fact survives its deletion. Removable fragments are filler or wrong statements. The
benchmark applies this to synthetic reports and four real-text datasets, with checks against known truth and human
labels.

```mermaid
flowchart LR
    I["Item: source text, request,<br>facts, gold answer"] --> M["Evaluated model"]
    M --> Y["Response"]
    Y --> R["Reader: request + response,<br>no source text"]
    R --> F["Filler map: each fragment<br>kept / filler / wrong"]
    F --> S["Removable share, facts,<br>completeness, overhead"]
```

## Status (October 2026)

- Code v1.2.5 (package `drybench`): fixes after a review of v1.1.0 (answer matching, the deletion search keeps the
  first statement of a fact, completeness, segmentation, plain greedy decoding, the truth of the checks, resumable
  runs), and the confirmation run on an A100 server with local models; [CHANGELOG.md](CHANGELOG.md).
- Results: development runs on a Colab T4 (models up to 1.7B, readers Qwen2.5-3B and Phi-3.5-mini), scored by
  v1.0.0. Redone with v1.2.1 (`t4_n20_v12`, `real_t4_n15_v12`; values of the confirmatory statistics in
  [docs/PROTOCOL.md](docs/PROTOCOL.md#development-values)); the LLM judge and Selective Context on them next.
- Next: the confirmation run on an A100 server (Llama-3.2-3B, Phi-4-mini, Qwen2.5-14B;
  [docs/PROTOCOL.md](docs/PROTOCOL.md)), human labels.

## Filler map vs simple metrics

![Pairs of responses ranked correctly by filler, per dataset](docs/img/filler_vs_baselines.png)

Metric: pairwise accuracy. Take two responses from one dataset whose true filler shares differ by at least 20
percentage points. A metric is right if it gives the response with more filler the higher score; a tie counts half.
50% is a coin flip, 100% is always right. With a two-class truth this is ROC-AUC. The confirmation run tests
pairwise accuracy and Spearman's correlation with the true filler share ([docs/PROTOCOL.md](docs/PROTOCOL.md)).

The responses have known filler. They consist of sentences that state the requested facts (templated on synthetic
reports, taken from the source text on real texts) and planted filler: openers, closings, hedges, generic remarks,
restated requests, true but unrequested sentences, summaries. On real texts one condition also plants a wrong
statement, which counts as removable. Development runs, reader Qwen2.5-3B, v1.0.0 scores.

- The filler map is right in 84–99% of pairs on every dataset. Left out: responses from which the reader reads no
  fact (39 of 298 on real texts). Counted as all removable, the filler map is right in 74–83% on real texts.
- Length works only where the fact sentences are short, so filler is what makes a response long: synthetic reports
  (median 32 characters) and Poddar et al. (83). On SQuAD 2.0, MuSiQue and CUAD (145–162) it is close to a coin
  flip (46–58%).
- Per fragment (filler or wrong vs needed), ranking fragments by the information lost when each is deleted alone
  gives ROC-AUC 0.98 on synthetic reports and 0.90–1.00 on real texts. The simple metrics give one number per
  response and have no per-fragment score.
- Not in the chart: answers of 10 LLMs labelled by people (Poddar et al.). There length is right in 92% of pairs
  and the filler map in 88% (95% CI of the difference: −8.5 to −0.2 points). The human labels of the confirmation
  run decide it ([docs/results.md](docs/results.md)).

The chart is rebuilt by `python scripts/readme_figure.py`.

## Documents

| Document | Contents |
|---|---|
| [docs/method.md](docs/method.md) | reader information, accepted answers, deletion search, kinds of deleted fragments, metrics |
| [docs/benchmark.md](docs/benchmark.md) | item format, synthetic generator, SQuAD 2.0, MuSiQue, CUAD, Poddar et al., leakage filter |
| [docs/validation.md](docs/validation.md) | checks with known truth, human labels, baselines, second reader |
| [docs/results.md](docs/results.md) | development runs: numbers, the v1.0.0 matching issue, open questions |
| [docs/examples.md](docs/examples.md) | one scored response per suite and one per check |
| [docs/running.md](docs/running.md) | install, Colab, GPU server, rescoring, inspecting a run |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | confirmation run: what is frozen, exclusions, the confirmatory tests |
| [docs/ANNOTATION.md](docs/ANNOTATION.md) | guide for annotators |
| [CHANGELOG.md](CHANGELOG.md) | versions |
| [results/dev_v1.0](results/dev_v1.0/README.md) | tables and raw outputs of the development runs |

## Layout

```text
drybench/     package: reader, filler map, metrics, datasets, baselines, reports
scripts/      runs: synthetic suite, real texts, confirmation run, annotation samples, inspection
colab/        notebooks and their templates (scripts/make_colab.py builds the notebooks)
annotation/   labelling page
tests/        tests (mock reader, no GPU)
results/      dev_v1.0: development runs; pilot: GPT-2 pilot; abstain_labels.json: hand labels
docs/         documents above
```

## License

Code: MIT ([LICENSE](LICENSE)). Data in `results/`: the licenses of their sources
([results/dev_v1.0](results/dev_v1.0/README.md#licenses)).

## Quick start

```bash
pip install -e ".[hf,plot,dev]"
python -m pytest -q          # 214 tests, no GPU
```

Runs with real models need a GPU: [docs/running.md](docs/running.md).
