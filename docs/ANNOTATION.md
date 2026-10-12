# Human labels

The confirmation run (`docs/PROTOCOL.md`) writes two samples into `RUN/annotation/` (also `RUN_annotation.zip`):

| File | What | Time |
|---|---|---|
| `sample.jsonl` | ≈190 answers to label fragment by fragment: 6 items per source (synthetic reports, SQuAD 2.0, MuSiQue, CUAD) × up to 8 answers per item from different models and prompts, so each source text is read once for several answers | 2–3 h |
| `abstain_sample.csv` | 150 answers to requests whose source does not give one of the requested facts | 30 min |

Do not open `key.jsonl` or `abstain_key.jsonl` until you finish: they name the models and hold the benchmark's
own labels.

## Fragment labels

1. Open the labelling page: the published page (labels are shared through the page) or
   `annotation/labelling_page.html` in a browser (labels stay in that browser until you export them).
2. Load `sample.jsonl`. Answers to the same item come one after another.
3. Label every fragment (a sentence or a list line). The guidelines are on the page (“How to label”, with two
   examples); in short:
   - **Needed** — deleting it loses something the request asks for: a requested value or answer, a step of an
     explanation the request asks for, or the correct statement that the source does not give a requested fact;
   - **Filler** — deleting it loses nothing requested, and nothing in it is wrong;
   - **Wrong** — a wrong answer or a claim the source contradicts (takes priority; add “+ needed fact” if the
     fragment also gives a requested fact correctly);
   - keys 1 / 2 / 3, 4 = + needed fact, U = unsure, Enter = next answer.
4. **Export labels** when done (`span_labels_DATE.json`).

Second annotator: the same page and file, at least the first 60 answers, independently, then export. The report
uses the most complete annotator as the truth and the other one for agreement (Cohen's κ).

## Abstention labels

Open `abstain_sample.csv` in a spreadsheet. For each row read the answer and fill `label (A/B/V/O)` for the
question in “question the source does not answer”:

| Label | The answer … |
|---|---|
| A | says the fact is not given |
| B | says so and also guesses a value |
| V | gives a value |
| O | neither (answers something else) |

## Report

```bash
drybench annotation-report --sample RUN/annotation --labels span_labels_me.json span_labels_second.json \
    --abstain-csv abstain_sample_filled.csv
```

`RUN/annotation/labels_report.md`: agreement between annotators; every method against the truth — the filler map
of each reader, the LLM judge and Selective Context (run on the sample by the confirmation run), length, Yap
excess and stop-word share — over all pairs and over pairs of answers to the same item, per source; the
wrong-statement label; what the filler map did with each label; and the abstention rule's precision and recall on
the held-out labels.

## Chat LLMs as extra annotators (optional)

`python scripts/chat_annotators.py make RUN/annotation/sample.jsonl --out chat/` writes a prompt with the same
guidelines and the sample in parts; paste them into a chat, save the replies, then
`python scripts/chat_annotators.py collect RUN/annotation/sample.jsonl --model gpt chat/gpt_*.txt --out gpt.jsonl`
and add `gpt.jsonl` to `--labels` (it appears as another annotator). Chat interfaces do not fix the model version;
the reproducible judge is `drybench baselines --method judge`.

## Labels of Poddar et al.

The check against Poddar et al.'s human span labels (R4) needs nothing by hand: their labels come with the data.
