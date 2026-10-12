# Validation

Checks with known truth, run for every reader. Table ids: `T*` in `tables_synthetic.md`, `R*` in
`tables_real.md`. Numbers of the development runs: [results.md](results.md).

| Check | Truth | Tables |
|---|---|---|
| Reader check | gold answers with and without each fact | T1 (bottom), R1 |
| Planted filler, synthetic | responses built from fact lines + filler of known type | T1, R2 |
| Planted filler, real texts | the item's own source sentences + filler of known type, one wrong statement | R3 |
| Human labels, Poddar et al. | 10 LLMs' responses with span labels by people | R4 |
| Baselines | the responses of the three checks above | T1, R5 |
| Second reader | the same responses scored by another reader | T4, "Agreement between readers" |
| Human labels of the benchmark's answers | sample of the confirmation run | `labels_report.md` (pending) |
| Abstention rule, held out | 150 missing-fact answers | `labels_report.md` (pending) |

## Reader check

| Number | Meaning |
|---|---|
| AUC present vs absent | information of a fact when the gold states it vs when it is left out |
| gold unreadable | share of gold answers the reader reads less than 1 nat from |
| wrong-statement detection | share of questions the reader answers "unknown" with no message; wrong statements are detectable only then |
| nats per fact | median information of a stated fact (cap 13.8) |

## Planted filler

```mermaid
flowchart TD
    I["Item"] --> F["Fact fragments:<br>synthetic: one line per fact;<br>real: the source sentence stating each fact"]
    I --> W["Planted fragments of known type"]
    F --> R["Response with a true label<br>per fragment"]
    W --> R
    R --> M["Filler map"]
    M --> C["Compare with the truth"]
    C --> N["no fact lost · precision · recall<br>repeats resolved · error of the share"]
```

Conditions per item: no filler, 2 or 5 filler fragments, 2 with one fact left out; real texts also 1 filler + 1
wrong statement.

| Planted type | Synthetic | Real texts |
|---|---|---|
| opener, restated request, hedge, closing | yes | yes |
| generic remark ("Carriers, ports and cargo types can vary a lot between shipments.") | yes | yes |
| true but unrequested fact about the shipment; a line about another shipment | yes | — |
| true source sentence from the fact's paragraph or another paragraph | — | yes |
| summary that repeats the facts | yes | yes |
| wrong statement: another item's answer given as the answer | — | yes |

- **No fact lost**: share of responses whose remaining fragments still deliver every fact.
- **Precision**: share of deleted fragments that were planted filler or wrong statements.
- **Recall**: share of planted filler fragments deleted (95% CI, bootstrap over items).
- **Repeats resolved**: no fact is stated twice after deletion (either copy may go).
- A planted fragment left in is reported as "costs more than the tolerance alone" or "blocked in the queue".

## Human labels (Poddar et al.)

Poddar et al. (2025) labelled 500 responses of 10 LLMs character by character: minimal answer (MINANS), additional
information, explanation, conversational, redundant, irrelevant. Used: the passage subsets with a reference answer
of up to 6 words (189 responses; the benchmark's own Poddar et al. items use up to 5).

```mermaid
flowchart TD
    A["Response + character labels"] --> C["Cut text after the last labelled span"]
    C --> S["Fragments; label = the label of<br>most of the fragment's characters"]
    S --> T["MINANS → needed;<br>other labels → not minimal;<br>unlabelled → left out"]
    A --> F["Fact: the reference answer's forms<br>plus what the reader reads<br>from the MINANS fragments"]
    F --> M["Filler map"]
    S --> M
    M --> E["precision · recall · F1 · Cohen's κ<br>share deleted per human label<br>ranking of the 10 LLMs: Kendall τ"]
    T --> E
```

The fact is the response's own answer: annotators marked the minimal answer whether or not it was right. It is
read from the MINANS fragments, so for those responses the human labels enter the method; R4 also reports the
responses whose fact is the reference answer ([results.md](results.md#checks)).

## Baselines

All methods are scored on the same responses.

| Method | Removable score |
|---|---|
| length | characters |
| Yap excess | characters beyond the gold minimal answer, as in YapBench (arXiv:2601.00624) |
| stop-word share | share of English stop words |
| LLM judge | per fragment: p(filler) + p(wrong), read off the probabilities of the label words (`needed`, `filler`, `wrong`); the annotators' guidelines (`drybench/judge.py`) |
| Selective Context | least self-information first, as many characters as the filler map deletes (`drybench/selective.py`) |

Fragment level: F1, Cohen's κ, AUC of removable vs needed. Response level: Spearman with the true removable share,
partial Spearman given length (95% CI, bootstrap over items), and the pair test:

```mermaid
flowchart LR
    P["All pairs of responses<br>in the set"] --> S{"True removable shares<br>differ by ≥ 0.2, and the response<br>with more is the shorter one?"}
    S -->|no| X["Not used"]
    S -->|yes| R{"Does the method score<br>that response higher?"}
    R --> O["Share ranked right<br>(ties reported apart)"]
```

Length is wrong on every such pair by construction. Responses from which a reader reads nothing are reported both
without and with them (then all of such a response counts as removable).

## Second reader

The same responses scored by another reader family (development: Phi-3.5-mini; confirmation: also Phi-4). Reported:
Spearman per response, Kendall τ of the model × prompt ranking, agreement per fragment (deleted or kept, Cohen's κ).
