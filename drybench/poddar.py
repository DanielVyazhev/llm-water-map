"""Poddar et al. (2025), "Brevity is the soul of sustainability: Characterizing LLM response
lengths" (Findings of ACL 2025; github.com/sohampoddar26/LLM-brevity, GPL-3.0).

data.jsonl holds 500 responses of 10 LLMs (GPT-3.5/4o-mini, Llama, Gemma, Mistral, Phi) to
questions from five datasets, with human character-span labels:
  MINANS   minimal answer            ADDINFO  additional helpful information
  EXPLAIN  reasoning / explanation   CONVERS  conversational phrases, disclaimers
  REDINFO  redundant (repeated)      IRREL    irrelevant information
Three subsets come with a passage and a short reference answer (NarrativeQA summaries,
TweetQA tweets, MS MARCO search passages); Dolly and GooAQ do not (open questions without a
source text), so they are not used.

Two uses:
  poddar_items   the passage questions as benchmark items for the evaluated models
                 (reference answer of at most MAX_REF_WORDS words; one cell per subset)
  labelled_cases the 10 LLMs' responses to the same questions with their human labels, to
                 compare the filler map's deletions with what annotators marked as not minimal
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from drybench.realdata import _EDGE_PUNCT, _end, _norm_space, text_answer_forms
from drybench.types import Item, Probe, read_jsonl

PODDAR_SUBSETS = ("narrqa", "tweetqa", "msmarco")
PODDAR_CELLS: list[dict[str, Any]] = [{"subset": s} for s in PODDAR_SUBSETS]
DOC_WORDS = {"narrqa": ("Story summary", "story summary"), "tweetqa": ("Tweet", "tweet"),
             "msmarco": ("Search results", "search results")}
LABELS = ("MINANS", "ADDINFO", "EXPLAIN", "CONVERS", "REDINFO", "IRREL")
SEP = "\n*#\n*#\n*#\n"
MAX_REF_WORDS = 5        # items for the evaluated models (as SQuAD: short extractive answers)
MAX_REF_WORDS_LABELLED = 6  # labelled responses: the response's own wording of the answer is accepted too
MIN_MATCH_F1 = 0.75


def parse_record(row: dict) -> dict | None:
    """One data.jsonl line -> {rid, subset, model, qid, passage, question, reference, response,
    labels: [(start, end, label)] with offsets into the response}."""
    text = row["text"]
    header, _, _ = text.partition("\n")
    parts = header.rsplit("_", 2)
    cut = text.find(SEP)
    if len(parts) != 3 or cut < 0:
        return None
    subset, model, qid = parts
    pre, response, base = text[:cut], text[cut + len(SEP):], cut + len(SEP)
    m = re.search(r"\n### Question: (.*?)\n### Response: (.*)$", pre, re.S)
    if m:
        passage, question, reference = pre[len(header) + 1: m.start()], m.group(1), m.group(2)
    else:
        m = re.search(r"\n### Response: (.*)$", pre, re.S)
        if not m:
            return None
        passage, question, reference = None, pre[len(header) + 1: m.start()], m.group(1)
    labels = sorted((max(a, base) - base, min(b, len(text)) - base, lab)
                    for a, b, lab in row.get("label", []) if b > base and lab in LABELS)
    return {"rid": row.get("id"), "subset": subset, "model": model, "qid": qid, "passage": passage,
            "question": _norm_space(question), "reference": _norm_space(reference), "response": response,
            "labels": labels}


def load_poddar(path: str | Path) -> list[dict]:
    return [r for r in map(parse_record, read_jsonl(path)) if r is not None]


def _tokens(s: str) -> list[str]:
    s = re.sub(r"['’.]", "", s.lower())
    return [w for w in re.findall(r"\w+", s) if w not in ("a", "an", "the")]


def best_match(text: str, ref: str, min_f1: float = MIN_MATCH_F1) -> str | None:
    """The word n-gram of `text` closest to `ref` (token F1 after lowercasing and dropping
    articles and apostrophes), as written in `text`; None below min_f1."""
    rt = Counter(_tokens(ref))
    if not rt:
        return None
    words = list(re.finditer(r"\S+", text))
    n_max = len(ref.split()) + 2
    best_f, best = 0.0, None
    for i in range(len(words)):
        for n in range(1, n_max + 1):
            if i + n > len(words):
                break
            span = text[words[i].start(): words[i + n - 1].end()].strip(_EDGE_PUNCT)
            st = Counter(_tokens(span))
            common = sum((st & rt).values())
            if not common:
                continue
            p, r = common / sum(st.values()), common / sum(rt.values())
            f = 2 * p * r / (p + r)
            if f > best_f + 1e-9 or (abs(f - best_f) <= 1e-9 and best is not None and len(span) < len(best)):
                best_f, best = f, span
    return best if best is not None and best_f >= min_f1 else None


def _question(q: str) -> str:
    q = _norm_space(q)
    q = q[:1].upper() + q[1:]
    return q if q.endswith("?") else q.rstrip(".") + "?"


def _document(passage: str) -> str:
    lines = [_norm_space(line) for line in passage.split("\n")]
    return "\n".join(line for line in lines if line)


def answer_forms(record: dict, extra_texts: Sequence[str] = ()) -> list[str]:
    """Reference answer + its closest wording in the passage (and in `extra_texts`)."""
    ref = record["reference"].strip(_EDGE_PUNCT + " ")
    found = [m for t in [record["passage"] or "", *extra_texts] for m in [best_match(t, ref)] if m]
    return text_answer_forms([ref, *found])


def _item(record: dict, forms: list[str], item_id: str, ci: int, n_labelled: int) -> Item:
    subset = record["subset"]
    label, noun = DOC_WORDS[subset]
    q = _question(record["question"])
    ref = record["reference"].strip(" ")
    probe = Probe(question=q, answers=forms, fact_id=f"poddar:{subset}:{record['qid']}", value=ref.strip(_EDGE_PUNCT))
    knobs = {"k": 1, "d": 0, "answerable": True, "ambiguous": False, "hops": 1, "explain": False,
             "source": "poddar", "subset": subset}
    meta = {"source": "poddar", "cell": ci, "doc_label": label, "doc_noun": noun, "subset": subset,
            "qid": record["qid"], "reference": ref, "labelled_responses": n_labelled}
    return Item(item_id, _document(record["passage"]), q, [probe], _end(ref.strip(_EDGE_PUNCT)), knobs, meta)


def usable(record: dict, max_ref_words: int = MAX_REF_WORDS) -> bool:
    """A passage question with a short reference answer (a few source records carry a tweet's
    signature line instead of a question; those are skipped)."""
    q = record["question"].strip()
    return (record["subset"] in PODDAR_SUBSETS and bool(record["passage"]) and q[:1].isalpha()
            and 0 < len(record["reference"].split()) <= max_ref_words)


def poddar_items(records: Sequence[dict], n_per_cell: int, seed: int = 0,
                 cells: Sequence[dict[str, Any]] = PODDAR_CELLS) -> tuple[list[Item], dict]:
    """One item per distinct passage question (records of several LLMs share a question)."""
    import random

    items, stats = [], {}
    for ci, cell in enumerate(cells):
        subset = cell["subset"]
        groups: dict[str, list[dict]] = {}
        for r in records:
            if r["subset"] == subset and usable(r):
                groups.setdefault(r["qid"], []).append(r)
        qids = sorted(groups)
        random.Random(f"{seed}-poddar-{ci}").shuffle(qids)
        j = 0
        for qid in qids:
            if j >= n_per_cell:
                break
            rec = groups[qid][0]
            forms = answer_forms(rec)
            if not forms:
                continue
            items.append(_item(rec, forms, f"pd{ci:02d}-{j:04d}", ci, len(groups[qid])))
            j += 1
        stats[f"poddar:{ci}"] = {"cell": cell, "built": j, "wanted": n_per_cell, "available": len(groups)}
    return items, stats


# ---------------------------------------------------------------- human labels

def labelled_cases(records: Sequence[dict], max_ref_words: int = MAX_REF_WORDS_LABELLED) -> list[dict]:
    """The labelled responses to usable passage questions: {item, response, labels, record}.

    The accepted answers also include the response's own wording of the reference (closest
    n-gram). Text after the last labelled span is cut off: annotators left it unlabelled, and in
    these data it is a degenerate continuation (a model writing new '### Prompt:' blocks)."""
    out = []
    for r in records:
        if not usable(r, max_ref_words) or not r["labels"]:
            continue
        end = max(b for _, b, _ in r["labels"])
        response = r["response"][:end].rstrip()
        forms = answer_forms(r, [response])
        if not forms:
            continue
        ci = PODDAR_SUBSETS.index(r["subset"])
        out.append({"item": _item(r, forms, f"pdh-{r['subset']}-{r['qid']}-{r['model']}", ci, 1),
                    "response": response, "labels": r["labels"], "record": r,
                    "cut_chars": len(r["response"][end:].strip())})
    return out


def fragment_labels(spans: Sequence, labels: Sequence[tuple[int, int, str]]) -> list[str]:
    """Majority human label of each fragment (by non-space characters); 'NONE' when most of the
    fragment is unlabelled."""
    out = []
    for s in spans:
        count: Counter = Counter()
        text = s.text
        for off, ch in enumerate(text):
            if ch.isspace():
                continue
            pos = s.start + off
            lab = next((lab for a, b, lab in labels if a <= pos < b), "NONE")
            count[lab] += 1
        out.append(count.most_common(1)[0][0] if count else "NONE")
    return out
