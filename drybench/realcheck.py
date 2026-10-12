"""Checks of the filler map on real texts, where the truth is known.

planted_check   Simulated responses built from an item's own document: the sentence that states
                each requested fact, plus planted filler of known type. Filler is either formulaic
                (preamble, restatement, hedge, generic remark, closing), real but unrequested
                document sentences (from the fact's paragraph: elaboration; from another
                paragraph: unrequested), or a summary that repeats the facts. One condition also
                plants a wrong statement (another item's answer given as the answer), which the
                filler map should delete and label as wrong. Truth per fragment is known, so
                safety, precision and recall are exact; baselines are scored on the same
                responses.

human_label_check  Responses of 10 LLMs to Poddar et al.'s passage questions with human span
                labels (minimal answer vs additional / explanation / conversational / redundant
                / irrelevant): the filler map's deletions against the annotators' labels.

Both write rows.jsonl (resumable) and summary.json into an output folder.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from drybench.baselines import EN_STOPWORDS, length_chars, stopword_share, yap_excess_chars
from drybench.items import NOT_STATED
from drybench.realdata import _contains, _norm, cell_key
from drybench.reader import Reader
from drybench.segment import segment
from drybench.simulate import SimResponse, evaluate_filler_map, true_filler_fraction, filler_check_metrics
from drybench.types import Item, Span
from drybench.filler import dry_score

REAL_FILLER_TYPES = ("preamble", "restatement", "hedge", "generic", "elaboration", "unrequested", "summary",
                    "closing")
FORMULAIC = ("preamble", "restatement", "hedge", "generic", "closing")
# name: (filler fragments, facts left out, wrong statements)
CONDITIONS = {"dense": (0, 0, 0), "light": (2, 0, 0), "heavy": (5, 0, 0), "incomplete": (2, 1, 0),
              "wrong": (1, 0, 1)}
TEMPLATES = {
    "preamble": ["Great question!", "Sure, let me look at the {noun} for you.",
                 "Thanks for asking, here is what the {noun} says.", "Certainly, I checked the {noun}."],
    "restatement": ["You asked: {question}", "Here is the information you requested from the {noun}."],
    "hedge": ["Please note that the {noun} may not cover every detail.",
              "It is a good idea to double-check this against the full {noun}.",
              "Keep in mind that this answer is based only on the {noun} provided."],
    "generic": ["Questions like this can often be answered in more than one way.",
                "Details like these are sometimes reported differently in different sources.",
                "Such details are usually given close to the relevant passage."],
    "closing": ["I hope this helps!", "Let me know if you need anything else.",
                "Feel free to ask if you have more questions."],
}
# row prefix: (absolute tolerance in nats, skip the queue's blocking span); "loo" is the default,
# "prefix" the deletion search of v0.7.1 and before (ablation)
SETTINGS = {"loo": (0.5, True), "t2": (2.0, True), "prefix": (0.5, False)}
_SKIP_SENT = re.compile(r"^(?:Document \d+:|\[\.\.\.\])")


def _noun(item: Item) -> str:
    return item.meta.get("doc_noun") or "text"


def document_sentences(document: str, min_words: int = 1) -> list[tuple[int, str]]:
    """(paragraph index, sentence) for the document's sentences of min_words+ words (MuSiQue's
    'Document i:' headers and the '[...]' marks of an excerpt are left out)."""
    out = []
    for pi, para in enumerate(re.split(r"\n\s*\n", document)):
        for s in segment(para):
            t = " ".join(s.text.replace("[...]", " ").split())
            if len(t.split()) >= min_words and not _SKIP_SENT.match(s.text) and re.search(r"[A-Za-z0-9]", t):
                out.append((pi, t))
    return out


def _mentions_any(text: str, forms: Sequence[str]) -> bool:
    """A form occurs as a whole word or phrase (any case)."""
    return any(_contains(text, f) for f in forms if f and f.lower() not in ("not stated",))


def _starts_any(text: str, forms: Sequence[str]) -> bool:
    """A form occurs at the start of a word (any case): the reader's answer 'cylinder' covers
    'cylinders', so such a sentence states the answer."""
    low = text.lower()
    return any(re.search(r"(?<!\w)" + re.escape(f.lower()), low) for f in forms if f and f.lower() != "not stated")


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in EN_STOPWORDS and len(w) > 2}


def fact_sentence(item: Item, probe, sents: Sequence[tuple[int, str]]) -> tuple[int | None, str]:
    """The document sentence that states the probe's answer: the one with the item's evidence
    phrase for this fact (CUAD), else one with the answer in the exact case the reader is asked
    for, then most words shared with the question, then the shortest; a constructed sentence when
    the document has none."""
    if NOT_STATED in probe.answers:
        q = probe.question
        return None, f"Not stated in the {_noun(item)}: {q[:1].lower() + q[1:]}"
    cands = [(pi, t) for pi, t in sents if _mentions_any(t, probe.answers)]
    if not cands:
        return None, f"{probe.question} {probe.value or probe.answers[0]}."
    evidence = (item.meta.get("evidence") or {}).get(probe.fact_id, "")
    qw = _content(probe.question)

    def rank(c):
        exact = any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", c[1]) for a in probe.answers)
        return (bool(evidence) and evidence.lower() in c[1].lower(), exact, len(qw & _content(c[1])), -len(c[1]))

    return max(cands, key=rank)


def _answer_kind(item: Item, probe) -> str:
    """What kind of value answers the probe: the CUAD fact type, else the question word and
    whether the answer has digits (a wrong answer of the same kind is a plausible one)."""
    digits = "#" if re.search(r"\d", probe.value or probe.answers[0]) else ""
    if item.meta.get("source") == "cuad":
        return (probe.fact_id or "").rsplit(":", 1)[-1]
    words = re.findall(r"[a-z]+", probe.question.lower())
    wh = next((w for w in words if w in ("who", "what", "when", "where", "which", "how", "why", "whose")), "")
    if wh == "how" and len(words) > words.index("how") + 1:
        wh += " " + words[words.index("how") + 1]
    return wh + digits


def wrong_pool(items: Sequence[Item]) -> dict[str, list[tuple[str, str]]]:
    """(answer, kind) of every answerable probe, per source: another item's answer of the same kind
    is a plausible wrong answer."""
    pool: dict[str, set] = {}
    for it in items:
        for p in it.required_probes:
            if NOT_STATED not in p.answers:
                pool.setdefault(it.meta.get("source", "synthetic"), set()).add(
                    (p.value or p.answers[0], _answer_kind(it, p)))
    return {k: sorted(v) for k, v in pool.items()}


def _same_value(a: str, b: str, kind: str) -> bool:
    """Two answers that say the same thing ("24 months" = "2 years", "England" ⊂ "England and Wales")."""
    if kind == "term":
        from drybench.cuad import _duration_key

        ka, kb = _duration_key(a), _duration_key(b)
        if ka and kb:
            return ka[0] * (12 if ka[1] == "year" else 1) == kb[0] * (12 if kb[1] == "year" else 1)
    if kind == "date":
        from drybench.cuad import parse_date

        if parse_date(a) and parse_date(a) == parse_date(b):
            return True
    na, nb = _norm(a), _norm(b)
    return bool(na and nb) and (na == nb or f" {na} " in f" {nb} " or f" {nb} " in f" {na} ")


def _wrong_value(item: Item, probe, pool: Sequence[tuple[str, str]], rng: random.Random) -> str | None:
    kind = _answer_kind(item, probe)
    own = [a for p in item.probes for a in p.answers] + [p.value for p in item.probes if p.value]
    ok = [(w, k) for w, k in pool if not any(_same_value(w, a, kind) for a in own)
          and not _contains(item.document, w) and not _contains(item.request, w)]
    digits = "#" in kind or kind in ("date", "term")
    cands = [w for w, k in ok if k == kind] or [w for w, k in ok if bool(re.search(r"\d", w)) == digits]
    return rng.choice(cands) if cands else None


def simulate_real(item: Item, filler: Sequence[str] = (), omit: Sequence[str] = (), n_wrong: int = 0,
                  rng: random.Random | None = None, pool: Sequence[str] = ()) -> SimResponse:
    """A response to a real-text item from its own document, with planted filler of known type."""
    rng = rng or random.Random(0)
    unknown = set(filler) - set(REAL_FILLER_TYPES)
    if unknown:
        raise ValueError(f"unknown filler types {unknown}")
    noun = _noun(item)
    sents = document_sentences(item.document)
    probes = item.required_probes
    all_forms = [a for p in item.probes for a in p.answers if NOT_STATED not in p.answers]
    all_forms += list(item.meta.get("plausible_answers") or [])

    facts: list[list] = []  # [text, paragraph, fact ids]
    for p in probes:
        if p.fact_id in omit:
            continue
        pi, text = fact_sentence(item, p, sents)
        same = next((f for f in facts if f[0] == text), None)
        if same:
            same[2].append(p.fact_id)
        else:
            facts.append([text, pi, [p.fact_id]])
    for f in facts:  # a sentence states every requested fact it mentions (an omitted one included)
        for p in probes:
            if p.fact_id not in f[2] and NOT_STATED not in p.answers and _mentions_any(f[0], p.answers):
                f[2].append(p.fact_id)
    present = [p.fact_id for p in probes if any(p.fact_id in f[2] for f in facts)]
    fact_texts = {f[0] for f in facts}
    fact_paras = {f[1] for f in facts if f[1] is not None}
    free = [(pi, t) for pi, t in sents if t not in fact_texts and len(t.split()) >= 4
            and not _starts_any(t, all_forms)]
    near = [t for pi, t in free if pi in fact_paras]
    far = [t for pi, t in free if pi not in fact_paras]
    rng.shuffle(near)
    rng.shuffle(far)
    question = probes[0].question if probes else item.request

    def tmpl(kind: str) -> str:
        return rng.choice(TEMPLATES[kind]).format(noun=noun, question=question)

    head: list[tuple[str, str, tuple]] = []
    body: list[tuple[str, str, tuple]] = [(t, "fact", tuple(fids)) for t, _, fids in facts]
    tail: list[tuple[str, str, tuple]] = []
    used: set[str] = set()
    for kind in filler:
        if kind in ("preamble", "restatement"):
            head.append((tmpl(kind), f"filler:{kind}", ()))
        elif kind in ("hedge", "generic"):
            body.insert(rng.randint(0, len(body)), (tmpl(kind), f"filler:{kind}", ()))
        elif kind in ("elaboration", "unrequested"):
            src, label = (near, "elaboration") if kind == "elaboration" else (far, "unrequested")
            if not [t for t in src if t not in used]:  # one paragraph only: take what there is
                src, label = (far, "unrequested") if kind == "elaboration" else (near, "elaboration")
            fresh = [t for t in src if t not in used]
            if not fresh:
                continue
            used.add(fresh[0])
            span = (fresh[0], f"filler:{label}", ())
            if label == "elaboration" and body:
                body.insert(rng.randint(1, len(body)), span)
            else:
                tail.append(span)
        elif kind == "summary":
            vals = ["not stated" if NOT_STATED in p.answers else (p.value or p.answers[0])
                    for p in probes if p.fact_id in present]
            if vals:
                tail.append((f"In summary: {'; '.join(vals)}.", "filler:summary", tuple(present)))
        elif kind == "closing":
            tail.append((tmpl(kind), "filler:closing", ()))
    targets = [p for p in probes if p.fact_id in present and NOT_STATED not in p.answers]
    for _ in range(n_wrong):
        if not targets:
            break
        p = rng.choice(targets)
        w = _wrong_value(item, p, pool, rng)
        if w:
            tail.insert(0, (f'The answer to "{p.question}" is {w}.', "wrong", ()))
    head.sort(key=lambda x: x[1] != "filler:preamble")
    rank = {"wrong": 0, "filler:unrequested": 1, "filler:elaboration": 1, "filler:summary": 2, "filler:closing": 3}
    tail.sort(key=lambda x: rank[x[1]])
    spans, pos, pieces = [], 0, []
    for text, label, fids in head + body + tail:
        if pieces:
            pos += 1
        spans.append(Span(pos, pos + len(text), text, label=label, fact_ids=tuple(fids)))
        pieces.append(text)
        pos += len(text)
    return SimResponse(text=" ".join(pieces), spans=spans, present_facts=present, filler_types=list(filler))


# ---------------------------------------------------------------- planted check

def _corr(x, y) -> dict:
    from scipy.stats import pearsonr, spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return {"spearman": float("nan"), "pearson": float("nan"), "n": int(m.sum())}
    return {"spearman": float(spearmanr(x[m], y[m])[0]), "pearson": float(pearsonr(x[m], y[m])[0]),
            "n": int(m.sum())}


def _pairwise(rows: Sequence[dict], key: str, min_gap: float = 0.2) -> dict:
    """Among pairs where the more removable response is SHORTER, how often does the metric rank it higher?"""
    w = np.array([r["W_true"] for r in rows], float)
    L = np.array([r["tokens"] for r in rows], float)
    v = np.array([r.get(key, np.nan) for r in rows], float)
    hits = total = 0
    for i in range(len(rows)):
        sel = (w[i] - w >= min_gap) & (L[i] < L) & np.isfinite(v) & np.isfinite(v[i])
        total += int(sel.sum())
        hits += int((v[i] > v[sel]).sum())
    return {"accuracy": hits / total if total else float("nan"), "pairs": total}


def _score_sim(reader: Reader, item: Item, sim: SimResponse, length_fn: Callable[[str], int]) -> dict:
    lengths = [max(1, length_fn(s.text)) for s in sim.spans]
    row: dict = {"W_true": true_filler_fraction(sim, lengths), "tokens": int(sum(lengths)),
                 "coverage": len(sim.present_facts) / max(1, len(item.required_probes))}
    for prefix, (tol, skip) in SETTINGS.items():
        first = prefix == "loo"
        res = dry_score(reader, item, sim.text, mode="loo", spans=sim.spans, abs_tol=tol, length_fn=length_fn,
                        with_kappa=first, classify=first, loo_skip=skip)
        kept = [s.kept for s in res.spans]
        row.update({f"{prefix}_status": res.status, f"{prefix}_W": res.W, f"{prefix}_kept": kept})
        if res.status == "ok":
            ev = evaluate_filler_map(sim, kept, lengths)
            row.update({f"{prefix}_{k}": v for k, v in ev.items() if k != "W_true"})
        if first:
            row.update({"loo_kinds": [s.kind for s in res.spans], "loo_W_filler": res.W_filler,
                        "loo_W_misleading": res.W_misleading, "info": res.info, "kappa": res.kappa,
                        "facts": res.facts, "loo_candidate": [s.candidate for s in res.spans],
                        "loo_cost": [s.loo_cost for s in res.spans], "per_probe": res.per_probe})
    return row


def select_items(items: Sequence[Item], n_per_cell: int) -> list[Item]:
    """The first n items of every cell (items arrive in pool order, which is random per cell)."""
    seen: Counter = Counter()
    out = []
    for it in items:
        key = cell_key(it)
        if seen[key] < n_per_cell:
            out.append(it)
            seen[key] += 1
    return out


def planted_check(reader: Reader, items: Sequence[Item], out_dir: str | Path, n_per_cell: int = 4,
                  conditions: dict = CONDITIONS, quiet: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pools = wrong_pool(items)
    chosen = select_items(items, n_per_cell)
    rows_path = out / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["item_id"], r["condition"]) for r in rows}
    length_fn = reader.backend.count_tokens
    todo = [(it, c) for it in chosen for c in conditions if (it.item_id, c) not in done]
    t0 = time.time()
    for n, (it, cond) in enumerate(todo, 1):
        n_filler, n_omit, n_wrong = conditions[cond]
        rng = random.Random(f"{it.item_id}-{cond}")
        probes = it.required_probes
        if n_omit and len(probes) < 2:
            continue  # one fact: leaving it out leaves nothing to read
        kinds = rng.sample(REAL_FILLER_TYPES, n_filler)
        pool = pools.get(it.meta.get("source", "synthetic"), [])
        sim, omit = None, []
        for p in (rng.sample(probes, len(probes)) if n_omit else [None]):
            omit = [p.fact_id] if p is not None else []
            sim = simulate_real(it, kinds, omit=omit, n_wrong=n_wrong, rng=random.Random(f"{it.item_id}-{cond}"),
                                pool=pool)
            if not omit or omit[0] not in sim.present_facts:
                break  # (another fact's sentence may state the left-out fact as well)
        if omit and omit[0] in sim.present_facts:
            continue  # every fact is stated in another fact's sentence: nothing can be left out
        if n_wrong and "wrong" not in [s.label for s in sim.spans]:
            continue  # no wrong answer of the same kind available
        row = {"item_id": it.item_id, "source": it.meta.get("source", "synthetic"), "cell": cell_key(it),
               "condition": cond, "filler_types": kinds, "n_omitted": len(probes) - len(sim.present_facts),
               "text": sim.text,
               "labels": [s.label for s in sim.spans], "span_texts": [s.text for s in sim.spans],
               "span_facts": [list(s.fact_ids) for s in sim.spans],
               "chars": length_chars(sim.text), "yap_excess": yap_excess_chars(sim.text, it.gold_minimal),
               "stopword_share": stopword_share(sim.text)}
        row.update(_score_sim(reader, it, sim, length_fn))
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not quiet and (n % 25 == 0 or n == len(todo)):
            print(f"planted {n}/{len(todo)} [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    summary = summarize_planted(rows)
    summary["config"] = {"n_per_cell": n_per_cell, "conditions": conditions, "items": len(chosen),
                         "settings": SETTINGS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _wrong_stats(rows: Sequence[dict]) -> dict:
    planted = deleted = labelled = 0
    for r in rows:
        if r.get("loo_status") != "ok":
            continue
        for lab, k, kind in zip(r["labels"], r["loo_kept"], r["loo_kinds"]):
            if lab == "wrong":
                planted += 1
                deleted += not k
                labelled += kind == "misleading"
    return {"planted": planted, "deleted": deleted / planted if planted else float("nan"),
            "labelled_wrong": labelled / planted if planted else float("nan")}


def _false_wrong(rows: Sequence[dict]) -> dict:
    """Planted filler and kept facts that were labelled wrong (should be ~0)."""
    filler = wrong = 0
    for r in rows:
        if r.get("loo_status") != "ok":
            continue
        for lab, kind in zip(r["labels"], r["loo_kinds"]):
            if lab.startswith("filler:"):
                filler += 1
                wrong += kind == "misleading"
    return {"filler_spans": filler, "labelled_wrong": wrong / filler if filler else float("nan")}


def summarize_planted(rows: Sequence[dict]) -> dict:
    ok = [r for r in rows if r.get("loo_status") == "ok"]
    out: dict = {"n_responses": len(rows), "n_scored_ok": len(ok)}
    if not ok:
        return out
    out["overall"] = filler_check_metrics(ok, "loo")
    out["overall_t2"] = filler_check_metrics([r for r in rows if r.get("t2_status") == "ok"], "t2")
    prefix = [r for r in rows if r.get("prefix_status") == "ok"]
    out["overall_prefix"] = filler_check_metrics(prefix, "prefix") if prefix else {"n_responses": 0}
    complete = [r for r in ok if r.get("facts") == 1.0]  # the reader reads every fact of the response
    out["overall_complete"] = filler_check_metrics(complete, "loo") if complete else {"n_responses": 0}
    out["by_source"] = {}
    for src in sorted({r["source"] for r in ok}):
        sub = [r for r in ok if r["source"] == src]
        pf = [r for r in prefix if r["source"] == src]
        out["by_source"][src] = {**filler_check_metrics(sub, "loo"), "wrong": _wrong_stats(sub),
                                 "W_vs_true": _corr([r["loo_W"] for r in sub], [r["W_true"] for r in sub]),
                                 "prefix": filler_check_metrics(pf, "prefix") if pf else {"n_responses": 0}}
    out["wrong"] = _wrong_stats(ok)
    out["filler_labelled_wrong"] = _false_wrong(ok)
    out["W_vs_true"] = _corr([r["loo_W"] for r in ok], [r["W_true"] for r in ok])
    metrics = {"removable share (filler map)": "loo_W", "length": "tokens", "Yap excess (chars)": "yap_excess",
               "stop-word share": "stopword_share"}
    out["metric_vs_true_filler"] = {name: _corr([r.get(k) for r in ok], [r["W_true"] for r in ok])
                                   for name, k in metrics.items()}
    out["length_confound_pairs"] = {name: _pairwise(ok, k) for name, k in metrics.items()}
    inc = [r for r in ok if r.get("kappa") is not None and r["kappa"] == r["kappa"]]
    out["kappa"] = {"vs_coverage": _corr([r["kappa"] for r in inc], [r["coverage"] for r in inc]),
                    "complete_mean": float(np.mean([r["kappa"] for r in inc if r["n_omitted"] == 0]))
                    if any(r["n_omitted"] == 0 for r in inc) else float("nan"),
                    "incomplete_mean": float(np.mean([r["kappa"] for r in inc if r["n_omitted"] > 0]))
                    if any(r["n_omitted"] > 0 for r in inc) else float("nan")}
    out["by_condition"] = {c: {"n": len(g), "W": float(np.mean([r["loo_W"] for r in g])),
                               "W_true": float(np.mean([r["W_true"] for r in g]))}
                           for c in CONDITIONS for g in [[r for r in ok if r["condition"] == c]] if g}
    return out


# ---------------------------------------------------------------- human labels (Poddar et al.)

def _binary(y_true: Sequence[bool], y_pred: Sequence[bool], weight: Sequence[float] | None = None) -> dict:
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    w = np.ones(len(t)) if weight is None else np.asarray(weight, float)
    if not len(t):
        return {"n": 0}
    tp, fp, fn = float(w[t & p].sum()), float(w[~t & p].sum()), float(w[t & ~p].sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    acc = float(w[t == p].sum() / w.sum())
    pt, pp = float(w[t].sum() / w.sum()), float(w[p].sum() / w.sum())
    pe = pt * pp + (1 - pt) * (1 - pp)
    return {"n": int(len(t)), "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "cohen_kappa": (acc - pe) / (1 - pe) if pe < 1 else float("nan"), "human_positive": pt,
            "predicted_positive": pp}


def own_answer(reader: Reader, item: Item, response: str, labels: Sequence[tuple[int, int, str]]) -> str | None:
    """The answer the response itself gives: the reader's short reply to the question when it reads
    only the fragments the annotators marked as the minimal answer (MINANS). None when the reader
    replies 'unknown' or with words it did not copy from those fragments."""
    mins = " ".join(response[a:b].strip() for a, b, lab in labels if lab == "MINANS").strip()
    if not mins:
        return None
    ans = reader.answer(item.request, mins, item.probes[0].question)
    low = ans.lower()
    if not ans or low.startswith(("unknown", "not stated")) or low not in " ".join(mins.lower().split()):
        return None
    return ans


def human_label_check(reader: Reader, cases: Sequence[dict], out_dir: str | Path, abs_tol: float = 0.5,
                      quiet: bool = False) -> dict:
    """The filler map vs human span labels. The fact a response delivers is its own answer: the
    reference answer's forms plus what the reader reads from the response's minimal-answer span
    (annotators labelled minimal vs extra text whether or not the answer is right)."""
    from drybench.poddar import fragment_labels
    from drybench.realdata import text_answer_forms
    from drybench.types import Probe

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r["case"] for r in rows}
    todo = [c for c in cases if c["item"].item_id not in done]
    t0 = time.time()
    for n, c in enumerate(todo, 1):
        it, response = c["item"], c["response"]
        spans = segment(response)
        if not spans:
            continue
        probe = it.probes[0]
        ref_hit = any(_contains(response, f) for f in probe.answers)
        own = own_answer(reader, it, response, c["labels"])
        if own:
            forms = text_answer_forms(list(probe.answers) + [own])
            it = Item(it.item_id, it.document, it.request,
                      [Probe(probe.question, forms, probe.fact_id, True, probe.value)], it.gold_minimal,
                      it.knobs, it.meta)
        res = dry_score(reader, it, response, mode="loo", spans=spans, abs_tol=abs_tol)
        pf = dry_score(reader, it, response, mode="loo", spans=spans, abs_tol=abs_tol, with_kappa=False,
                       classify=False, loo_skip=False)
        rec = c["record"]
        row = {"case": it.item_id, "subset": rec["subset"], "model": rec["model"], "qid": rec["qid"],
               "request": it.request, "answers": it.probes[0].answers, "own_answer": own,
               "answer_source": "reference" if ref_hit else ("own answer" if own else "none"),
               "status": res.status, "info": res.info,
               "W": res.W, "W_filler": res.W_filler, "W_misleading": res.W_misleading, "tokens": res.tokens,
               "facts": res.facts, "texts": [s.text for s in spans], "chars": [len(s.text) for s in spans],
               "kept": [s.kept for s in res.spans], "kinds": [s.kind for s in res.spans],
               "candidate": [s.candidate for s in res.spans], "loo_cost": [s.loo_cost for s in res.spans],
               "prefix_status": pf.status, "prefix_kept": [s.kept for s in pf.spans],
               "human": fragment_labels(spans, c["labels"]), "resp_chars": length_chars(response),
               "document": it.document, "response": response, "source_label": it.meta.get("doc_label"),
               "cut_chars": c.get("cut_chars", 0),
               "yap_excess": yap_excess_chars(response, it.gold_minimal), "stopword_share": stopword_share(response),
               "added_answers": res.added_answers}
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not quiet and (n % 25 == 0 or n == len(todo)):
            print(f"labelled {n}/{len(todo)} [{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    summary = summarize_human(rows)
    summary["config"] = {"abs_tol": abs_tol, "cases": len(cases)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _shares(r: dict) -> tuple[float, float]:
    """(deleted share, human non-minimal share) of the labelled characters of a response."""
    lab = [(c, h, k) for c, h, k in zip(r["chars"], r["human"], r["kept"]) if h != "NONE"]
    tot = sum(c for c, _, _ in lab)
    if not tot:
        return float("nan"), float("nan")
    return (sum(c for c, _, k in lab if not k) / tot, sum(c for c, h, _ in lab if h != "MINANS") / tot)


def _agreement(rows: Sequence[dict]) -> dict:
    frag = [(h, k, kind, c) for r in rows for h, k, kind, c in zip(r["human"], r["kept"], r["kinds"], r["chars"])
            if h != "NONE"]
    if not frag:
        return {"n_responses": len(rows)}
    y_true = [h != "MINANS" for h, *_ in frag]
    y_pred = [not k for _, k, *_ in frag]
    ours, human = zip(*[_shares(r) for r in rows]) if rows else ((), ())
    with_min = [r for r in rows if "MINANS" in r["human"]]
    return {"n_responses": len(rows), "fragments": _binary(y_true, y_pred),
            "characters": _binary(y_true, y_pred, [c for *_, c in frag]),
            "share_vs_human": _corr(ours, human),
            "share_mae": float(np.nanmean(np.abs(np.array(ours) - np.array(human)))),
            "deleted_share_mean": float(np.nanmean(ours)), "human_share_mean": float(np.nanmean(human)),
            "minimal_answer_kept": float(np.mean([any(k for h, k in zip(r["human"], r["kept"]) if h == "MINANS")
                                                  for r in with_min])) if with_min else float("nan")}


def summarize_human(rows: Sequence[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    out: dict = {"n_responses": len(rows), "n_scored_ok": len(ok),
                 "status": dict(Counter(r.get("status") for r in rows)),
                 "answer_source": dict(Counter(r.get("answer_source", "reference") for r in rows)),
                 "scored_by_answer_source": dict(Counter(r.get("answer_source", "reference") for r in ok)),
                 "cut_responses": sum(1 for r in rows if r.get("cut_chars", 0) > 20)}
    if not ok:
        return out
    out["overall"] = _agreement(ok)
    ref = [r for r in ok if r.get("answer_source", "reference") == "reference"]
    out["overall_reference_answer"] = _agreement(ref) if ref else {"n_responses": 0}
    prefix = [dict(r, kept=r["prefix_kept"]) for r in ok if r.get("prefix_status") == "ok"]
    out["overall_prefix"] = _agreement(prefix) if prefix else {"n_responses": 0}
    # why fragments the annotators marked as not minimal stayed (as in the planted check)
    kept_nm = [c for r in ok if r.get("candidate") for h, k, c in zip(r["human"], r["kept"], r["candidate"])
               if h not in ("MINANS", "NONE") and k]
    if kept_nm:
        out["kept_not_minimal"] = {"fragments": len(kept_nm),
                                   "not_candidate": float(np.mean([not c for c in kept_nm])),
                                   "blocked": float(np.mean([bool(c) for c in kept_nm]))}
    by_label: dict = {}
    for r in ok:
        for h, k, kind in zip(r["human"], r["kept"], r["kinds"]):
            d = by_label.setdefault(h, Counter())
            d["fragments"] += 1
            d["deleted"] += not k
            d[kind] += 1
    out["by_label"] = {h: {"fragments": d["fragments"], "deleted": d["deleted"] / d["fragments"],
                           "uninformative": d["uninformative"] / d["fragments"], "duplicate": d["duplicate"] / d["fragments"],
                           "wrong": d["misleading"] / d["fragments"]} for h, d in sorted(by_label.items())}
    out["by_subset"] = {s: _agreement([r for r in ok if r["subset"] == s]) for s in sorted({r["subset"] for r in ok})}
    human = [_shares(r)[1] for r in ok]
    out["baselines_vs_human"] = {
        "removable share (filler map)": _corr([_shares(r)[0] for r in ok], human),
        "length": _corr([r["resp_chars"] for r in ok], human),
        "Yap excess (chars)": _corr([r["yap_excess"] for r in ok], human),
        "stop-word share": _corr([r["stopword_share"] for r in ok], human)}
    models: dict = {}
    for r in ok:
        ours_s, human_s = _shares(r)
        models.setdefault(r["model"], []).append((ours_s, human_s))
    per = {m: {"n": len(v), "deleted": float(np.nanmean([a for a, _ in v])), "human": float(np.nanmean([b for _, b in v]))}
           for m, v in sorted(models.items())}
    out["by_model"] = per
    big = {m: v for m, v in per.items() if v["n"] >= 5}
    if len(big) >= 4:
        from scipy.stats import kendalltau

        tau = kendalltau([v["deleted"] for v in big.values()], [v["human"] for v in big.values()])
        out["model_ranking_kendall_tau"] = {"tau": float(tau[0]), "p": float(tau[1]), "models": len(big)}
    return out
