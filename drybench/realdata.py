"""Real-text items: SQuAD 2.0, MuSiQue, CUAD and Poddar et al. mapped onto the benchmark's knobs.

The synthetic generator guarantees that the reader cannot know an answer without the
response (all names are invented). On real text this has to be enforced: `filter_items`
(needs a reader) drops items with a fact the reader answers from an empty message, or one it
cannot read from the gold answer. Items are built deterministically from the dataset files
(`build_real_items`, oversampled per cell), then filtered to n per cell.

SQuAD 2.0 (CC BY-SA 4.0; dev set)
  One Wikipedia paragraph with its article title. k = 1..4 questions about it, asked in one
  request (numbered when k > 1). answerable=False puts one of SQuAD 2.0's unanswerable
  questions among them (written to look answerable; a plausible wrong answer is in the text);
  the ideal response says it is not stated.

MuSiQue (CC BY 4.0; dev set)
  2-4 hop questions composed from single-hop ones, over the supporting paragraphs plus
  `distractors` of MuSiQue's own (hard, retrieved) distractor paragraphs. explain=True adds
  "Explain how you found it." and makes every intermediate answer a required fact, as the
  carrier is for synthetic 2-hop items. answerable=False takes MuSiQue-Full's unanswerable
  version of a 2-hop question (a supporting paragraph removed).

CUAD (CC BY 4.0) contract excerpts with short facts and Poddar et al.'s passage questions (GPL-3.0)
are built in drybench.cuad and drybench.poddar.

Accepted answers are the dataset's gold texts plus light variants (without a leading article
or preposition, capitalised, without thousands separators), reduced to a prefix-free set.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from drybench.items import NOT_STATED
from drybench.types import Item, Probe, read_jsonl

SOURCES = ("squad2", "musique", "cuad", "poddar")
NOT_STATED_FORMS = [NOT_STATED, "Not stated"]
MAX_ANSWER_WORDS = 5

SQUAD_CELLS: list[dict[str, Any]] = [
    {"k": 1}, {"k": 2}, {"k": 3}, {"k": 4},
    {"k": 1, "answerable": False}, {"k": 3, "answerable": False},
]
MUSIQUE_CELLS: list[dict[str, Any]] = [
    {"hops": 2}, {"hops": 2, "explain": True},
    {"hops": 3}, {"hops": 3, "explain": True},
    {"hops": 4}, {"hops": 4, "explain": True},
    {"hops": 2, "answerable": False},
]

_LEAD_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.I)
_LEAD_PREP = re.compile(r"^(?:in|on|at|by|from|during|since|of|to|for|about|around|approximately)\s+", re.I)
_EDGE_PUNCT = ".,;:!?\"'“”‘’()[]"


def _norm_space(s: Any) -> str:
    return " ".join(str(s or "").split())


def _norm(s: str) -> str:
    """Comparison key: lowercase, no punctuation, no articles."""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def _contains(text: str, phrase: str) -> bool:
    phrase = phrase.strip().lower()
    return bool(phrase) and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text.lower()) is not None


def _as_question(q: str) -> str:
    q = _norm_space(q)
    return q if q.endswith("?") else q.rstrip(".") + "?"


def _end(s: str) -> str:
    return s if s.endswith((".", "!", "?")) else s + "."


_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _number_variant(text: str) -> str | None:
    """'six years' <-> '6 years' (a response may write the number either way; the reader copies it)."""
    first, _, rest = text.partition(" ")
    tail = f" {rest}" if rest else ""
    if first.lower() in _NUMBER_WORDS:
        return str(_NUMBER_WORDS.index(first.lower())) + tail
    if first.isdigit() and str(int(first)) == first and int(first) < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[int(first)] + tail
    return None


def text_answer_forms(texts: Iterable[str]) -> list[str]:
    """Prefix-free accepted forms for free-text gold answers (shortest forms win: "Denver"
    covers "Denver Broncos"). Non-numeric forms shorter than two characters are dropped."""
    cands: list[str] = []
    for t in texts:
        t = _norm_space(t)
        if not t:
            continue
        base = [t, t.strip(_EDGE_PUNCT).strip()]
        for b in list(base):
            for rx in (_LEAD_ARTICLE, _LEAD_PREP):
                s = rx.sub("", b)
                if s != b:
                    base.append(s)
        base += [re.sub(r"(?<=\d),(?=\d{3}\b)", "", b) for b in base]  # 1,000 -> 1000
        base += [v for v in map(_number_variant, base) if v]
        for b in base:
            if b:
                cands += [b, b[:1].upper() + b[1:]]
                if re.match(r"(?:The|A|An) ", b):  # a reader copying "the X" from mid-sentence
                    cands.append(b[:1].lower() + b[1:])
    cands = [c for c in dict.fromkeys(cands) if len(c) >= 2 or c.isdigit()]
    out: list[str] = []
    for c in sorted(cands, key=len):
        if not any(c.startswith(o) for o in out):
            out.append(c)
    return out


# ---------------------------------------------------------------- SQuAD 2.0

def load_squad(path: str | Path) -> list[dict]:
    """Paragraphs of a SQuAD 2.0 file: {pid, title, context, qas}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    paras = []
    for art in data["data"]:
        for pi, para in enumerate(art["paragraphs"]):
            paras.append({"pid": f"{art['title']}#{pi}", "title": art["title"].replace("_", " "),
                          "context": para["context"], "qas": para["qas"]})
    return paras


def _squad_fact(qa: dict) -> dict | None:
    """An answerable question with a short gold answer that the question does not give away."""
    if qa.get("is_impossible"):
        return None
    texts = [_norm_space(a.get("text")) for a in qa.get("answers", []) if _norm_space(a.get("text"))]
    if not texts:
        return None
    counts = Counter(texts)  # canonical value: the span most annotators chose, then the shortest
    value = min(counts, key=lambda t: (-counts[t], len(t)))
    q = _as_question(qa["question"])
    if len(value.split()) > MAX_ANSWER_WORDS or len(q.split()) < 4 or _contains(q, value):
        return None
    forms = text_answer_forms(texts)
    return {"qid": qa["id"], "question": q, "value": value, "forms": forms} if forms else None


def _squad_missing(qa: dict) -> dict | None:
    if not qa.get("is_impossible"):
        return None
    q = _as_question(qa["question"])
    if len(q.split()) < 4:
        return None
    plausible = [_norm_space(a.get("text")) for a in qa.get("plausible_answers", []) if a.get("text")]
    return {"qid": qa["id"], "question": q, "value": NOT_STATED, "forms": list(NOT_STATED_FORMS),
            "plausible": sorted(set(plausible))}


def _compatible(x: dict, chosen: Sequence[dict]) -> bool:
    """Distinct questions and answers, and no answer spelled out in another question."""
    for y in chosen:
        if _norm(x["question"]) == _norm(y["question"]):
            return False
        if x["value"] != NOT_STATED and y["value"] != NOT_STATED and _norm(x["value"]) == _norm(y["value"]):
            return False
        if x["value"] != NOT_STATED and _contains(y["question"], x["value"]):
            return False
        if y["value"] != NOT_STATED and _contains(x["question"], y["value"]):
            return False
    return True


def _pick_squad(para: dict, k: int, answerable: bool, rng: random.Random) -> list[dict] | None:
    """k mutually compatible questions (one of them unanswerable when answerable=False)."""
    facts = [x for x in map(_squad_fact, para["qas"]) if x]
    rng.shuffle(facts)
    starts: list[list[dict]] = [[]]
    if not answerable:
        starts = [[m] for m in map(_squad_missing, para["qas"]) if m]
        rng.shuffle(starts)
    for chosen in starts:
        chosen = list(chosen)
        for x in facts:
            if len(chosen) == k:
                break
            if _compatible(x, chosen):
                chosen.append(x)
        if len(chosen) == k:
            if not answerable:  # the unanswerable question at a random position
                m = chosen.pop(0)
                chosen.insert(rng.randrange(len(chosen) + 1), m)
            return chosen
    return None


def _squad_item(para: dict, chosen: list[dict], k: int, answerable: bool, ci: int, j: int) -> Item:
    questions = [x["question"] for x in chosen]
    if k == 1:
        request = questions[0]
    else:
        request = "Answer these questions about the text:\n" + "\n".join(
            f"{i}. {q}" for i, q in enumerate(questions, 1))
    probes = [Probe(question=x["question"], answers=list(x["forms"]), fact_id=f"squad2:{x['qid']}",
                    value=x["value"]) for x in chosen]
    vals = ["Not stated in the text" if x["value"] == NOT_STATED else x["value"] for x in chosen]
    gold = _end(vals[0]) if k == 1 else "\n".join(f"{i}. {_end(v)}" for i, v in enumerate(vals, 1))
    missing = [x for x in chosen if x["value"] == NOT_STATED]
    meta = {"source": "squad2", "cell": ci, "doc_label": "Text", "doc_noun": "text",
            "title": para["title"], "paragraph": para["pid"], "qids": [x["qid"] for x in chosen],
            "missing": missing[0]["qid"] if missing else None,
            "plausible_answers": missing[0]["plausible"] if missing else []}
    knobs = {"k": k, "d": 0, "answerable": answerable, "ambiguous": False, "hops": 1, "explain": False,
             "source": "squad2"}
    document = f"{para['title']}\n\n{_norm_space(para['context'])}"
    return Item(f"sq{ci:02d}-{j:04d}", document, request, probes, gold, knobs, meta)


def squad_items(paras: Sequence[dict], n_per_cell: int, seed: int = 0,
                cells: Sequence[dict[str, Any]] = SQUAD_CELLS) -> tuple[list[Item], dict]:
    """Up to n_per_cell items per cell, each cell from distinct paragraphs (deterministic)."""
    items, stats = [], {}
    for ci, cell in enumerate(cells):
        k, answerable = int(cell.get("k", 1)), bool(cell.get("answerable", True))
        rng = random.Random(f"{seed}-squad2-{ci}")
        order = list(range(len(paras)))
        rng.shuffle(order)
        j = 0
        for pi in order:
            if j >= n_per_cell:
                break
            chosen = _pick_squad(paras[pi], k, answerable, rng)
            if chosen is not None:
                items.append(_squad_item(paras[pi], chosen, k, answerable, ci, j))
                j += 1
        stats[f"squad2:{ci}"] = {"cell": cell, "built": j, "wanted": n_per_cell}
    return items, stats


# ---------------------------------------------------------------- MuSiQue

def load_musique(paths: str | Path | Sequence[str | Path]) -> list[dict]:
    """MuSiQue instances (official jsonl format); duplicates across files are dropped."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    seen, out = set(), []
    for p in paths:
        for inst in read_jsonl(p):
            key = (inst["id"], bool(inst.get("answerable", True)), inst["question"])
            if key not in seen:
                seen.add(key)
                out.append(inst)
    return out


def musique_hops(inst: dict) -> int:
    m = re.match(r"(\d)hop", str(inst.get("id", "")))
    return int(m.group(1)) if m else len(inst.get("question_decomposition") or [])


def _fill(question: str, answers: Sequence[str]) -> str | None:
    """A decomposition step as a self-contained question: '#k' -> answer k; 'X >> relation'
    (knowledge-base steps) -> 'What is the relation of X?'."""
    q = _norm_space(question)
    for k in sorted({int(m) for m in re.findall(r"#(\d+)", q)}, reverse=True):
        if not 1 <= k <= len(answers):
            return None
        q = re.sub(rf"#{k}(?!\d)", answers[k - 1], q)
    if ">>" in q:
        subj, _, rel = q.partition(">>")
        subj, rel = subj.strip(), rel.strip().rstrip("?").strip()
        if not subj or not rel or ">>" in rel:
            return None
        q = f"What is the {rel} of {subj}?"
    return _as_question(q)


def _musique_item(inst: dict, hops: int, explain: bool, answerable: bool, distractors: int,
                  rng: random.Random, ci: int, j: int) -> Item | None:
    dec = inst.get("question_decomposition") or []
    if len(dec) != hops:
        return None
    answers = [_norm_space(x.get("answer")) for x in dec]
    if not all(answers) or any(len(a.split()) > MAX_ANSWER_WORDS + 1 for a in answers):
        return None
    subqs = [_fill(x.get("question", ""), answers) for x in dec]
    if any(q is None for q in subqs):
        return None
    question = _as_question(inst["question"])
    request = question + (" Explain how you found it." if explain else "")
    final = _norm_space(inst.get("answer")) or answers[-1]
    if answerable:
        if len(final.split()) > MAX_ANSWER_WORDS + 1 or _contains(request, final):
            return None
        if explain and any(_contains(request, a) for a in answers[:-1]):
            return None
    paras = inst.get("paragraphs") or []
    sup = [p for p in paras if p.get("is_supporting")]
    dis = [p for p in paras if not p.get("is_supporting")]
    if not sup or len(dis) < distractors:
        return None
    chosen = sup + rng.sample(dis, distractors)
    rng.shuffle(chosen)
    document = "\n\n".join(f"Document {i}: {_norm_space(p.get('title'))}\n{_norm_space(p.get('paragraph_text'))}"
                           for i, p in enumerate(chosen, 1))
    probes: list[Probe] = []
    if answerable:
        for s, (q, a) in enumerate(zip(subqs[:-1], answers[:-1]), 1):
            forms = text_answer_forms([a])
            if not forms:
                return None
            probes.append(Probe(question=q, answers=forms, fact_id=f"musique:{inst['id']}:step{s}",
                                required=explain, value=a))
        forms = text_answer_forms([final] + [a for a in inst.get("answer_aliases") or [] if a])
        if not forms:
            return None
        probes.append(Probe(question=question, answers=forms, fact_id=f"musique:{inst['id']}:final",
                            value=final))
        if explain:  # one sentence per step ("question: answer."), so no step splits into two spans
            gold = " ".join(f"{q.rstrip('?')}: {_end(a)}" for q, a in zip(subqs[:-1], answers[:-1])) + \
                f" So the answer is {_end(final)}"
        else:
            gold = _end(final)
    else:
        probes.append(Probe(question=question, answers=list(NOT_STATED_FORMS),
                            fact_id=f"musique:{inst['id']}:final", value=NOT_STATED))
        gold = "Not stated in the documents."
    knobs = {"k": 1, "d": distractors, "answerable": answerable, "ambiguous": False, "hops": hops,
             "explain": explain, "source": "musique"}
    meta = {"source": "musique", "cell": ci, "doc_label": "Documents", "doc_noun": "documents",
            "musique_id": inst["id"], "decomposition": [{"question": q, "answer": a} for q, a in zip(subqs, answers)],
            "missing": None if answerable else "final"}
    return Item(f"mq{ci:02d}-{j:04d}", document, request, probes, gold, knobs, meta)


def musique_items(instances: Sequence[dict], n_per_cell: int, seed: int = 0, distractors: int = 3,
                  cells: Sequence[dict[str, Any]] = MUSIQUE_CELLS) -> tuple[list[Item], dict]:
    """Up to n_per_cell items per cell; every question is used in at most one cell."""
    items, stats, used = [], {}, set()
    for ci, cell in enumerate(cells):
        hops = int(cell.get("hops", 2))
        explain, answerable = bool(cell.get("explain", False)), bool(cell.get("answerable", True))
        if explain and not answerable:
            raise ValueError("unanswerable MuSiQue cells cannot ask for an explanation")
        pool = [i for i, inst in enumerate(instances)
                if musique_hops(inst) == hops and bool(inst.get("answerable", True)) == answerable]
        rng = random.Random(f"{seed}-musique-{ci}")
        rng.shuffle(pool)
        j = 0
        for i in pool:
            if j >= n_per_cell:
                break
            key = _norm(instances[i]["question"])
            if key in used:
                continue
            it = _musique_item(instances[i], hops, explain, answerable, distractors, rng, ci, j)
            if it is not None:
                items.append(it)
                used.add(key)
                j += 1
        stats[f"musique:{ci}"] = {"cell": cell, "built": j, "wanted": n_per_cell, "available": len(pool)}
    return items, stats


# ---------------------------------------------------------------- building and filtering

def source_key(item: Item) -> str:
    """What an item is made from — a SQuAD paragraph, a MuSiQue question, a CUAD contract, a Poddar et al.
    question — so that a held-out run can leave out every item built from text a development run used."""
    m, src = item.meta, item.meta.get("source") or item.knobs.get("source", "synthetic")
    if src == "squad2":
        return f"squad2:{m.get('paragraph') or m.get('title')}"
    if src == "musique":
        return f"musique:{m.get('musique_id')}"
    if src == "cuad":
        return f"cuad:{m.get('contract')}"
    if src == "poddar":
        return f"poddar:{m.get('subset')}:{m.get('qid')}"
    return f"{src}:{item.item_id}"


def build_real_items(source: str, paths: str | Path | Sequence[str | Path], n_per_cell: int,
                     seed: int = 0, distractors: int = 3) -> tuple[list[Item], dict]:
    """Deterministic pool of items for one source (oversample: filter_items keeps n per cell)."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if source == "squad2":
        paras = [p for path in paths for p in load_squad(path)]
        return squad_items(paras, n_per_cell, seed=seed)
    if source == "musique":
        instances = load_musique(paths)
        cells = list(MUSIQUE_CELLS)
        if not any(not inst.get("answerable", True) for inst in instances):
            cells = [c for c in cells if c.get("answerable", True)]  # MuSiQue-Ans only
        return musique_items(instances, n_per_cell, seed=seed, distractors=distractors, cells=cells)
    if source == "cuad":
        from drybench.cuad import cuad_items, load_cuad

        return cuad_items([c for path in paths for c in load_cuad(path)], n_per_cell, seed=seed)
    if source == "poddar":
        from drybench.poddar import load_poddar, poddar_items

        return poddar_items([r for path in paths for r in load_poddar(path)], n_per_cell, seed=seed)
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def cell_key(item: Item) -> str:
    return f"{item.meta.get('source', 'synthetic')}:{item.meta.get('cell', '?')}"


def filter_items(reader, items: Sequence[Item], n_per_cell: int | None = None, max_prior: float = 0.05,
                 min_gold_p: float = 0.5, keep: Sequence[Item] = ()) -> tuple[list[Item], dict]:
    """Keep items whose facts the reader cannot answer without a response (prior < max_prior)
    and can read from the gold answer (p >= min_gold_p for every required probe).

    Items are taken in pool order, n_per_cell per cell at most. 'not stated' probes are exempt
    from the prior test: a reader that says 'not stated' to an empty message knows nothing.
    `keep`: items kept by an earlier filter (their responses exist); they stay as they are and
    count towards their cell, which is topped up from the pool."""
    keep_ids = {it.item_id for it in keep}
    groups: "OrderedDict[str, list[Item]]" = OrderedDict()
    for it in items:
        groups.setdefault(cell_key(it), []).append(it)
    before: dict[str, list[Item]] = {}
    for it in keep:
        before.setdefault(cell_key(it), []).append(it)
        groups.setdefault(cell_key(it), [])
    kept, stats = [], {}
    for key, its in groups.items():
        prev = before.get(key, [])
        kept += prev
        st = {"pool": len(its), "checked": 0, "known": 0, "unreadable": 0, "kept": len(prev),
              "kept_before": len(prev), "wanted": n_per_cell}
        for it in its:
            if n_per_cell is not None and st["kept"] >= n_per_cell:
                break
            if it.item_id in keep_ids:
                continue
            st["checked"] += 1
            probes = it.required_probes
            facts = [p for p in probes if NOT_STATED not in p.answers]
            prior_max = None
            if facts:
                prior = np.exp(reader.prior_logprobs(it.request, facts))
                prior_max = float(prior.max())
                if prior_max >= max_prior:
                    st["known"] += 1
                    continue
            p_gold = np.exp(reader.probe_logprobs(it.request, [it.gold_minimal], probes)[0])
            if float(p_gold.min()) < min_gold_p:
                st["unreadable"] += 1
                continue
            it.meta["filter"] = {"prior_max": prior_max, "gold_p_min": float(p_gold.min())}
            kept.append(it)
            st["kept"] += 1
        stats[key] = st
    return kept, stats
