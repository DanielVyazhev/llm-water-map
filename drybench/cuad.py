"""CUAD v1 contracts as benchmark items (CC BY 4.0, The Atticus Project).

CUAD labels 41 clause categories in 510 commercial contracts. Most labelled answers are whole
clauses, which a reader cannot be asked to copy as a short phrase; four categories carry a
short value, and only those become facts:

  title  'Document Name' as labelled                 "WEB SITE HOSTING AGREEMENT"
  date   'Agreement Date', parsed into a calendar date "April 6, 1999" (any usual spelling)
  law    the jurisdiction named in 'Governing Law'    "New York"
  term   the initial term in 'Expiration Date'        "five (5) years"

A whole contract is too long for a small model (median ~5,000 words), so an item shows an
excerpt: the opening of the contract (title, parties, date), the paragraphs that state the
requested facts and `d` other paragraphs of the same contract (distractors), in document order,
with "[...]" where text is left out. Distractors never come from paragraphs CUAD labels as a
date, term or governing-law clause, so every requested fact is stated once. answerable=False asks
for the governing law of an excerpt without the governing-law clause (incorporation statements
such as "a Delaware corporation" may remain: a plausible wrong answer, as in SQuAD 2.0).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from drybench.items import NOT_STATED
from drybench.realdata import NOT_STATED_FORMS, _end, _norm_space
from drybench.segment import segment
from drybench.types import Item, Probe

CUAD_FACTS = ("title", "date", "law", "term")
CUAD_QUESTIONS = {
    "title": "What is the title of this agreement?",
    "date": "What is the date of this agreement?",
    "law": "Which jurisdiction's law governs this agreement?",
    "term": "How long is the initial term of this agreement?",
}
CUAD_CELLS: list[dict[str, Any]] = [
    {"k": 1}, {"k": 2}, {"k": 3},
    {"k": 1, "long": True},
    {"k": 2, "answerable": False},   # the governing law (absent) and one present fact
]
DISTRACTORS, LONG_DISTRACTORS = 2, 8
MAX_WORDS, LONG_MAX_WORDS = 900, 1800
OPENING_WORDS, BLOCK_WORDS, MIN_DISTRACTOR_WORDS = 150, 120, 25
GAP = "[...]"
# CUAD categories whose paragraphs never serve as distractors (they state or restate facts)
FACT_CATEGORIES = ("Document Name", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
                   "Notice Period To Terminate Renewal", "Governing Law")
GOVERNING = re.compile(r"\bgovern(?:ed|ing|s)\b|\bconstrued\b|\bgoverning\s+law\b|\bchoice\s+of\s+law\b", re.I)

# ---------------------------------------------------------------- values

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
_MONTH_NUM = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MONTH_NUM.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
_MONTH_NUM["sept"] = 9
_MON = r"(" + "|".join(sorted(_MONTH_NUM, key=len, reverse=True)) + r")\.?"
_DAY = r"(\d{1,2})\s*(?:st|nd|rd|th|t\s?h)?"
_DATE_MDY = re.compile(rf"\b{_MON}\s+{_DAY}\s*,?\s*(\d{{4}})\b", re.I)                       # April 6, 1999
_DATE_DMY = re.compile(rf"\b{_DAY}\s+(?:day\s+)?(?:of\s+)?{_MON}\s*,?\s*(\d{{4}})\b", re.I)    # 6th day of April, 1999


def find_date(text: str) -> tuple[tuple[int, int, int], str] | None:
    """((year, month, day), the date as written) for the first fully written date in `text`."""
    s = _norm_space(text)
    hits = []
    for rx, order in ((_DATE_MDY, "mdy"), (_DATE_DMY, "dmy")):
        m = rx.search(s)
        if m:
            a, b, y = m.group(1), m.group(2), int(m.group(3))
            mon, day = (_MONTH_NUM[a.lower()], int(b)) if order == "mdy" else (_MONTH_NUM[b.lower()], int(a))
            hits.append((m.start(), (y, mon, day), m.group(0)))
    if not hits:
        return None
    _, (y, mon, day), written = min(hits)
    return ((y, mon, day), written) if 1 <= day <= 31 and 1950 <= y <= 2030 else None


def parse_date(text: str) -> tuple[int, int, int] | None:
    """(year, month, day) of the first fully written date in `text`, or None."""
    got = find_date(text)
    return got[0] if got else None


def _ordinal(d: int) -> str:
    return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def _prefix_free(forms: Sequence[str]) -> list[str]:
    """Distinct non-empty forms, none a prefix of another (the shorter one covers it)."""
    out: list[str] = []
    for f in sorted(dict.fromkeys(f for f in forms if f), key=len):
        if not any(f.startswith(o) for o in out):
            out.append(f)
    return out


def date_forms(date: tuple[int, int, int], written: str | None = None) -> list[str]:
    """Accepted spellings of a date: as written in the contract plus the usual rewrites (every
    alias costs one reader pass per message, so only the common ones)."""
    y, m, d = date
    mon, o = MONTHS[m - 1], _ordinal(d)
    forms = [f"{mon} {d}, {y}", f"{mon} {o}, {y}", f"{d} {mon} {y}", f"{o} day of {mon}, {y}", f"{m}/{d}/{y}",
             f"{y}-{m:02d}-{d:02d}", f"{mon.upper()} {d}, {y}"]
    if written:  # the date exactly as the contract writes it ("6th day of April, 1999", "JUNE 23, 2015")
        forms.insert(0, _norm_space(written).strip(" ,.;:()"))
    return _prefix_free(forms)


US_STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
             "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
             "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
             "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
             "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
             "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
             "West Virginia", "Wisconsin", "Wyoming", "District of Columbia"]
OTHER_JURISDICTIONS = [
    "Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba", "Nova Scotia", "New Brunswick", "Saskatchewan",
    "England and Wales", "England", "Scotland", "Northern Ireland", "Ireland", "United Kingdom",
    "United States of America", "United States", "Canada", "Israel", "Japan", "People's Republic of China", "PRC",
    "China", "Hong Kong", "Singapore", "India", "Federal Republic of Germany", "Germany", "France", "Switzerland",
    "Netherlands", "Sweden", "Denmark", "Norway", "Finland", "Italy", "Spain", "Australia", "New South Wales",
    "Victoria", "Republic of Korea", "Korea", "Taiwan", "Cayman Islands", "Bermuda", "Luxembourg", "Belgium",
    "Austria", "Brazil", "Mexico"]
DEMONYMS = {"English": "England", "Israeli": "Israel", "German": "Germany", "Swiss": "Switzerland",
            "French": "France", "Dutch": "Netherlands", "Irish": "Ireland", "Scottish": "Scotland",
            "Chinese": "China", "Japanese": "Japan", "Korean": "Korea", "Indian": "India", "Canadian": "Canada",
            "Australian": "Australia", "Swedish": "Sweden", "Italian": "Italy", "Spanish": "Spain"}
_JURIS = sorted(US_STATES + OTHER_JURISDICTIONS, key=len, reverse=True)
_JURIS_RX = re.compile(r"\b(" + "|".join(re.escape(j).replace("'", "['’]?") for j in _JURIS) + r")\b", re.I)
_LAW_BEFORE = re.compile(r"\blaws?\b[^.;]{0,40}$", re.I)
_DEMONYM_LAW = re.compile(r"\b(" + "|".join(DEMONYMS) + r"|" + "|".join(re.escape(j) for j in _JURIS) +
                          r")\s+law\b", re.I)


def _canonical(name: str) -> str:
    low = name.lower().replace("’", "'").replace("peoples", "people's")
    for j in _JURIS:
        if j.lower() == low:
            return j
    return name


_JOINED = re.compile(r"^\W*(?:and|or)\s+(?:the\s+)?(?:(?:State|Commonwealth|Province)\s+of\s+)?$", re.I)
_NATIONAL = {"United States", "United States of America"}


def governing_law(clauses: Sequence[str]) -> tuple[str, str, str] | None:
    """(jurisdiction, name as written, phrase) of the first governing-law clause that names one:
    a jurisdiction within a few words after "law(s)" ("the laws of the State of New York"), or
    "<jurisdiction or demonym> law" ("English law"). Two jurisdictions joined by "and"/"or": a
    state wins over "United States"; any other pair is ambiguous and the clause is skipped."""
    for c in clauses:
        c = _norm_space(c)
        best, ambiguous = None, False
        for m in _JURIS_RX.finditer(c):
            lm = _LAW_BEFORE.search(c[: m.start()])
            if not lm:
                continue
            name, written, end = _canonical(m.group(1)), m.group(1), m.end()
            for m2 in _JURIS_RX.finditer(c, m.end()):
                if not _JOINED.match(c[end: m2.start()]):
                    break
                other = _canonical(m2.group(1))
                if name in _NATIONAL and other not in _NATIONAL:
                    name, written, end = other, m2.group(1), m2.end()
                elif other not in _NATIONAL and other != name:
                    ambiguous = True
                break
            best = (lm.start(), name, written, c[lm.start(): end])
            break
        m3 = _DEMONYM_LAW.search(c)
        if m3 and (best is None or m3.start() < best[0]):
            word = m3.group(1)
            name = DEMONYMS.get(word.capitalize()) or _canonical(word)
            best, ambiguous = (m3.start(), name, m3.group(0), m3.group(0)), False
        if best and not ambiguous:
            return best[1], best[2], best[3]
    return None


def law_forms(name: str, written: str | None = None) -> list[str]:
    forms = [name, f"State of {name}", f"the State of {name}", f"the laws of the State of {name}"] \
        if name in US_STATES else [name, f"the laws of {name}"]
    forms += [f"{dem} law" for dem, n in DEMONYMS.items() if n == name]
    if name in ("People's Republic of China", "PRC"):
        forms += ["China", "PRC"]
    if written and written != name:
        forms.append(written)
    forms.append(name.upper())  # clauses in capitals ("THE LAWS OF THE STATE OF NEW YORK")
    return _prefix_free(forms)


_NUMW = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
         "seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety")
_DUR = (rf"(?:(?:{_NUMW})(?:[- ](?:{_NUMW}))?(?:\s*\(\d+\))?|\d+)\s+(?:calendar\s+|consecutive\s+|full\s+)?"
        r"(?:years?|months?)")
_LEAD = r"(?:a\s+period\s+of\s+)?(?:an?\s+(?:initial\s+)?term\s+of\s+)?"
_TERM = re.compile(
    rf"(?:(?:the\s+)?(?:initial\s+)?term\s+of\s+this\s+agreement(?:\s+\([^)]{{0,30}}\))?\s+(?:shall\s+be|is|will\s+be)\s+"
    rf"(?:for\s+)?{_LEAD}"
    rf"|(?:the\s+)?initial\s+term(?:\s+\([^)]{{0,30}}\))?\s+(?:shall\s+be|is|will\s+be|of)\s+(?:for\s+)?{_LEAD}"
    rf"|agreement(?:\s+\([^)]{{0,30}}\))?\s+(?:shall|will)\s+(?:commence[^.;]{{0,100}}?\s+and\s+(?:shall\s+|will\s+)?)?"
    rf"(?:remain\s+in\s+(?:full\s+force\s+and\s+)?effect|continue(?:\s+in\s+(?:full\s+force\s+and\s+)?effect)?"
    rf"|be\s+in\s+effect|be\s+valid|extend)\s+for\s+{_LEAD}"
    rf"|agreement(?:\s+\([^)]{{0,30}}\))?\s+(?:shall|will)\s+have\s+an?\s+(?:initial\s+)?term\s+of\s+"
    rf"|agreement[^.;]{{0,100}}?\s+for\s+an?\s+(?:initial\s+)?term\s+of\s+)({_DUR})\b", re.I)
_DUR_RX = re.compile(rf"\b{_DUR}\b", re.I)
# a duration right after these words is a renewal or extension period, not the initial term
_RENEWAL = re.compile(r"(?:renew\w*|successive|additional|extension|extend\w*|thereafter)\W+(?:[\w()]+\W+){0,5}$", re.I)
_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_NUM.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
             "ninety": 90})


def _duration_key(text: str) -> tuple[int, str] | None:
    """('five (5) years' | 'five years' | '5 years') -> (5, 'year')."""
    t = _norm_space(text).lower()
    unit = "month" if "month" in t else "year"
    num = re.search(r"\((\d+)\)", t) or re.match(r"(\d+)", t)
    if num:
        return int(num.group(1)), unit
    words = re.findall(r"[a-z]+", t.split("year")[0].split("month")[0])
    vals = [_NUM[w] for w in words if w in _NUM]
    return (sum(vals), unit) if vals else None


def initial_term(clauses: Sequence[str]) -> tuple[str, str] | None:
    """(duration, matched phrase) of the first expiration clause that states one fixed initial term;
    a clause naming another duration that is not a renewal period is skipped as ambiguous."""
    for c in clauses:
        c = _norm_space(c)
        m = _TERM.search(c)
        if not m or _RENEWAL.search(c[: m.start(1)]):
            continue
        key = _duration_key(m.group(1))
        others = {_duration_key(d.group(0)) for d in _DUR_RX.finditer(c) if not _RENEWAL.search(c[: d.start()])}
        if key is None or others - {key}:
            continue
        return m.group(1), m.group(0)
    return None


def term_forms(value: str) -> list[str]:
    v = _norm_space(value)
    plain, short = re.sub(r"\s*\(\d+\)", "", v), _short_term(v)
    return _prefix_free([v, plain, short, short.title()] + [f[:1].upper() + f[1:] for f in (v, plain)])


def _short_term(value: str) -> str:
    key = _duration_key(value)
    if key is None:
        return _norm_space(value)
    n, unit = key
    return f"{n} {unit}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------- contracts

def load_cuad(path: str | Path) -> list[dict]:
    """Contracts of CUADv1.json: {cid, context, answers: {category: [(text, start), ...]}}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for art in data["data"]:
        for para in art["paragraphs"]:
            answers: dict[str, list[tuple[str, int]]] = {}
            for qa in para["qas"]:
                cat = qa["id"].split("__")[-1]
                answers[cat] = [(a["text"], int(a["answer_start"])) for a in qa.get("answers", [])]
            out.append({"cid": art["title"], "context": para["context"], "answers": answers})
    return out


def _blocks(context: str) -> list[tuple[int, int]]:
    """Paragraphs: text between blank lines, as (start, end) offsets of the stripped text."""
    out, pos = [], 0
    for m in list(re.finditer(r"\n[ \t\r\f\v\xa0]*\n", context)) + [None]:
        end = m.start() if m else len(context)
        seg = context[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            out.append((pos + lead, pos + lead + len(seg.strip())))
        pos = m.end() if m else end
    return out


def _block_of(blocks: Sequence[tuple[int, int]], offset: int) -> int | None:
    for i, (a, b) in enumerate(blocks):
        if a <= offset < b:
            return i
    return None


def contract_facts(contract: dict) -> dict[str, dict]:
    """The short facts this contract states, each with the paragraph offset that states it."""
    ans = contract["answers"]
    facts: dict[str, dict] = {}
    title = ans.get("Document Name") or []
    if title:
        text, start = title[0]
        value = re.sub(r"\s*[,(]\s*[\"“]?[A-Za-z]{0,2}$", "", _norm_space(text)).strip(" ,.;:")  # 'X AGREEMENT, d'
        value = re.sub(r"\s+(?:is|are|dated|made|by|between|and|of|the|this)$", "", value)  # 'X AGREEMENT is'
        whole = re.search(r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])", _norm_space(contract["context"]), re.I)
        if 2 <= len(value.split()) <= 12 and re.search(r"[A-Za-z]", value) and whole:  # not cut mid-word
            shown = value.title() if value.isupper() else value
            forms = _prefix_free([value, shown, value.title(), value.upper()])
            facts["title"] = {"value": shown, "forms": forms, "start": start, "evidence": value}
    date = ans.get("Agreement Date") or []
    if date:
        text, start = date[0]
        got = find_date(text)
        if got:
            (y, m, day), written = got
            facts["date"] = {"value": f"{MONTHS[m - 1]} {day}, {y}", "forms": date_forms((y, m, day), written),
                             "start": start, "evidence": _norm_space(written), "date": (y, m, day)}
    law = ans.get("Governing Law") or []
    for text, start in law:
        got = governing_law([text])
        if got:
            name, written, phrase = got
            facts["law"] = {"value": name, "forms": law_forms(name, written), "start": start, "evidence": phrase}
            break
    for text, start in ans.get("Expiration Date") or []:
        got = initial_term([text])
        if got:
            value, phrase = got
            facts["term"] = {"value": _short_term(value), "forms": term_forms(value), "start": start,
                             "evidence": phrase}
            break
    for name, f in facts.items():
        f["question"] = CUAD_QUESTIONS[name]
    return facts


def _find(text: str, phrase: str) -> int:
    i = text.find(phrase)
    return i if i >= 0 else text.lower().find(phrase.lower())


def _crop(text: str, focus: Sequence[tuple[int, int]], budget: int) -> str:
    """The sentences of `text` around the focus ranges, at most `budget` words (focus sentences are
    always kept); "[...]" marks what is left out."""
    sents = segment(text)
    words = [len(s.text.split()) for s in sents]
    if not sents or sum(words) <= budget:
        return text
    keep = {i for i, s in enumerate(sents) for a, b in focus if s.start < b and a < s.end} or {0}
    total = sum(words[i] for i in keep)
    grew = True
    while grew:
        grew = False
        for j in (max(keep) + 1, min(keep) - 1):
            if 0 <= j < len(sents) and j not in keep and total + words[j] <= budget:
                keep.add(j)
                total += words[j]
                grew = True
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append(GAP)
        out.append(sents[i].text)
        prev = i
    if min(keep) > 0:
        out.insert(0, GAP)
    if max(keep) < len(sents) - 1:
        out.append(GAP)
    return " ".join(out)


def build_excerpt(contract: dict, facts: dict[str, dict], requested: Sequence[str], n_distractors: int,
                  rng: random.Random, max_words: int, without_law: bool = False) -> str | None:
    """Opening + the paragraphs stating the requested facts + `n_distractors` other paragraphs, or
    None when the contract cannot give such an excerpt within `max_words`."""
    ctx = contract["context"]
    blocks = _blocks(ctx)
    if not blocks:
        return None
    norm = [_norm_space(ctx[a:b]) for a, b in blocks]
    law_blocks = {_block_of(blocks, s) for _, s in contract["answers"].get("Governing Law", [])} | \
        {i for i, t in enumerate(norm) if GOVERNING.search(t)}
    fact_blocks = {_block_of(blocks, s) for cat in FACT_CATEGORIES for _, s in contract["answers"].get(cat, [])}
    focus: dict[int, list[str]] = {}
    for name in requested:
        if name == "law" and without_law:
            continue
        f = facts[name]
        i = _block_of(blocks, f["start"])
        if i is None or _find(norm[i], f["evidence"]) < 0 or (without_law and i in law_blocks):
            return None
        focus.setdefault(i, []).append(f["evidence"])
    chosen, words = [], 0
    for i, t in enumerate(norm):  # the opening
        if words >= OPENING_WORDS:
            break
        if not (without_law and i in law_blocks):
            chosen.append(i)
            words += len(t.split())
    chosen += [i for i in focus if i not in chosen]
    pool = [i for i in range(len(blocks)) if i not in chosen and i not in law_blocks and i not in fact_blocks
            and len(norm[i].split()) >= MIN_DISTRACTOR_WORDS]
    if len(pool) < n_distractors:
        return None
    idx = sorted(chosen + rng.sample(pool, n_distractors))
    parts, prev = [], None
    for i in idx:
        spans = [(p, p + len(e)) for e in focus.get(i, []) for p in [_find(norm[i], e)]]
        part = _crop(norm[i], spans, BLOCK_WORDS)
        if prev is not None and i != prev + 1 and not parts[-1].endswith(GAP) and not part.startswith(GAP):
            parts.append(GAP)
        parts.append(part)
        prev = i
    if idx[-1] < len(blocks) - 1 and not parts[-1].endswith(GAP):
        parts.append(GAP)
    text = "\n\n".join(parts)
    if len(text.split()) > max_words or (without_law and GOVERNING.search(text)):
        return None
    if "date" in requested:  # an excerpt that also shows another labelled date asks an ambiguous question
        for t, start in contract["answers"].get("Effective Date", []):
            other = parse_date(t)
            if other and other != facts["date"]["date"] and _block_of(blocks, start) in idx:
                return None
    for name in requested:  # every present fact must be readable in the excerpt as shown
        if not (name == "law" and without_law) and _find(text, facts[name]["evidence"]) < 0:
            return None
    return text


def _cuad_item(contract: dict, facts: dict[str, dict], requested: list[str], answerable: bool, long: bool,
               excerpt: str, n_distractors: int, ci: int, j: int) -> Item:
    k = len(requested)
    questions = [CUAD_QUESTIONS[n] for n in requested]
    request = questions[0] if k == 1 else "Answer these questions about the contract:\n" + "\n".join(
        f"{i}. {q}" for i, q in enumerate(questions, 1))
    probes, vals = [], []
    for n in requested:
        missing = n == "law" and not answerable
        probes.append(Probe(question=CUAD_QUESTIONS[n], answers=list(NOT_STATED_FORMS) if missing else facts[n]["forms"],
                            fact_id=f"cuad:{ci}-{j}:{n}", value=NOT_STATED if missing else facts[n]["value"]))
        vals.append("Not stated in the excerpt" if missing else facts[n]["value"])
    gold = _end(vals[0]) if k == 1 else "\n".join(f"{i}. {_end(v)}" for i, v in enumerate(vals, 1))
    meta = {"source": "cuad", "cell": ci, "doc_label": "Contract excerpt", "doc_noun": "contract excerpt",
            "contract": contract["cid"], "facts": requested, "missing": None if answerable else "law",
            "words": len(excerpt.split()),
            "evidence": {f"cuad:{ci}-{j}:{n}": facts[n]["evidence"] for n in requested
                         if not (n == "law" and not answerable)}}
    knobs = {"k": k, "d": n_distractors, "answerable": answerable, "ambiguous": False, "hops": 1,
             "explain": False, "source": "cuad", "long": long}
    return Item(f"cu{ci:02d}-{j:04d}", excerpt, request, probes, gold, knobs, meta)


def cuad_items(contracts: Sequence[dict], n_per_cell: int, seed: int = 0,
               cells: Sequence[dict[str, Any]] = CUAD_CELLS) -> tuple[list[Item], dict]:
    """Up to n_per_cell items per cell, each cell from distinct contracts; fact types are balanced
    within a cell (the least used type first)."""
    facts_of = [contract_facts(c) for c in contracts]
    items, stats = [], {}
    for ci, cell in enumerate(cells):
        k, answerable = int(cell.get("k", 1)), bool(cell.get("answerable", True))
        long = bool(cell.get("long", False))
        n_dis = LONG_DISTRACTORS if long else DISTRACTORS
        max_words = LONG_MAX_WORDS if long else MAX_WORDS
        rng = random.Random(f"{seed}-cuad-{ci}")
        order = list(range(len(contracts)))
        rng.shuffle(order)
        used = {n: 0 for n in CUAD_FACTS}
        j = 0
        for idx in order:
            if j >= n_per_cell:
                break
            facts = facts_of[idx]
            if answerable:
                avail = [n for n in CUAD_FACTS if n in facts]
                if len(avail) < k:
                    continue
                avail.sort(key=lambda n: (used[n], rng.random()))
                requested = sorted(avail[:k], key=CUAD_FACTS.index)
            else:
                if "law" not in facts:  # the clause exists, so its absence is the excerpt's doing
                    continue
                others = [n for n in ("title", "date", "term") if n in facts]
                if len(others) < k - 1:
                    continue
                others.sort(key=lambda n: (used[n], rng.random()))
                requested = sorted(others[: k - 1] + ["law"], key=CUAD_FACTS.index)
            excerpt = build_excerpt(contracts[idx], facts, requested, n_dis, rng, max_words, without_law=not answerable)
            if excerpt is None:
                continue
            for n in requested:
                used[n] += 1
            items.append(_cuad_item(contracts[idx], facts, requested, answerable, long, excerpt, n_dis, ci, j))
            j += 1
        stats[f"cuad:{ci}"] = {"cell": cell, "built": j, "wanted": n_per_cell, "facts": used}
    return items, stats
