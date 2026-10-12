"""Labelling guidelines: one text for the human labelling page (annotation/dry_span_labels.html), chat
LLM annotators (scripts/chat_annotators.py) and the LLM judge baseline (drybench.judge), so that all of
them label by the same rules.

Every fragment of a response (a sentence or a list line, as drybench.segment cuts it) gets one label:
  needed  deleting it loses something the request asks for
  filler  deleting it loses nothing requested, and nothing in it is wrong
  wrong   it gives a wrong answer or contradicts the source (takes priority)
'needed' fragments are what the filler map must keep; 'filler' and 'wrong' are what it may delete
(removable), and 'wrong' is compared with its wrong-statement label.

scripts/build_annotation_page.py writes GUIDE and EXAMPLES into the page; tests check they match.
"""

from __future__ import annotations

import html
import json

LABELS = ("needed", "filler", "wrong")
REMOVABLE = ("filler", "wrong")

GUIDE: dict = {
    "intro": ("Each response answers a request about a source text: an operations report, Wikipedia "
              "paragraphs, a contract excerpt, a story summary, a tweet or web search results. The model "
              "that wrote it was told to answer from that text. Label every fragment of the response (a "
              "sentence or a list line), judging it against the source text and the request only. Style, "
              "grammar and length as such do not matter."),
    "intro_page": (" The page does not show which model wrote a response or how the benchmark scored it; do "
                   "not open key.jsonl until you finish."),
    "needed": ("Deleting it would lose something the request asks for: a requested value or answer, a step of "
               "an explanation the request asks for, or the correct statement that the source does not give a "
               "requested fact. A fragment that gives a requested answer is Needed even if it also has extra "
               "words (“As the text explains, the bridge was designed by Mira Ostrova.”)."),
    "filler": ("Deleting it loses nothing the request asks for, and nothing in it is wrong: restating the "
              "question, openers and sign-offs, offers of help, generic caveats, a repeat of an answer already "
              "given, details nobody asked for (whether or not the source mentions them), steps toward the "
              "answer when the request does not ask for an explanation."),
    "wrong": ("It gives a wrong answer or contradicts the source: a wrong value, another entity's value given "
              "as the asked one, a guess at a requested fact the source does not give, “not stated” for a fact "
              "the source gives, or any claim the source contradicts. Wrong takes priority over the other two "
              "labels. If the fragment also gives a requested fact correctly, {also}."),
    "also_page": "add <b>+ needed fact</b> (key 4)",
    "also_text": 'add "also_needed": true',
    "rules_title": "Tricky cases",
    "rules": [
        ("Repeats", "The first full statement of an answer is Needed; later repeats are Filler: a summary that "
                    "restates the answers, or a clause quoted again after its value was given. A later fragment "
                    "is Needed only if it adds a requested fact."),
        ("Explanations", "When the request says “Explain how you found it”, the steps that lead to the answer "
                         "are requested (which carrier handled the shipment and where that carrier is based; "
                         "which engineer designed the bridge and where she was born): they are Needed. Describing "
                         "the search itself (“I looked through the documents…”) is Filler. Without such a request "
                         "the steps are Filler, even when they are correct and helpful."),
        ("Several questions", "In a numbered request every question is requested: a fragment that answers any "
                              "of them is Needed. A heading or a number with no content (“Answers:”, “1.”, “For "
                              "shipment KR-4417:”) is Filler."),
        ("Two entities fit the request", "When two shipments (people, places, …) fit the request, the "
                                         "requested value of each one is requested. A fragment that only says "
                                         "that two of them match is Filler when their values are given "
                                         "elsewhere; a value given for just one of them is still Needed."),
        ("Missing fact", "“The source does not state X” is Needed when X really is absent from the source, and "
                         "Wrong when the source gives X."),
        ("Unrequested details", "Details nobody asked for are Filler whether or not the source mentions them, "
                                "unless the source contradicts them (then Wrong)."),
        ("Unsure?", "{unsure}"),
    ],
    "unsure_page": "Choose the best label and add Unsure (key U).",
    "unsure_text": 'Choose the best label and add "unsure": true.',
    "examples_title": "Examples",
    "keys_title": "Keys",
    "keys": ("<kbd>1</kbd> Needed · <kbd>2</kbd> Filler · <kbd>3</kbd> Wrong · <kbd>4</kbd> + needed fact · "
             "<kbd>U</kbd> Unsure · <kbd>0</kbd> clear · <kbd>↑</kbd> <kbd>↓</kbd> fragments · <kbd>Enter</kbd> "
             "or <kbd>→</kbd> next response · <kbd>←</kbd> back"),
}

# invented sources (no real names), shown on the page and given to chat annotators
EXAMPLES: list[dict] = [
    {"source_label": "Report",
     "document": ["Operations report, Velmora terminal.",
                  "Shipment KR-4417 was handled by Dunmar Freight.",
                  "Shipment KR-4417 carried 23 crates of tin ingots.",
                  "Shipment KR-4417 left on 14 May.",
                  "Shipment TB-2081 was handled by Olpel Lines.",
                  "Figures are provisional until the quarterly audit."],
     "request": "For shipment KR-4417, what were the carrier and the cargo?",
     "spans": [
         ["Sure! Here is the information about shipment KR-4417:", "filler", "opener"],
         ["The carrier was Dunmar Freight.", "needed", "requested value"],
         ["It carried 23 crates of tin ingots.", "needed", "requested value"],
         ["It left on 4 May.", "wrong", "the report says 14 May"],
         ["Shipment TB-2081 was handled by Olpel Lines.", "filler", "true, but not asked"],
         ["In short, Dunmar Freight carried the tin ingots.", "filler", "repeat"],
     ]},
    {"source_label": "Documents",
     "document": ["Document 1: Varn Bridge",
                  "The Varn Bridge is a stone arch bridge over the river Ilse. It was designed by the engineer "
                  "Mira Ostrova and opened in 1891.",
                  "",
                  "Document 2: Mira Ostrova",
                  "Mira Ostrova (1849–1920) was an engineer born in Kessel. She later taught at the Kessel "
                  "Polytechnic."],
     "request": "In which town was the designer of the Varn Bridge born?",
     "spans": [
         ["The Varn Bridge was designed by Mira Ostrova.", "filler", "a step; no explanation was asked for"],
         ["She was born in Kessel.", "needed", "requested value"],
         ["She later taught at the Kessel Polytechnic.", "filler", "true, but not asked"],
         ["It is one of the oldest bridges in the region.", "filler",
          "not asked; the source neither states nor contradicts it"],
     ]},
]


def _text(key: str, target: str) -> str:
    """One field with its placeholders filled for the page ('page', HTML) or a plain prompt ('text')."""
    return GUIDE[key].replace("{also}", GUIDE[f"also_{target}"])


def definitions(target: str = "text") -> dict[str, str]:
    """The three label definitions (plain text for 'text', HTML for 'page')."""
    out = {}
    for lab in LABELS:
        v = _text(lab, target)
        out[lab] = v if target == "page" else html.unescape(v.replace("<b>", "").replace("</b>", ""))
    return out


def rules(target: str = "text") -> list[tuple[str, str]]:
    unsure = GUIDE[f"unsure_{target}"]
    return [(title, body.replace("{unsure}", unsure)) for title, body in GUIDE["rules"]]


def guidelines_text(judge: bool = False) -> str:
    """The guidelines as plain text for an LLM: labels, then the tricky cases. judge=True: for a judge
    that gives one label per fragment and nothing else (no 'also_needed', no 'unsure')."""
    d = definitions("text")
    rl = rules("text")
    if judge:
        d["wrong"] = d["wrong"][: d["wrong"].index(" If the fragment also gives")]
        rl = [r for r in rl if not r[1].startswith("Choose the best label")]
    lines = ["LABELS. Give every fragment exactly one label:"]
    lines += [f'- "{lab}": {d[lab]}' for lab in LABELS]
    lines += ["", "TRICKY CASES."]
    lines += [f"{n}. {title}: {body}" for n, (title, body) in enumerate(rl, 1)]
    return "\n".join(lines)


def _page_guide() -> dict:
    esc = lambda s: html.escape(s, quote=False)  # noqa: E731  (definitions may carry <b> tags: build them here)
    d = {lab: esc(GUIDE[lab]).replace("{also}", GUIDE["also_page"]) for lab in LABELS}
    unsure = esc(GUIDE["unsure_page"])
    return {"intro": GUIDE["intro"] + GUIDE["intro_page"], **d, "rulesTitle": GUIDE["rules_title"],
            "rules": [f"<b>{esc(t)}{'' if t.endswith(('?', '!', '.')) else '.'}</b> "
                      f"{esc(b).replace('{unsure}', unsure)}" for t, b in GUIDE["rules"]],
            "exTitle": GUIDE["examples_title"], "keysTitle": GUIDE["keys_title"], "keys": GUIDE["keys"]}


def page_js() -> str:
    """The GUIDE and EXAMPLES constants of the labelling page."""
    return ("const GUIDE = " + json.dumps(_page_guide(), ensure_ascii=False, indent=1) + ";\n\n"
            "const EXAMPLES = " + json.dumps(EXAMPLES, ensure_ascii=False, indent=1) + ";\n")
