"""LLM judge baseline: an instruction-tuned model labels every fragment of a response needed / filler /
wrong. It reads the source text, the request and the whole response, and follows the guidelines the
human annotators follow (drybench.guidelines).

The label is read off the judge's probabilities of the three words, each in lower case and capitalised
(the guidelines write "Needed", the reply instruction "needed"; LMBackend.choice_logprobs: no free text to
parse, deterministic): the most probable word is the label, and p(filler) + p(wrong) is the judge's
probability that the fragment is removable. The six raw log-probabilities and their total probability are
kept with every label (a small total means the judge wanted to say something else). One pass per fragment.

Unlike the filler map's reader, the judge must read the source text: its cost grows with the document,
the reader's does not (it reads only the request and the response).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from drybench.guidelines import LABELS, guidelines_text

SYSTEM = ("You are one of several annotators in a research study on unnecessary text in chatbot answers. A "
          "chatbot answered a request about a source text; its answer is split into numbered fragments. You "
          "label one fragment at a time.\n\n" + guidelines_text(judge=True) + "\n\nJudge the fragment against the "
          "source text and the request only; style, grammar and length as such do not matter.")


def judge_messages(document: str, request: str, fragments: Sequence[str], i: int,
                   source_label: str = "Source text") -> list[dict[str, str]]:
    """Chat messages asking for the label of fragment i (0-based; shown 1-based)."""
    frag = "\n".join(f"[{k + 1}] {' '.join(f.split())}" for k, f in enumerate(fragments))
    user = (f"{source_label}:\n{document}\n\nRequest: {request}\n\nAnswer, split into fragments:\n{frag}\n\n"
            f"Which label does fragment [{i + 1}] get? Reply with one word: needed, filler or wrong.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def judge_fragments(backend, document: str, request: str, fragments: Sequence[str],
                    source_label: str = "Source text") -> list[dict]:
    """[{label, p: [p(needed), p(filler), p(wrong)]}] per fragment."""
    if not fragments:
        return []
    ctxs = [backend.format_chat(judge_messages(document, request, fragments, i, source_label))
            for i in range(len(fragments))]
    prefix = "" if backend.has_chat_template else " "
    options = [prefix + form for lab in LABELS for form in (lab, lab.capitalize())]
    raw = np.asarray(backend.choice_logprobs(ctxs, options), float)
    per_label = np.logaddexp(raw[:, 0::2], raw[:, 1::2])
    mass = np.logaddexp.reduce(raw, axis=1)
    lp = per_label - per_label.max(axis=1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    return [{"label": LABELS[int(np.argmax(row))], "p": [float(x) for x in row],
             "lp": [float(x) for x in r], "mass": float(np.exp(m))} for row, r, m in zip(p, raw, mass)]
