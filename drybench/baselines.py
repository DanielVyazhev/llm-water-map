"""Cheap baselines for filler / verbosity.

length        response length (tokens, words or characters)
yap_excess    YapBench-style excess characters over the minimal sufficient answer
              (needs a curated gold minimal answer)
stopword_share share of stop words among words (a score of SEO text checkers such as Advego)
judge_filler  LLM-judge estimate of the unnecessary share of a response, read off the judge's
              distribution over "0".."100" (expected value; no free-text parsing)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drybench.backends.base import LMBackend

EN_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just let me more
most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves also may might must shall please note however
""".split())

_WORD = re.compile(r"[\w'-]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def length_chars(text: str) -> int:
    return len(text.strip())


def length_words(text: str) -> int:
    return len(words(text))


def yap_excess_chars(response: str, gold_minimal: str) -> int:
    return max(0, len(response.strip()) - len(gold_minimal.strip()))


def stopword_share(text: str) -> float:
    ws = words(text)
    if not ws:
        return 0.0
    return sum(w in EN_STOPWORDS for w in ws) / len(ws)


JUDGE_SYSTEM = (
    "You evaluate answers for unnecessary text. Estimate what percentage of the answer could be "
    "deleted without losing anything the request asked for. Reply with a percentage such as 30%."
)
JUDGE_LEVELS = list(range(0, 101, 10))


def judge_filler(backend: "LMBackend", request: str, response: str) -> float:
    """Expected unnecessary fraction under the judge's distribution over 0%, 10%, ..., 100%.

    Options end with "%" so they are prefix-free for any tokenizer ("10%" vs "100%")."""
    user = f"Request: {request}\n\nAnswer:\n{response}\n\nPercentage of unnecessary text:"
    ctx = backend.format_chat([{"role": "system", "content": JUDGE_SYSTEM},
                               {"role": "user", "content": user}])
    prefix = "" if backend.has_chat_template else " "
    lps = backend.continuation_logprobs([(ctx, f"{prefix}{v}%") for v in JUDGE_LEVELS])
    p = np.exp(lps - lps.max())
    p /= p.sum()
    return float((p * np.array(JUDGE_LEVELS)).sum() / 100.0)
