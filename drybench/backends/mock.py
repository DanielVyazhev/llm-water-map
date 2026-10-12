"""Deterministic mock backend for unit tests.

Reader semantics: p(answer | prompt) is high when the answer string occurs in the
prompt's message (everything before the last "Question:"), low otherwise. More
occurrences give slightly higher probability, so redundancy is visible.

Tokens are whitespace-separated words.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Sequence

import numpy as np

from drybench.backends.base import LMBackend

FORMULAIC_WORDS = {
    "great", "question", "sure", "let", "me", "hope", "this", "helps", "know", "if", "you",
    "need", "anything", "else", "feel", "free", "to", "ask", "thanks", "for", "asking",
    "certainly", "here", "is", "what", "the", "report", "says", "i", "a", "an", "of", "and",
}


def _default_entropy(word: str) -> float:
    w = re.sub(r"[^\w-]", "", word).lower()
    return 0.5 if w in FORMULAIC_WORDS else 2.5


ARTICLES = {"a", "an", "the", "A", "An", "The"}
# a message that says the source does not give something: the reader replies 'not stated' / 'Not ...'
NOT_SAID = re.compile(r"\bnot (?:stated|specified|mentioned|provided|given)\b", re.I)
ARTICLE_BEFORE = re.compile(r"(?:^|\s)(?:a|an|the|A|An|The)\s$")


class MockBackend(LMBackend):
    name = "mock"

    def __init__(
        self,
        p_present: float = 0.8,
        p_absent: float = 1e-4,
        entropy_fn: Callable[[str], float] | None = None,
        next_entropy_fn: Callable[[str], float] | None = None,
        generate_fn: Callable[[str], str] | None = None,
        prior_fn: Callable[[str], float] | None = None,
        unknown_fn: Callable[[str], float] | None = None,
    ) -> None:
        self.p_present = p_present
        self.p_absent = p_absent
        self.prior_fn = prior_fn
        # p('unknown' | message): a chat reader says 'unknown' unless the message asserts
        # something; test messages mark a wrong assertion with the token WRONG
        self.unknown_fn = unknown_fn or (lambda msg: 0.02 if "WRONG" in msg else 0.9)
        self.entropy_fn = entropy_fn or _default_entropy
        self.next_entropy_fn = next_entropy_fn or (lambda ctx: (sum(map(ord, ctx)) % 300) / 100.0)
        self.generate_fn = generate_fn or (lambda ctx: "I hope this helps!")
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        return [hash(w) % 50000 for w in text.split()]

    def _message_part(self, context: str) -> str:
        """The message the reader is asked about (not the request, system prompt or few-shot)."""
        cut = context.rfind("Question:")
        cut = cut if cut >= 0 else len(context)
        start = context.rfind("Message:", 0, cut)
        return context[start + len("Message:"):cut] if start >= 0 else context[:cut]

    @staticmethod
    def _judge_label(context: str) -> str | None:
        """A judge prompt (drybench.judge) asks for the label of fragment [i]: 'wrong' when the fragment
        has the token WRONG, 'filler' when all its words are formulaic, else 'needed'."""
        m = re.search(r"Which label does fragment \[(\d+)\] get\?", context)
        if not m:
            return None
        frag = re.search(r"^\[" + m.group(1) + r"\] (.*)$", context, re.M)
        text = frag.group(1) if frag else ""
        if "WRONG" in text:
            return "wrong"
        words = [re.sub(r"[^\w-]", "", w).lower() for w in text.split()]
        return "filler" if all(w in FORMULAIC_WORDS for w in words if w) else "needed"

    def continuation_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        out = []
        for context, continuation in requests:
            self.calls += 1
            ans = continuation.strip()
            judged = self._judge_label(context)
            if judged is not None:
                out.append(math.log(0.8 if ans == judged else 0.1))
                continue
            if (ans in ("yes", "Yes") and "Does the message" in context) or (ans == "B" and "Reply A if" in context):
                share = 1.0 if ans == "B" else 0.5  # two forms ('yes', 'Yes') share the mass
                msg = self._message_part(context)
                # a made-up value (WRONG) is weak evidence to a yes/no reader (Qwen2.5-3B: p ≈ 0.01)
                p = self.p_present * share if NOT_SAID.search(msg) else (0.01 * share if "WRONG" in msg else 1e-6)
                out.append(math.log(p))
                continue
            if ans in ("unknown", "Unknown", "not", "Not"):  # absence replies of a chat reader
                msg = self._message_part(context)
                if ans == "unknown":
                    p = self.unknown_fn(msg)
                elif ans in ("not", "Not") and NOT_SAID.search(msg):  # 'not stated' / 'not specified' ...
                    p = self.p_present / 2
                else:
                    p = 1e-6
                out.append(math.log(p))
                continue
            msg = self._message_part(context)
            pattern = r"(?<!\w)" + re.escape(ans) + r"(?!\w)"
            # a reader copies whole phrases: "labour dispute" inside "a labour dispute" is the same
            # occurrence as the alias "a labour dispute", so count it only once
            bare = ans.split(" ", 1)[0] not in ARTICLES
            n = sum(1 for m in re.finditer(pattern, msg)
                    if not (bare and ARTICLE_BEFORE.search(msg[: m.start()]))) if ans else 0
            prior = self.prior_fn(ans) if self.prior_fn else self.p_absent
            if n == 0:
                p = prior
            else:
                p = max(prior, 1.0 - (1.0 - self.p_present) * (0.75 ** (n - 1)))
            out.append(math.log(p))
        return np.array(out)

    def token_entropies(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", continuation)]
        ents = np.array([self.entropy_fn(continuation[a:b]) for a, b in offsets], dtype=float)
        return ents, offsets

    def token_logprobs(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Formulaic words are predictable (surprisal = their mock entropy)."""
        ents, offsets = self.token_entropies(context, continuation)
        return -ents, offsets

    def next_token_entropy(self, contexts: Sequence[str]) -> np.ndarray:
        return np.array([self.next_entropy_fn(c) for c in contexts], dtype=float)

    def generate(self, context: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        return self.generate_fn(context)
