"""Reader: how much task-relevant information a message delivers.

For a request x, a message y and probes {(q_j, A_j)} (A_j = accepted aliases):

    lp_j(y) = log sum_{a in A_j} p_R(a | x, y, q_j)
    I_j(y)  = max(lp_j(y), log eps) - max(lp_j(empty), log eps)      (nats)
    I(y)    = sum_j I_j(y)

The reader never sees the source document, so on closed-world items the prior
lp_j(empty) is low and I(y) measures what the message actually transmits. The floor
eps (ReaderConfig.prior_floor, 1e-6) treats probabilities below one in a million as a
blind guess: a chat reader told to answer 'unknown' puts ~1e-20 on any specific answer
without the message, and without the floor every fact would be "worth" 47-60 nats, a
context word that lifts the answer from 1e-20 to 1e-10 would carry half of that, and
tolerances and ratios would scale with that arbitrary prior. A higher floor (1e-3)
also erases the signal of weak readers: GPT-2 gives a stated answer only ~1e-2.
A missing fact asked as a yes/no question has two possible replies and its own floor,
0.1 (ReaderConfig.binary_floor, BINARY_FLOOR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from drybench import abstain as _abstain
from drybench.backends.base import LMBackend
from drybench.types import Probe

PLAIN_TEMPLATE = "{fewshot}Request: {request}\nMessage: {message}\nQuestion: {question}\nAnswer:"
PLAIN_TEMPLATE_NO_REQUEST = "{fewshot}Message: {message}\nQuestion: {question}\nAnswer:"
EMPTY_MESSAGE = "(no message)"
# "unknown" (not "not stated") is the reader's default when the message has no answer, so a
# response that explicitly says a fact is "not stated" still carries information about it.
CHAT_SYSTEM = (
    "You read a message and answer a question about it. Use only the message. "
    "Reply with a short phrase copied from the message. If the message does not "
    "contain the answer, reply 'unknown'."
)
# A missing fact is delivered by a message that says it is not stated. Told to answer 'unknown'
# when the message has no answer, a chat reader also answers 'unknown' to "Not stated in the
# text." (v0.7.0: no gold answer of a one-question missing-fact item was readable), so the
# question of such a probe asks for 'not stated' explicitly. With that request alone the reader
# also said 'not stated' to an empty message for some questions (v0.7.1: a high prior, a weak
# missing fact), hence the second sentence. The reply is accepted when it starts with 'not':
# readers copy the response's wording ("not specified", "not mentioned", "not provided").
NOT_STATED_HINT = " If the message says that this is not stated, reply 'not stated'. If there is no message, reply 'unknown'."
MISSING_FORMS = ["not", "Not"]
# v0.7.2's hint makes readers say 'not stated' to a message that only does not mention the fact
# (CUAD, governing law left out: deleting "The governing law is not specified." cost 0.00 nats, so
# 19 of 21 correct abstentions were deleted as filler). Candidate questions for a missing fact, compared
# by `drybench missing-check` before one replaces the default ({question} = the content question):
#   hint      v0.7.2–v0.7.5 (silence read as 'not stated' by Qwen, an empty message by Phi)
#   plain     the content question alone (v0.7.0's): Qwen answers 'unknown' even to an abstention
#   silence   the hint with 'if the message does not mention it, reply unknown': no better
#   yesno     does the message say it is not stated? Qwen: right; Phi: 'yes' to an empty message
#   yesno2    the same with 'if the message only leaves it out, reply no'
#   contains  does the message include a sentence saying the source does not give it?
#   choice    A answers it, B says the source does not give it, C does not mention it: B
SILENCE_HINT = (" If the message says that its source does not give this, reply 'not stated'. If the message does "
                "not mention it at all, reply 'unknown'.")
YESNO_QUESTION = "Does the message say that this is not stated: \"{question}\"? Reply 'yes' or 'no'."
YES_FORMS = ["yes", "Yes"]
MISSING_QUESTIONS: dict[str, tuple[str, list[str]]] = {
    "hint": ("{question}" + NOT_STATED_HINT, MISSING_FORMS),
    "plain": ("{question}", MISSING_FORMS),
    "silence": ("{question}" + SILENCE_HINT, MISSING_FORMS),
    "yesno": (YESNO_QUESTION, YES_FORMS),
    "yesno2": ("Does the message say that its source does not give the answer to this question: \"{question}\"? "
               "If the message only leaves it out, reply 'no'. Reply 'yes' or 'no'.", YES_FORMS),
    "contains": ("Does the message include a sentence saying that its source does not give the answer to this "
                 "question: \"{question}\"? Reply 'yes' or 'no'.", YES_FORMS),
    "choice": ("What does the message do with this question: \"{question}\"? Reply A if it gives an answer, B if it "
               "says that its source does not give the answer, C if it does not mention it.", ["B"]),
}
MISSING_STYLES = tuple(MISSING_QUESTIONS)
# The question per reader, from `missing-check` on the real run (2 rounds, 7 wordings; checks: the
# filler map keeps an abstention ≥ 0.9, the reader reads gold and abstentions ≥ 0.8, reads one from an
# answer without it ≤ 0.2, and from no message p ≤ 0.05). Qwen2.5-3B passes with yesno only (kept 0.93,
# read 0.83–0.89, 0.17, 0.000). No wording passes for Phi-3.5-mini: every yes/no or A/B/C form gets
# 'yes'/'B' from an empty message too; its best is plain (kept 0.93, never read from silence or no
# message, but only half of the abstentions read). Granite-3.3-2B passed nothing either. Other readers
# get yesno; run `drybench missing-check` before relying on one.
DEFAULT_MISSING_STYLE = "yesno"
READER_MISSING_STYLE = {"qwen2.5-3b-instruct": "yesno", "phi-3.5-mini-instruct": "plain"}
# Questions answered 'yes'/'no' (or A/B/C) get a floor of their own (ReaderConfig.binary_floor). With
# two possible replies a reader never puts one in a million on 'yes': Qwen2.5-3B says 'yes' with
# p ≈ 0.01 (median) to an answer that gives a made-up value and ~0 to one that is silent. Against the
# floor 1e-6 that 0.01 is ~9.5 nats, so the made-up value "delivered" the missing fact and the filler map
# kept it (v0.8.0, scoring 7: "Delaware Law" kept in CUAD answers whose contract names no governing law;
# wrong statements on CUAD missing-law items 0.48–0.58 → 0.03–0.07). With 0.1 a 'yes' below one in ten
# reads as 'no': the fact counts as delivered from p ≥ 0.165 (0.5 nats) and carries at most ln 10 =
# 2.3 nats. 0.1 is the lowest round value above the largest p(yes) the main reader gave a gold answer
# without its 'not stated' sentence (0.139, real items), the reader check's 'silent' case.
BINARY_STYLES = ("yesno", "yesno2", "contains", "choice")
BINARY_FLOOR = 0.1


def missing_style_for(model: str) -> str:
    """The missing-fact question for a reader model (HF id or local path)."""
    return READER_MISSING_STYLE.get(str(model).rstrip("/").split("/")[-1].lower(), DEFAULT_MISSING_STYLE)


# fragments that say a fact is not stated (an abstention), in any wording; see drybench.abstain
ABSTAIN = _abstain.ABSTAIN


def is_missing(probe: Probe) -> bool:
    """A probe whose right answer is that the source does not state it."""
    return (probe.value or "").strip().lower() == "not stated" or any(
        a.strip().lower() == "not stated" for a in probe.answers)


# One demonstration for base models (fictional names disjoint from the item generator's).
GPT2_FEWSHOT = (
    "Request: For shipment QW-1180, what was the carrier?\n"
    "Message: Sure! The shipment was handled by Tarnell Lines. It carried 40 bales of jute.\n"
    "Question: Which carrier handled shipment QW-1180?\n"
    "Answer: Tarnell Lines\n\n"
)


@dataclass
class ReaderConfig:
    use_chat: bool = False
    template: str = PLAIN_TEMPLATE
    fewshot: str = ""
    empty_message: str = EMPTY_MESSAGE
    answer_prefix: str | None = None  # None: " " for plain templates, "" for chat
    system: str = CHAT_SYSTEM
    prior_floor: float = 1e-6  # probabilities below it count as a blind guess; 0 = plain PMI
    missing_style: str = DEFAULT_MISSING_STYLE  # how a missing fact is asked (MISSING_STYLES); "hint" through v0.7.6
    binary_floor: float = BINARY_FLOOR  # floor for a missing fact asked as a yes/no question (BINARY_STYLES)

    @property
    def prefix(self) -> str:
        if self.answer_prefix is not None:
            return self.answer_prefix
        return "" if self.use_chat else " "


@dataclass
class Reader:
    backend: LMBackend
    config: ReaderConfig = field(default_factory=ReaderConfig)

    cache_size: int = 500_000

    def __post_init__(self) -> None:
        self._prior_cache: dict[tuple, float] = {}
        self._lp_cache: dict[int, float] = {}  # hash((context, continuation)) -> log-prob

    @classmethod
    def for_backend(cls, backend: LMBackend, fewshot: str | None = None) -> "Reader":
        """Chat reader if the backend has a chat template, plain few-shot reader otherwise."""
        if backend.has_chat_template:
            return cls(backend, ReaderConfig(use_chat=True, missing_style=missing_style_for(backend.name)))
        return cls(backend, ReaderConfig(fewshot=GPT2_FEWSHOT if fewshot is None else fewshot))

    def prompt(self, request: str, message: str, question: str) -> str:
        msg = message.strip() or self.config.empty_message
        if self.config.use_chat:
            user = f"Request: {request}\nMessage: {msg}\nQuestion: {question}"
            return self.backend.format_chat(
                [{"role": "system", "content": self.config.system}, {"role": "user", "content": user}])
        return self.config.template.format(fewshot=self.config.fewshot, request=request,
                                           message=msg, question=question)

    def question(self, probe: Probe) -> str:
        """The question that measures a probe's information (a missing fact: config.missing_style)."""
        if self.config.use_chat and is_missing(probe):
            return MISSING_QUESTIONS[self.config.missing_style][0].format(question=probe.question)
        return probe.question

    def content_question(self, probe: Probe) -> str:
        """The question for reading a wrong answer (what does the message say the answer is):
        a missing fact asks for 'not stated' explicitly, whatever measures its information."""
        if self.config.use_chat and is_missing(probe):
            return probe.question + NOT_STATED_HINT
        return probe.question

    def forms(self, probe: Probe) -> list[str]:
        """Accepted replies: the probe's answers; for a missing fact (chat readers) any reply that
        starts with 'not' ('not stated', 'not specified', 'not mentioned', ...), or the reply that
        means 'it says so' when the reader is asked whether the message says the fact is not stated."""
        if self.config.use_chat and is_missing(probe):
            return list(MISSING_QUESTIONS[self.config.missing_style][1])
        return list(probe.answers)

    def answer(self, request: str, message: str, question: str, max_new_tokens: int = 24) -> str:
        """The reader's own short reply (greedy), first line, without quotes or a final period."""
        text = self.backend.generate(self.prompt(request, message, question), max_new_tokens, 0.0)
        line = (text.strip().splitlines() or [""])[0].strip()
        return line.strip("\"'“”‘’`*").rstrip(".").strip()

    def reply_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        """log p(the reply starts with each continuation). A continuation that ends with a digit must not go on
        with another digit: "70" does not cover "700", "3" does not cover "30" (v1.2)."""
        out = np.zeros(len(requests))
        term = [i for i, (_, k) in enumerate(requests) if k[-1:].isdigit()]
        plain = [i for i, (_, k) in enumerate(requests) if not k[-1:].isdigit()]
        if plain:
            out[plain] = self.backend.continuation_logprobs([requests[i] for i in plain])
        if term:
            out[term] = self.backend.terminated_logprobs([requests[i] for i in term])
        return out

    def probe_logprobs(self, request: str, messages: Sequence[str],
                       probes: Sequence[Probe]) -> np.ndarray:
        """[n_messages, n_probes] matrix of log sum_alias p(alias | x, message, q)."""
        out = np.full((len(messages), len(probes)), -np.inf)
        reqs, index = [], []
        for mi, msg in enumerate(messages):
            for pj, probe in enumerate(probes):
                ctx = self.prompt(request, msg, self.question(probe))
                for alias in self.forms(probe):
                    cont = self.config.prefix + alias
                    key = hash((ctx, cont))
                    if key in self._lp_cache:  # repeated sub-messages across search steps
                        out[mi, pj] = np.logaddexp(out[mi, pj], self._lp_cache[key])
                    else:
                        reqs.append((ctx, cont))
                        index.append((mi, pj, key))
        if reqs:
            lps = self.reply_logprobs(reqs)
            if len(self._lp_cache) + len(reqs) > self.cache_size:
                self._lp_cache.clear()
            for (mi, pj, key), lp in zip(index, lps):
                self._lp_cache[key] = float(lp)
                out[mi, pj] = np.logaddexp(out[mi, pj], lp)
        return out

    def absence_logprobs(self, request: str, messages: Sequence[str], probes: Sequence[Probe],
                         strings: Sequence[str] | Sequence[Sequence[str]]) -> np.ndarray:
        """[n_messages, n_probes] log p(the reply starts like one of `strings`, e.g. 'unknown').

        `strings` is one list for all probes or one list per probe."""
        per_probe = [list(strings)] * len(probes) if strings and isinstance(strings[0], str) \
            else [list(x) for x in strings]
        out = np.full((len(messages), len(probes)), -np.inf)
        for pj, probe in enumerate(probes):
            tag = ("\0first",) + tuple(per_probe[pj])
            ctxs, index = [], []
            for mi, msg in enumerate(messages):
                ctx = self.prompt(request, msg, self.content_question(probe))
                key = hash((ctx, tag))
                if key in self._lp_cache:
                    out[mi, pj] = self._lp_cache[key]
                else:
                    ctxs.append(ctx)
                    index.append((mi, key))
            if ctxs:
                vals = self.backend.first_token_logmass(ctxs, [self.config.prefix + s for s in per_probe[pj]])
                if len(self._lp_cache) + len(ctxs) > self.cache_size:
                    self._lp_cache.clear()
                for (mi, key), v in zip(index, vals):
                    self._lp_cache[key] = float(v)
                    out[mi, pj] = v
        return out

    def prior_logprobs(self, request: str, probes: Sequence[Probe]) -> np.ndarray:
        keys = [(request, p.question, tuple(p.answers)) for p in probes]
        todo = [p for p, key in zip(probes, keys) if key not in self._prior_cache]
        if todo:
            vals = self.probe_logprobs(request, [""], todo)[0]
            for p, v in zip(todo, vals):
                self._prior_cache[(request, p.question, tuple(p.answers))] = float(v)
        return np.array([self._prior_cache[key] for key in keys])

    def floor(self, probe: Probe) -> float:
        """The probability below which a reply counts as a blind guess: prior_floor; for a missing fact
        asked as a yes/no question (chat readers, BINARY_STYLES) at least binary_floor."""
        if self.config.use_chat and self.config.missing_style in BINARY_STYLES and is_missing(probe):
            return max(self.config.prior_floor, self.config.binary_floor)
        return self.config.prior_floor

    def log_floors(self, probes: Sequence[Probe]) -> np.ndarray:
        """[n_probes] log floors (-inf: no floor)."""
        with np.errstate(divide="ignore"):
            return np.log(np.array([self.floor(p) for p in probes], float))

    def per_probe_information(self, request: str, messages: Sequence[str],
                              probes: Sequence[Probe]) -> np.ndarray:
        """[n_messages, n_probes] information gains (nats), with the floor of each probe."""
        lp = self.probe_logprobs(request, messages, probes)
        prior = self.prior_logprobs(request, probes)[None, :]
        floor = self.log_floors(probes)[None, :]
        return np.maximum(lp, floor) - np.maximum(prior, floor)

    def information(self, request: str, messages: Sequence[str], probes: Sequence[Probe]) -> np.ndarray:
        """I(y) for each message (nats)."""
        return self.per_probe_information(request, messages, probes).sum(axis=1)
