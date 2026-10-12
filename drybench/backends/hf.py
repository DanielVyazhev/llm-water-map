"""Hugging Face transformers backend (GPU or CPU).

Scoring uses left padding with explicit position ids and asks the model for logits of
the last positions only (`logits_to_keep`), so memory stays proportional to the
continuation length rather than to the full vocabulary x sequence.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from drybench.backends.base import (
    LMBackend,
    entropy_from_logits,
    split_context_continuation,
    token_budget_batches,
)

_DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16",
           "fp32": "float32", "float32": "float32"}


def _merge_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    if rest and rest[0]["role"] == "user":
        rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
    else:
        rest.insert(0, {"role": "user", "content": sys_text})
    return rest


class HFBackend(LMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        dtype: str = "auto",
        max_batch_tokens: int = 8192,
        max_length: int | None = None,
        prepend_bos: bool | None = None,
        trust_remote_code: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if dtype == "auto":
            torch_dtype: Any = torch.float32 if self.device == "cpu" else "auto"
        else:
            torch_dtype = getattr(torch, _DTYPES[dtype])
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **{dtype_kw: torch_dtype}
        ).to(self.device).eval()
        self.max_batch_tokens = max_batch_tokens
        cfg_len = getattr(self.model.config, "max_position_embeddings", None) or 4096
        self.max_length = max_length or min(int(cfg_len), 32768)
        tok = self.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if prepend_bos is None:
            prepend_bos = (tok.bos_token_id is not None
                           and tok("a").input_ids[:1] == [tok.bos_token_id]
                           and tok("a", add_special_tokens=False).input_ids[:1] != [tok.bos_token_id])
        self.prepend_bos = bool(prepend_bos)
        self.chat_template_kwargs = chat_template_kwargs or {}
        self._supports_keep = True

    # ------------------------------------------------------------------ tokenization
    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _start_ids(self, context: str, ctx_ids: list[int]) -> list[int]:
        tok = self.tokenizer
        if self.prepend_bos and not (tok.bos_token and context.startswith(tok.bos_token)):
            return [tok.bos_token_id] + ctx_ids
        if not ctx_ids:  # nothing to condition the first token on
            start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            return [start]
        return ctx_ids

    def _offsets(self, text: str, ids: list[int]) -> list[tuple[int, int]]:
        try:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if enc.input_ids == ids:
                return [tuple(o) for o in enc["offset_mapping"]]
        except (NotImplementedError, KeyError, TypeError):
            pass
        offsets, prev = [], 0
        for i in range(len(ids)):  # fallback: cumulative decoding
            cur = len(self.tokenizer.decode(ids[: i + 1]))
            offsets.append((prev, cur))
            prev = cur
        return offsets

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        if not self.has_chat_template:
            return super().format_chat(messages)
        kw = dict(tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
        try:
            return self.tokenizer.apply_chat_template(messages, **kw)
        except Exception:  # some templates reject the system role
            return self.tokenizer.apply_chat_template(_merge_system(messages), **kw)

    # ------------------------------------------------------------------ forward helpers
    def _forward_last(self, seqs: list[list[int]], n_keep: int):
        """Left-pad `seqs`, run the model, return float32 logits of the last n_keep positions."""
        torch = self.torch
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, T - len(s):] = torch.tensor(s, dtype=torch.long)
            mask[b, T - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        ids, mask, pos = ids.to(self.device), mask.to(self.device), pos.to(self.device)
        with torch.inference_mode():
            if self._supports_keep:
                try:
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                     logits_to_keep=n_keep)
                    return out.logits[:, -n_keep:].float()
                except TypeError:
                    self._supports_keep = False
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos)
            return out.logits[:, -n_keep:].float()

    def _prepare(self, context: str, continuation: str) -> tuple[list[int], list[int], int]:
        ctx, cont, moved = split_context_continuation(self.encode, context, continuation)
        ctx = self._start_ids(context, ctx)
        overflow = len(ctx) + len(cont) - self.max_length
        if overflow > 0:
            ctx = ctx[overflow:]
        return ctx, cont, moved

    # ------------------------------------------------------------------ interface
    def continuation_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        torch = self.torch
        enc = [self._prepare(c, k) for c, k in requests]
        out = np.zeros(len(requests))
        lengths = [len(c) + len(k) for c, k, _ in enc]
        for batch in token_budget_batches(lengths, self.max_batch_tokens):
            batch = [i for i in batch if enc[i][1]]
            if not batch:
                continue
            n_keep = max(len(enc[i][1]) for i in batch) + 1
            logits = self._forward_last([enc[i][0] + enc[i][1] for i in batch], n_keep)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"non-finite logits from {self.name}; reload with dtype='fp32' (or bf16 on newer GPUs)")
            for b, i in enumerate(batch):
                k = enc[i][1]
                sl = logits[b, n_keep - len(k) - 1: n_keep - 1]
                lp = torch.log_softmax(sl, dim=-1)
                out[i] = lp[torch.arange(len(k)), torch.tensor(k, device=lp.device)].sum().item()
        return out

    def first_token_logmass(self, contexts: Sequence[str], strings: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = np.full(len(contexts), -np.inf)
        seqs, firsts, slow = {}, {}, []
        for i, c in enumerate(contexts):
            enc = [self._prepare(c, s) for s in strings]
            enc = [(ctx, cont) for ctx, cont, _ in enc if cont]
            if not enc:
                continue
            if any(ctx != enc[0][0] for ctx, _ in enc):  # context tokens depend on the string
                slow.append(i)
                continue
            seqs[i], firsts[i] = enc[0][0], sorted({cont[0] for _, cont in enc})
        todo = list(seqs)
        for batch in token_budget_batches([len(seqs[i]) for i in todo], self.max_batch_tokens):
            idx = [todo[b] for b in batch]
            logits = self._forward_last([seqs[i] for i in idx], 1)[:, 0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"non-finite logits from {self.name}; reload with dtype='fp32' (or bf16 on newer GPUs)")
            lp = torch.log_softmax(logits.float(), dim=-1)
            for b, i in enumerate(idx):
                out[i] = torch.logsumexp(lp[b, firsts[i]], dim=0).item()
        if slow:
            out[slow] = super().first_token_logmass([contexts[i] for i in slow], strings)
        return out

    def token_entropies(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        return self._token_scores(context, continuation, "entropy")

    def token_logprobs(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        return self._token_scores(context, continuation, "logprob")

    def choice_logprobs(self, contexts: Sequence[str], options: Sequence[str]) -> np.ndarray:
        """First-token log-probabilities of the options in one pass per context when the options start
        with different tokens after it; the options' whole continuations otherwise."""
        torch = self.torch
        out = np.zeros((len(contexts), len(options)))
        seqs, firsts, slow = {}, {}, []
        for i, c in enumerate(contexts):
            enc = [self._prepare(c, o) for o in options]
            heads = [cont[0] if cont else None for _, cont, _ in enc]
            if (None in heads or len(set(heads)) < len(heads)
                    or any(ctx != enc[0][0] for ctx, _, _ in enc)):  # shared first token, or context
                slow.append(i)                                       # tokens depend on the option
                continue
            seqs[i], firsts[i] = enc[0][0], heads
        todo = list(seqs)
        for batch in token_budget_batches([len(seqs[i]) for i in todo], self.max_batch_tokens):
            idx = [todo[b] for b in batch]
            logits = self._forward_last([seqs[i] for i in idx], 1)[:, 0]
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"non-finite logits from {self.name}; reload with dtype='fp32' (or bf16 on newer GPUs)")
            lp = torch.log_softmax(logits.float(), dim=-1)
            for b, i in enumerate(idx):
                out[i] = lp[b, firsts[i]].cpu().numpy()
        if slow:
            out[slow] = super().choice_logprobs([contexts[i] for i in slow], options)
        return out

    def _token_scores(self, context: str, continuation: str, kind: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Entropy ('entropy') or log p ('logprob') at every continuation token, with character offsets."""
        ctx_raw, cont, moved = split_context_continuation(self.encode, context, continuation)
        ctx = self._start_ids(context, ctx_raw)
        if not cont:
            return np.zeros(0), []
        overflow = len(ctx) + len(cont) - self.max_length
        if overflow > 0:
            ctx = ctx[overflow:]
        logits = self._forward_last([ctx + cont], len(cont) + 1)[0, :-1]
        if kind == "entropy":
            ents = entropy_from_logits(logits.cpu().numpy())
        else:
            lp = self.torch.log_softmax(logits, dim=-1)
            ents = lp[self.torch.arange(len(cont)), self.torch.tensor(cont, device=lp.device)].cpu().numpy()
        eff_cont = (context[len(context) - moved:] if moved else "") + continuation
        eff_ctx = context[: len(context) - moved]
        whole_ids = self.encode(eff_ctx + eff_cont)
        if whole_ids[: len(ctx_raw)] == ctx_raw and whole_ids[len(ctx_raw):] == cont:
            offs = self._offsets(eff_ctx + eff_cont, whole_ids)[len(ctx_raw):]
            base = len(eff_ctx) + moved
        else:
            offs = self._offsets(eff_cont, cont)
            base = moved
        offsets = [(max(0, a - base), max(0, b - base)) for a, b in offs]
        return ents, offsets

    def next_token_entropy(self, contexts: Sequence[str]) -> np.ndarray:
        seqs = []
        for c in contexts:
            ids = self._start_ids(c, self.encode(c))
            seqs.append(ids[-self.max_length:])
        out = np.zeros(len(contexts))
        for batch in token_budget_batches([len(s) for s in seqs], self.max_batch_tokens):
            logits = self._forward_last([seqs[i] for i in batch], 1)[:, 0]
            ents = entropy_from_logits(logits.cpu().numpy())
            for b, i in enumerate(batch):
                out[i] = ents[b]
        return out

    def generate(self, context: str, max_new_tokens: int = 256, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        return self.generate_batch([context], max_new_tokens, temperature, seed)[0]

    def generate_batch(self, contexts: Sequence[str], max_new_tokens: int = 256,
                       temperature: float = 0.0, seed: int | None = None) -> list[str]:
        torch = self.torch
        seqs = [self._start_ids(c, self.encode(c)) for c in contexts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, s in enumerate(seqs):
            ids[b, T - len(s):] = torch.tensor(s)
            mask[b, T - len(s):] = 1
        kw: dict[str, Any] = dict(max_new_tokens=max_new_tokens, pad_token_id=self.pad_id,
                                  do_sample=temperature > 0)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=1.0, top_k=0)
            if seed is not None:
                torch.manual_seed(seed)
        else:
            kw.update(temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **kw)
        return [self.tokenizer.decode(o[T:], skip_special_tokens=True) for o in out]
