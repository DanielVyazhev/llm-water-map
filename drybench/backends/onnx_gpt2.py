"""GPT-2 (124M) via onnxruntime: a CPU-only backend used for the local pilot.

Weights: the ONNX Model Zoo export `gpt2-lm-head-10.onnx` (input `input1` of shape
[1, batch, seq]); tokenizer: GPT-2 byte-level BPE files (vocab.json, merges.txt).
There is no attention-mask input, so batches are right-padded and only logits at
real (unpadded) positions are read, which is exact for a causal model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from drybench.backends.base import (
    LMBackend,
    entropy_from_logits,
    log_softmax,
    split_context_continuation,
    token_budget_batches,
)

EOS_ID = 50256
MAX_LEN = 1024


class OnnxGPT2Backend(LMBackend):
    name = "gpt2-onnx"

    def __init__(self, model_path: str, vocab_path: str, merges_path: str,
                 max_batch_tokens: int = 2048, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import ByteLevelBPETokenizer

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.tok = ByteLevelBPETokenizer(vocab_path, merges_path)
        self.max_batch_tokens = max_batch_tokens

    @classmethod
    def from_dir(cls, assets_dir: str, **kwargs) -> "OnnxGPT2Backend":
        d = Path(assets_dir)
        return cls(str(d / "gpt2-lm-head-10.onnx"), str(d / "gpt2-vocab.json"),
                   str(d / "gpt2-merges.txt"), **kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def _logits(self, seqs: list[list[int]]) -> list[np.ndarray]:
        T = max(len(s) for s in seqs)
        arr = np.full((1, len(seqs), T), EOS_ID, dtype=np.int64)
        for i, s in enumerate(seqs):
            arr[0, i, : len(s)] = s
        out = self.sess.run(["output1"], {"input1": arr})[0][0]
        return [out[i, : len(s)] for i, s in enumerate(seqs)]

    def _prepare(self, context: str, continuation: str) -> tuple[list[int], list[int], int]:
        ctx, cont, moved = split_context_continuation(self.encode, context, continuation)
        if not ctx:
            ctx = [EOS_ID]
        overflow = len(ctx) + len(cont) - MAX_LEN
        if overflow > 0:
            ctx = ctx[overflow:]
        return ctx, cont, moved

    def continuation_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        enc = [self._prepare(c, k) for c, k in requests]
        out = np.zeros(len(requests))
        lengths = [len(c) + len(k) for c, k, _ in enc]
        for batch in token_budget_batches(lengths, self.max_batch_tokens):
            batch = [i for i in batch if enc[i][1]]
            if not batch:
                continue
            logits = self._logits([enc[i][0] + enc[i][1] for i in batch])
            for b, i in enumerate(batch):
                c, k, _ = enc[i]
                lp = log_softmax(logits[b][len(c) - 1: len(c) + len(k) - 1])
                out[i] = float(lp[np.arange(len(k)), k].sum())
        return out

    def token_entropies(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        return self._token_scores(context, continuation, "entropy")

    def token_logprobs(self, context: str, continuation: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        return self._token_scores(context, continuation, "logprob")

    def _token_scores(self, context: str, continuation: str, kind: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        ctx, cont, moved = self._prepare(context, continuation)
        if not cont:
            return np.zeros(0), []
        logits = self._logits([ctx + cont])[0][len(ctx) - 1: len(ctx) + len(cont) - 1]
        if kind == "entropy":
            ents = entropy_from_logits(logits)
        else:
            ents = log_softmax(logits)[np.arange(len(cont)), cont]
        eff_cont = (context[len(context) - moved:] if moved else "") + continuation
        enc = self.tok.encode(eff_cont)
        if enc.ids == cont:
            offs = enc.offsets
        else:  # boundary merge changed tokens: approximate by cumulative decoding
            offs, prev = [], 0
            for j in range(len(cont)):
                cur = len(self.tok.decode(cont[: j + 1]))
                offs.append((prev, cur))
                prev = cur
        offsets = [(max(0, a - moved), max(0, b - moved)) for a, b in offs]
        return ents, offsets

    def next_token_entropy(self, contexts: Sequence[str]) -> np.ndarray:
        seqs = [(self.encode(c) or [EOS_ID])[-MAX_LEN:] for c in contexts]
        out = np.zeros(len(contexts))
        for batch in token_budget_batches([len(s) for s in seqs], self.max_batch_tokens):
            logits = self._logits([seqs[i] for i in batch])
            for b, i in enumerate(batch):
                out[i] = float(entropy_from_logits(logits[b][-1]))
        return out

    def generate(self, context: str, max_new_tokens: int = 64, temperature: float = 0.0,
                 seed: int | None = None) -> str:
        rng = np.random.default_rng(seed)
        ids = (self.encode(context) or [EOS_ID])[-(MAX_LEN - max_new_tokens):]
        new: list[int] = []
        for _ in range(max_new_tokens):  # no KV cache in this export: fine for short outputs
            logits = self._logits([ids + new])[0][-1].astype(np.float64)
            if temperature > 0:
                p = np.exp(log_softmax(logits / temperature))
                nxt = int(rng.choice(len(p), p=p / p.sum()))
            else:
                nxt = int(np.argmax(logits))
            if nxt == EOS_ID:
                break
            new.append(nxt)
        return self.tok.decode(new)
