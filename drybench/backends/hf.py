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


def _merge_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    if rest and rest[0]["role"] == "user":
        if isinstance(sys_text, list):  # lists of parts
            rest[0]["content"] = sys_text + [{"type": "text", "text": "\n\n"}] + rest[0]["content"]
        else:
            rest[0]["content"] = sys_text + "\n\n" + rest[0]["content"]
    else:
        rest.insert(0, {"role": "user", "content": sys_text})
    return rest


def _folder_chat_template(path: str) -> str | None:
    """A local model folder's chat template kept outside the tokenizer config (chat_template.jinja, or
    chat_template.json as processors write it)."""
    from pathlib import Path

    folder = Path(path)
    if (folder / "chat_template.jinja").exists():
        return (folder / "chat_template.jinja").read_text(encoding="utf-8")
    if (folder / "chat_template.json").exists():
        import json

        return json.loads((folder / "chat_template.json").read_text(encoding="utf-8")).get("chat_template")
    return None


def _load_tokenizer(path: str, trust_remote_code: bool = False):
    """AutoTokenizer; for a folder whose tokenizer config names a class this transformers lacks (one saved by a
    newer version: "TokenizersBackend"), its tokenizer.json with the config's special tokens and chat template."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    except ValueError as e:
        from pathlib import Path

        folder = Path(path)
        if "Tokenizer class" not in str(e) or not (folder / "tokenizer.json").exists():
            raise
        import json

        from transformers import PreTrainedTokenizerFast

        cfg_path = folder / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

        def text(v):
            return v.get("content") if isinstance(v, dict) else v

        special = {k: text(cfg[k]) for k in ("bos_token", "eos_token", "unk_token", "pad_token") if cfg.get(k)}
        tok = PreTrainedTokenizerFast(tokenizer_file=str(folder / "tokenizer.json"), **special)
        if isinstance(cfg.get("chat_template"), str):
            tok.chat_template = cfg["chat_template"]
        return tok


def _load_model(path: str, kw: dict[str, Any]):
    """(model, class name): a causal LM, or a vision-language model's image-text-to-text class (used for text)."""
    import transformers

    try:
        return transformers.AutoModelForCausalLM.from_pretrained(path, **kw), "AutoModelForCausalLM"
    except ValueError as e:
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None or "Unrecognized configuration class" not in str(e):
            raise
        return cls.from_pretrained(path, **kw), "AutoModelForImageTextToText"


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
        device_map: str | None = None,
        quantize: str | None = None,
    ) -> None:
        """device_map="auto": the model spread over the visible GPUs (a model larger than one GPU; needs accelerate).
        quantize="8bit" / "4bit": weights quantized with bitsandbytes. A vision-language model (Mistral Small 3.x,
        Ministral 3) is loaded with its image-text-to-text class and used for text only (v1.2.3)."""
        import torch
        import transformers

        self.torch = torch
        self.name = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = _load_tokenizer(model_name_or_path, trust_remote_code)
        if not getattr(self.tokenizer, "chat_template", None):  # the template kept apart, as processors keep it
            folder_template = _folder_chat_template(model_name_or_path)
            if folder_template:
                self.tokenizer.chat_template = folder_template
        if dtype == "auto":
            torch_dtype: Any = torch.float32 if self.device == "cpu" else "auto"
        else:
            torch_dtype = getattr(torch, _DTYPES[dtype])
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
        kw: dict[str, Any] = {dtype_kw: torch_dtype, "trust_remote_code": trust_remote_code}
        if quantize:
            if quantize not in ("8bit", "4bit"):
                raise ValueError("quantize must be '8bit' or '4bit'")
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if quantize == "8bit" else
                                         BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
            device_map = device_map or "auto"
        if device_map:
            kw["device_map"] = device_map
        self.model, self.model_class = _load_model(model_name_or_path, kw)
        if device_map:  # accelerate placed it: the inputs go where the embeddings are
            self.device = str(self.model.get_input_embeddings().weight.device)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.device_map, self.quantize = device_map, quantize
        self.revision = getattr(self.model.config, "_commit_hash", None)  # the hub commit the weights came from
        self.dropped_generation_config = self._plain_generation_config()
        self.max_batch_tokens = max_batch_tokens
        cfg = self.model.config
        cfg_len = (getattr(cfg, "max_position_embeddings", None)
                   or getattr(getattr(cfg, "text_config", None), "max_position_embeddings", None) or 4096)
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
        self._digit_ids: Any = None
        self._warned_cut = False

    def _plain_generation_config(self) -> dict:
        """Replace the model's generation config by one with its special tokens only, so that generate() is plain
        greedy decoding for every model: Qwen2.5-0.5B/1.5B-Instruct ship repetition_penalty 1.1 (7B: 1.05),
        which transformers applies even without sampling (v1.2). Returns the settings that were dropped."""
        from transformers import GenerationConfig

        old = self.model.generation_config
        tokens = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id")
        keep = {k: getattr(old, k, None) for k in tokens}
        self.model.generation_config = GenerationConfig(**{k: v for k, v in keep.items() if v is not None})
        return {k: v for k, v in old.to_diff_dict().items() if k not in tokens and k != "transformers_version"}

    def describe(self) -> dict:
        """What was loaded: name, hub commit, dtype, device, generation settings dropped from the model's config."""
        return {"name": self.name, "revision": self.revision, "dtype": str(self.model.dtype).replace("torch.", ""),
                "device": self.device, "dropped_generation_config": self.dropped_generation_config,
                **({"class": self.model_class} if self.model_class != "AutoModelForCausalLM" else {}),
                **({"device_map": self.device_map} if self.device_map else {}),
                **({"quantize": self.quantize} if self.quantize else {})}

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
            try:
                return self.tokenizer.apply_chat_template(_merge_system(messages), **kw)
            except Exception:  # vision-language templates want the content as a list of parts
                parts = [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in messages]
                try:
                    return self.tokenizer.apply_chat_template(parts, **kw)
                except Exception:
                    return self.tokenizer.apply_chat_template(_merge_system(parts), **kw)

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
            if not self._warned_cut:
                import warnings

                warnings.warn(f"{self.name}: a prompt of {len(ctx) + len(cont)} tokens is longer than "
                              f"{self.max_length}; its start is cut off", stacklevel=2)
                self._warned_cut = True
            ctx = ctx[overflow:]
        return ctx, cont, moved

    def _digit_token_ids(self):
        """Ids of the tokens whose text starts with a digit (no leading space): they continue a number."""
        if self._digit_ids is None:
            tok = self.tokenizer
            ids = [i for i in range(len(tok)) if tok.decode([i])[:1].isdigit()]
            self._digit_ids = self.torch.tensor(ids, dtype=self.torch.long)
        return self._digit_ids

    # ------------------------------------------------------------------ interface
    def continuation_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        return self._continuations(requests, terminated=False)

    def terminated_logprobs(self, requests: Sequence[tuple[str, str]]) -> np.ndarray:
        return self._continuations(requests, terminated=True)

    def _continuations(self, requests: Sequence[tuple[str, str]], terminated: bool) -> np.ndarray:
        """Sum of log p of each continuation; terminated: times p(the next token does not start with a digit)."""
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
                if terminated:  # the position after the continuation (left padding: the last one)
                    nxt = torch.log_softmax(logits[b, n_keep - 1], dim=-1)
                    digit = torch.logsumexp(nxt[self._digit_token_ids().to(nxt.device)], dim=0).item()
                    out[i] += float(np.log1p(-min(np.exp(digit), 1.0 - 1e-12)))
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
        return [g["text"] for g in self.generate_batch_info(contexts, max_new_tokens, temperature, seed)]

    def _eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        ids = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    def generate_batch_info(self, contexts: Sequence[str], max_new_tokens: int = 256,
                            temperature: float = 0.0, seed: int | None = None) -> list[dict]:
        """Per context: the text, the number of new tokens (up to and with the end token) and whether the
        model ended it (False: cut at max_new_tokens)."""
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
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **kw)
        eos = self._eos_ids()
        res = []
        for o in out:
            new = o[T:].tolist()
            end = next((i for i, t in enumerate(new) if t in eos), None)
            res.append({"text": self.tokenizer.decode(new, skip_special_tokens=True),
                        "n_tokens": len(new) if end is None else end + 1, "finished": end is not None})
        return res
