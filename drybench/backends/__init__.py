from drybench.backends.base import LMBackend, split_context_continuation
from drybench.backends.mock import MockBackend


def _echo(context: str) -> str:
    """Mock generation: 'Sure!' + the first two sentences of the document in the prompt + a closing."""
    import re

    m = re.search(r"User: [^\n]*:\n(.*?)\n\nRequest:", context, re.S)
    text = " ".join((m.group(1) if m else context).split())
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return "Sure! " + " ".join(sents[:2]) + " I hope this helps!"


def make_backend(kind: str, path: str | None = None, **kwargs) -> LMBackend:
    """Factory: kind in {"hf", "onnx-gpt2", "mock"}."""
    if kind == "hf":
        from drybench.backends.hf import HFBackend

        if not path:
            raise ValueError("hf backend needs a model name or path")
        return HFBackend(path, **kwargs)
    if kind == "onnx-gpt2":
        from drybench.backends.onnx_gpt2 import OnnxGPT2Backend

        if not path:
            raise ValueError("onnx-gpt2 backend needs the assets directory")
        return OnnxGPT2Backend.from_dir(path, **kwargs)
    if kind == "mock":
        if path == "echo":  # smoke tests: answer with the document's first sentences plus filler
            kwargs.setdefault("generate_fn", _echo)
        return MockBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}")


__all__ = ["LMBackend", "MockBackend", "make_backend", "split_context_continuation"]
