"""Real GPT-2 weights: ONNX backend vs HF backend (converted from the same ONNX file).

Runs only when DRYBENCH_ASSETS points to a directory with gpt2-lm-head-10.onnx,
gpt2-vocab.json, gpt2-merges.txt and (for the HF half) a gpt2-hf/ checkpoint.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_onnx_backend_sanity(assets_dir):
    from drybench.backends.onnx_gpt2 import OnnxGPT2Backend

    b = OnnxGPT2Backend.from_dir(str(assets_dir))
    lp = b.continuation_logprobs([("Shipment KX-1234 was handled by Ostrell Freight. Question: Which "
                                   "carrier handled shipment KX-1234? Answer:", " Ostrell Freight"),
                                  ("Question: Which carrier handled shipment KX-1234? Answer:",
                                   " Ostrell Freight")])
    assert lp[0] > lp[1] + 5  # copying from context is far more likely than guessing
    ents, offs = b.token_entropies("Report:", " The carrier was Ostrell Freight.")
    assert len(ents) == len(offs) and offs[-1][1] == len(" The carrier was Ostrell Freight.")


def test_onnx_and_hf_agree(assets_dir):
    pytest.importorskip("torch")
    if not (assets_dir / "gpt2-hf" / "config.json").exists():
        pytest.skip("gpt2-hf checkpoint not found")
    from drybench.backends.hf import HFBackend
    from drybench.backends.onnx_gpt2 import OnnxGPT2Backend

    onnx = OnnxGPT2Backend.from_dir(str(assets_dir))
    hf = HFBackend(str(assets_dir / "gpt2-hf"), device="cpu", prepend_bos=False)
    reqs = [("Message: The carrier was Ostrell Freight.\nQuestion: Which carrier?\nAnswer:", " Ostrell Freight"),
            ("Great question! The delay was caused by", " a customs audit."),
            ("Hi", " there")]
    assert np.allclose(onnx.continuation_logprobs(reqs), hf.continuation_logprobs(reqs), atol=2e-3)
    e1, o1 = onnx.token_entropies("Report:", " It departed from Port Rosfen.")
    e2, o2 = hf.token_entropies("Report:", " It departed from Port Rosfen.")
    assert np.allclose(e1, e2, atol=2e-3) and o1 == o2
