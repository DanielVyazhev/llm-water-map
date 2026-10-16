#!/usr/bin/env python3
"""Can this machine run these models, and how do they behave in the benchmark's roles?

    python scripts/probe_models.py MODEL_DIR [MODEL_DIR ...] [--out probe.jsonl] [--dtype bf16] [--sha256 DIR]

Each model is probed in a process of its own (the GPU is freed in between); nothing is downloaded: the models are
local folders and the Hugging Face hub is off. Per model:
  weights      architecture, size of the weight files, quantization, vision part, chat template (config files);
  loads        whether drybench's HF backend loads it (AutoModelForCausalLM; else whether an image-text-to-text
               class does), the GPU memory it takes;
  answer       as an evaluated model: its greedy answer to a synthetic report request (a plain answer with the
               requested facts, not a reasoning trace);
  reader       the probability of each fact read from the gold answer (≥ 0.5: the reader reads it);
  judge        the labels it gives an opener and the gold answer's fragments, and the probability mass on the
               three label words at the start of its reply (low: it wanted to say something else).
--sha256 hashes the weight files of one model (to compare with the hub's, e.g. the main reader's).
One JSON line per model in --out; a table at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

REASONING = re.compile(r"\s*(?:<think>|<\|channel\|>|analysis\b|thinking\b|okay,? let|alright\b|we need to|let me )",
                       re.I)


def weights(path: Path) -> dict:
    """What the config files say, without loading anything."""
    cfg = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    files = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    tk = path / "tokenizer_config.json"
    chat = ((path / "chat_template.jinja").exists() or (path / "chat_template.json").exists()
            or (tk.exists() and "chat_template" in json.loads(tk.read_text())))
    return {"name": path.name, "architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
            "weights_gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
            "quantization": (cfg.get("quantization_config") or {}).get("quant_method"),
            "vision": "vision_config" in cfg, "chat_template": bool(chat),
            "stored_dtype": cfg.get("torch_dtype") or cfg.get("dtype") or (cfg.get("text_config") or {}).get("torch_dtype")}


def sha256(path: Path) -> dict:
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        out[f.name] = h.hexdigest()
    return out


def probe_one(path: Path, dtype: str) -> dict:
    """All checks of one model (run in a process of its own)."""
    import numpy as np
    import torch

    from drybench.backends.hf import HFBackend
    from drybench.items import evaluated_model_messages, generate_items
    from drybench.judge import judge_fragments
    from drybench.reader import Reader
    from drybench.segment import segment

    row = weights(path)
    cuda = torch.cuda.is_available()
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        row.update(gpu=torch.cuda.get_device_name(0), gpu_total_gb=round(total, 1))
        need = row["weights_gb"] / (2 if row.get("stored_dtype") == "float32" and dtype != "fp32" else 1)
        if need > 0.95 * total and not row["quantization"]:
            row["error"] = f"weights {need:.0f} GB > this GPU's {total:.0f} GB"
            return row
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        be = HFBackend(str(path), dtype=dtype, max_batch_tokens=4096)
    except Exception as e:  # noqa: BLE001 - reported
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        row["image_text_to_text"] = _loads_as_image_text_to_text(path, dtype)
        return row
    row.update(loads=True, load_s=round(time.time() - t0))
    item = generate_items(1, seed=0, cells=[{"k": 2, "d": 0}])[0]
    probes = item.required_probes
    try:  # as an evaluated model
        t0 = time.time()
        g = be.generate_batch_info([be.format_chat(evaluated_model_messages(item))], max_new_tokens=200)[0]
        vals = [(p.value or p.answers[0]).lower() for p in probes]
        row["answer"] = {"text": g["text"][:500], "tokens": g["n_tokens"], "finished": g["finished"],
                         "seconds": round(time.time() - t0, 1),
                         "facts_in_text": sum(v in g["text"].lower() for v in vals) / len(vals),
                         "reasoning_like": bool(REASONING.match(g["text"]))}
    except Exception as e:  # noqa: BLE001
        row["answer"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    try:  # as a reader
        reader = Reader.for_backend(be)
        p = np.exp(reader.probe_logprobs(item.request, [item.gold_minimal], probes)[0])
        row["reader"] = {"chat": reader.config.use_chat, "missing_style": reader.config.missing_style,
                         "p_gold": [round(float(x), 3) for x in p], "reads": bool((p >= 0.5).all())}
    except Exception as e:  # noqa: BLE001
        row["reader"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    try:  # as a judge: an opener, then the gold answer's fragments
        frags = ["Great question!"] + [s.text for s in segment(item.gold_minimal)]
        res = judge_fragments(be, item.document, item.request, frags, "Report")
        row["judge"] = {"labels": [r["label"] for r in res], "mass": [round(r["mass"], 3) for r in res],
                        "right": res[0]["label"] == "filler" and all(r["label"] == "needed" for r in res[1:])}
    except Exception as e:  # noqa: BLE001
        row["judge"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    if cuda:
        row["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    return row


def _loads_as_image_text_to_text(path: Path, dtype: str) -> str:
    """For a model AutoModelForCausalLM rejects: does an image-text-to-text class take its config (vision models)?
    Only the config is read."""
    try:
        import transformers

        cfg = transformers.AutoConfig.from_pretrained(str(path))
        cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if cls is None:
            return "no AutoModelForImageTextToText in this transformers"
        mapping = cls._model_mapping
        return f"image-text-to-text: {mapping[type(cfg)].__name__}" if type(cfg) in mapping else "no class for it"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:200]}"


def table(rows: list[dict]) -> str:
    head = ["model", "architecture", "GB", "loads", "answer: facts / plain", "reader p(gold)",
            "judge labels (mass)", "GPU GB"]
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        a, rd, j = r.get("answer") or {}, r.get("reader") or {}, r.get("judge") or {}
        lines.append(" | ".join([
            r["name"], ",".join(r.get("architectures") or ["?"]), str(r.get("weights_gb")),
            "yes" if r.get("loads") else f"no: {r.get('error', '')[:60]}",
            f"{a['facts_in_text']:.0%} / {'reasoning' if a['reasoning_like'] else 'plain'}" if "facts_in_text" in a else a.get("error", "–")[:40],
            ", ".join(f"{x:.2f}" for x in rd["p_gold"]) if "p_gold" in rd else rd.get("error", "–")[:40],
            (" ".join(j["labels"]) + f" ({min(j['mass']):.2f})") if "labels" in j else j.get("error", "–")[:40],
            str(r.get("gpu_mem_gb", "–"))]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="local model folders")
    ap.add_argument("--out", default="probe.jsonl")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sha256", action="append", default=[], help="a model folder whose weight files to hash")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # internal: probe models[0] here
    args = ap.parse_args(argv)
    if args.one:
        print("PROBE " + json.dumps(probe_one(Path(args.models[0]), args.dtype)), flush=True)
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TRANSFORMERS_VERBOSITY="error",
               HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join([str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    info = subprocess.run([sys.executable, "-c", "import sys, torch, transformers; print(sys.version.split()[0], "
                           "torch.__version__, transformers.__version__, torch.version.cuda, "
                           "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"],
                          capture_output=True, text=True, env=env)
    print("python, torch, transformers, CUDA, GPU:", (info.stdout or info.stderr).strip()[-300:], flush=True)
    rows = []
    out = Path(args.out).expanduser()
    for m in args.models:
        path = Path(m).expanduser()
        t0 = time.time()
        print(f"\n== {path}", flush=True)
        if not (path / "config.json").exists():
            row = {"name": path.name, "error": "no config.json in this folder"}
        else:
            p = subprocess.run([sys.executable, __file__, str(path), "--one", "--dtype", args.dtype],
                               capture_output=True, text=True, env=env)
            got = [x for x in p.stdout.splitlines() if x.startswith("PROBE ")]
            row = json.loads(got[-1][6:]) if got else {**weights(path), "error": (p.stderr or p.stdout)[-400:]}
        row["probe_s"] = round(time.time() - t0)
        if str(path) in [str(Path(x).expanduser()) for x in args.sha256]:
            row["sha256"] = sha256(path)
        print(json.dumps(row, ensure_ascii=False)[:1500], flush=True)
        rows.append(row)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n" + table(rows))
    print(f"\nAll: {out}")


if __name__ == "__main__":
    main()
