#!/usr/bin/env python3
"""Can this machine run these models, and how do they behave in the benchmark's roles?

    python scripts/probe_models.py MODEL_DIR [MODEL_DIR ...] [--out probe.jsonl] [--dtype bf16] [--sha256 DIR]
                                   [--static DIR]

Each model is probed in a process of its own (the GPU is freed in between); nothing is downloaded: the models are
local folders and the Hugging Face hub is off. First the machine: library versions and GPUs. Per model:
  weights      origin (the config's _name_or_path), architecture, size of the weights transformers loads (not a
               second copy in another format), stored dtype, vision part, chat template, tokenizer class;
  loads        whether drybench's HF backend loads it: on one GPU, spread over several (device_map auto), or with
               8-bit weights (bitsandbytes) when it fits neither; a vision-language model through its
               image-text-to-text class; the GPU memory it takes;
  answer       as an evaluated model: its greedy answer to a synthetic report request (a plain answer with the
               requested facts, not a reasoning trace);
  reader       the probability of each fact read from the gold answer (≥ 0.5: the reader reads it);
  judge        the labels it gives an opener and the gold answer's fragments, and the probability mass on the
               three label words at the start of its reply (low: it wanted to say something else).
--sha256 hashes the weight files of a model; --static only describes a model (no loading).
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


def weight_files(path: Path) -> list[Path]:
    """The weight files transformers loads: the index's, else model*.safetensors (or .bin); never Mistral's
    consolidated.* copy of the same weights."""
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        return sorted({path / f for f in json.loads(idx.read_text())["weight_map"].values()})
    return (sorted(path.glob("model*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
            or [f for f in sorted(path.glob("*.safetensors")) if not f.name.startswith("consolidated")])


def weights(path: Path) -> dict:
    """What the config files say, without loading anything. The size counts the weights transformers loads (the
    index's files, else model*.safetensors), not a second copy in another format (Mistral's consolidated.*)."""
    cfg = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    text = cfg.get("text_config") or {}
    files = weight_files(path)
    stored = cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype") or text.get("dtype")
    size = sum(f.stat().st_size for f in files if f.exists()) / 1e9
    tk = path / "tokenizer_config.json"
    tk_cfg = json.loads(tk.read_text()) if tk.exists() else {}
    chat = (path / "chat_template.jinja").exists() or (path / "chat_template.json").exists() or "chat_template" in tk_cfg
    return {"name": path.name, "origin": cfg.get("_name_or_path") or None, "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"), "weights_gb": round(size, 1), "stored_dtype": stored,
            "bf16_gb": round(size / 2 if stored == "float32" else size, 1),
            "layers": cfg.get("num_hidden_layers") or text.get("num_hidden_layers"),
            "hidden": cfg.get("hidden_size") or text.get("hidden_size"),
            "quantization": (cfg.get("quantization_config") or {}).get("quant_method"),
            "vision": "vision_config" in cfg, "chat_template": bool(chat),
            "tokenizer_class": tk_cfg.get("tokenizer_class"),
            "second_copy": any(path.glob("consolidated*.safetensors"))}


def environment() -> dict:
    """Library versions and GPUs of this machine."""
    out = {"python": sys.version.split()[0]}
    for lib in ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers"):
        try:
            out[lib] = __import__(lib).__version__
        except Exception:  # noqa: BLE001 - not installed
            out[lib] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["gpus"] = [[torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)]
                       for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        out["gpus"] = []
    return out


def load_mode(row: dict, env: dict) -> tuple[str | None, str]:
    """How a model fits: on one GPU (None), spread over several ("auto"), quantized to 8 bit ("8bit"), or not:
    (mode, why)."""
    gpus = [gb for _, gb in env.get("gpus", [])]
    need = row["bf16_gb"] * 1.1  # weights and the activations of short prompts
    if not gpus or row.get("quantization") or need <= 0.95 * gpus[0]:
        return None, "one GPU"
    if len(gpus) > 1 and env.get("accelerate") and need <= 0.9 * sum(gpus):
        return "auto", f"{len(gpus)} GPUs"
    if env.get("bitsandbytes") and env.get("accelerate") and need / 2 <= 0.95 * gpus[0]:
        return "8bit", "8-bit weights on one GPU"
    return "no", (f"needs ~{need:.0f} GB; GPUs {', '.join(f'{g:.0f}' for g in gpus)} GB"
                  + ("" if env.get("bitsandbytes") else "; no bitsandbytes for 8 bit"))


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
    env = environment()
    cuda = torch.cuda.is_available()
    mode, why = load_mode(row, env)
    row["load_mode"] = why
    if mode == "no":
        row["error"] = why
        return row
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        be = HFBackend(str(path), dtype=dtype, max_batch_tokens=4096,
                       device_map="auto" if mode == "auto" else None, quantize="8bit" if mode == "8bit" else None)
    except Exception as e:  # noqa: BLE001 - reported
        row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return row
    row["class"] = be.model_class
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
        row["gpu_mem_gb"] = round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
                                  / 1e9, 1)
    return row


def table(rows: list[dict]) -> str:
    head = ["model", "origin", "architecture", "GB (bf16)", "loads", "answer: facts / plain", "reader p(gold)",
            "judge labels (mass)", "GPU GB"]
    lines = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        a, rd, j = r.get("answer") or {}, r.get("reader") or {}, r.get("judge") or {}
        if r.get("static"):
            loads = "not loaded (--static)"
        elif r.get("loads"):
            loads = "yes" + (f" ({r['load_mode']})" if r.get("load_mode") not in (None, "one GPU") else "")
        else:
            loads = f"no: {r.get('error', '')[:70]}"
        lines.append(" | ".join([
            r["name"], str(r.get("origin") or "–"), ",".join(r.get("architectures") or ["?"]), str(r.get("bf16_gb")),
            loads,
            f"{a['facts_in_text']:.0%} / {'reasoning' if a['reasoning_like'] else 'plain'}" if "facts_in_text" in a
            else a.get("error", "–")[:40],
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
    ap.add_argument("--static", action="append", default=[], help="a model folder only to describe, not to load")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # internal: probe models[0] here
    args = ap.parse_args(argv)
    if args.one:
        print("PROBE " + json.dumps(probe_one(Path(args.models[0]), args.dtype)), flush=True)
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TRANSFORMERS_VERBOSITY="error",
               HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=os.pathsep.join([str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    machine = environment()
    print("this machine:", json.dumps(machine), flush=True)
    rows = []
    out = Path(args.out).expanduser()
    for m in args.models:
        path = Path(m).expanduser()
        t0 = time.time()
        print(f"\n== {path}", flush=True)
        if not (path / "config.json").exists():
            row = {"name": path.name, "error": "no config.json in this folder"}
        elif str(path) in [str(Path(x).expanduser()) for x in args.static]:
            row = {**weights(path), "static": True}
        else:
            p = subprocess.run([sys.executable, __file__, str(path), "--one", "--dtype", args.dtype],
                               capture_output=True, text=True, env=env)
            got = [x for x in p.stdout.splitlines() if x.startswith("PROBE ")]
            row = json.loads(got[-1][6:]) if got else {**weights(path), "error": (p.stderr or p.stdout)[-400:]}
        row["probe_s"] = round(time.time() - t0)
        row["machine"] = machine
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
