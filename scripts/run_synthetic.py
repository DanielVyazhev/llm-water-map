#!/usr/bin/env python3
"""The benchmark on synthetic operations reports, for a GPU server (the T4 notebook runs the same steps).

Once, in the repository root:
    python -m pip install -e ".[hf]" accelerate matplotlib

Run (re-running the same command resumes; a new --run name starts a fresh run):
    python scripts/run_synthetic.py --run runs/syn_v1 --n-per-cell 30 --seed 1 --models LABEL=HF_ID ...

Steps: items (15 knob cells) → checks of each reader where the truth is known (reader check, filler map on
simulated responses with planted filler) → responses and entropy signals of every model → scores with the
reader and the other readers (--readers2 LABEL=ID ...; --reader-items-per-cell LABEL=N scores the answers
to the first N items per cell only) → how each reader reads a missing fact (--missing-check) → the LLM
judge and Selective Context on the planted responses (--judges / --selective LABEL=ID) → analysis →
runs/<name>/tables_synthetic.md and runs/<name>.zip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_real as rr  # noqa: E402  (shared steps: run, versions, checks, baselines)

DEFAULT_MODELS = rr.DEFAULT_MODELS
ITEM_KEYS = ("n_per_cell", "seed", "reader")


def cell_of(item) -> str:
    """The knob cell of a synthetic item (its id is c<cell>-<n>)."""
    return item.item_id.split("-", 1)[0]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/synthetic_v1", help="output folder; the same folder resumes")
    ap.add_argument("--n-per-cell", type=int, default=30, help="items per knob cell (15 cells)")
    ap.add_argument("--seed", type=int, default=0, help="item seed (the T4 development runs used 0)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="label=hf_id ...")
    ap.add_argument("--reader", default=rr.READER)
    ap.add_argument("--readers2", nargs="*", default=None, metavar="LABEL=ID", help="other readers")
    ap.add_argument("--reader-items-per-cell", nargs="*", default=[], metavar="LABEL=N")
    ap.add_argument("--variants", nargs="+", default=["default", "concise"], choices=["default", "concise"])
    ap.add_argument("--dtype", default="bf16", help="bf16 on A100/H100; fp16 on T4/V100")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--reader-max-batch-tokens", type=int, default=8192)
    ap.add_argument("--sampled", type=int, default=0, help=">0: also sampled answer entropy (N samples)")
    ap.add_argument("--filler-check-n", type=int, default=5, help="items per cell of the planted-filler check (0 = skip)")
    ap.add_argument("--filler-check-seed", type=int, default=2026, help="its item seed (the development runs: 2026)")
    ap.add_argument("--free-model-cache", action="store_true",
                    help="delete an evaluated model's download after its responses and signals")
    ap.add_argument("--judges", nargs="*", default=None, metavar="LABEL=ID")
    ap.add_argument("--selective", nargs="*", default=None, metavar="LABEL=ID")
    ap.add_argument("--missing-check", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"], help="mock: a smoke test without models")
    args = ap.parse_args(argv)

    t_start = time.time()
    try:
        import torch

        gpu = torch.cuda.is_available()
        print("GPU:", torch.cuda.get_device_name(0) if gpu else "none")
    except ImportError:
        gpu = False
    if not gpu and not args.allow_cpu and args.backend == "hf":
        raise SystemExit("No GPU visible to torch (check CUDA_VISIBLE_DEVICES); --allow-cpu to force.")
    dtype = args.dtype if gpu else "auto"
    models, extra = rr.model_map(args.models), rr.label_map(args.readers2)
    subset = {k: int(v) for k, v in rr.label_map(args.reader_items_per_cell).items()}
    judges, selective = rr.label_map(args.judges), rr.label_map(args.selective)
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    print("Outputs:", out.resolve())
    cfg_path = out / "config.json"
    cfg = {"suite": "synthetic", "n_per_cell": args.n_per_cell, "seed": args.seed, "reader": args.reader,
           "readers2": extra, "reader_items_per_cell": subset, "models": models, "variants": args.variants,
           "judges": judges, "selective": selective}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = [k for k in ITEM_KEYS if old.get(k) != cfg[k]]
        if diff:
            raise SystemExit(f"{out} was made with other item settings ({', '.join(diff)}); "
                             "use a new --run name or the old settings.")
        cfg["models"] = {**old.get("models", {}), **models}
    cfg_path.write_text(json.dumps(cfg, indent=1))
    rr.write_hardware(out)
    rr.check_versions(out, args.reader, extra, seed=args.seed)
    items, resp, signals = out / "items.jsonl", out / "responses.jsonl", out / "signals.jsonl"
    template = ["--chat-template-kwargs", json.dumps({"enable_thinking": False})] if args.no_thinking else []
    common = ["--reader-backend", args.backend, "--reader-dtype", dtype,
              "--reader-max-batch-tokens", args.reader_max_batch_tokens]

    print("\n[1/7] items")
    if not items.exists():
        rr.run(rr.CLI + ["items", "--out", items, "--n-per-cell", args.n_per_cell, "--seed", args.seed])
    print(f"    {sum(1 for _ in open(items))} items")

    print("\n[2/7] checks of the reader where the truth is known: reader check, filler map on planted filler")
    readers = {"main": (args.reader, out)} | {lab: (mid, out / f"reader2_{lab}") for lab, mid in extra.items()}
    for label, (model_id, sub) in readers.items():
        sub.mkdir(exist_ok=True)
        if not (sub / "reader_check.json").exists():
            rr.run(rr.CLI + ["check-reader", "--items", items, "--reader", model_id, "--n", 40,
                             "--out", sub / "reader_check.json"] + common)
        rr.filler_check(args, model_id, sub / "filler_check", dtype)

    print("\n[3/7] responses + entropy signals")
    for label, model_id in models.items():
        rr.run(rr.CLI + ["generate", "--items", items, "--out", resp, "--backend", args.backend, "--model", model_id,
                         "--model-name", label, "--variant", *args.variants, "--dtype", dtype,
                         "--batch-size", args.batch_size, "--max-new-tokens", args.max_new_tokens, *template])
        rr.run(rr.CLI + ["signals", "--items", items, "--responses", resp, "--out", signals, "--backend", args.backend,
                         "--model", model_id, "--model-name", label, "--dtype", dtype, "--sampled", args.sampled,
                         *template])
        if args.free_model_cache:
            rr.free_model_cache(model_id, {args.reader, *extra.values(), *judges.values(), *selective.values()})

    print("\n[4/7] scores")
    for label, (model_id, sub) in readers.items():
        answers = resp
        if subset.get(label):
            answers = rr.subset_responses(items, resp, sub / "responses_scored.jsonl", subset[label], cell_of)
            print(f"    reader {label} scores the answers to the first {subset[label]} items per cell")
        rr.run(rr.CLI + ["score", "--items", items, "--responses", answers, "--out", sub / "scored.jsonl",
                         "--reader", model_id, "--mode", "loo", "--ablation-abs-tol", 2.0] + common)

    print("\n[5/7] how each reader reads a missing fact; baselines")
    if args.missing_check:
        rr.missing_checks(out, {lab: mid for lab, (mid, _) in readers.items()}, args.backend, dtype)
    if judges or selective:
        rr.run_baselines(out, ["planted_synthetic"], judges, selective, args.backend, dtype,
                         args.reader_max_batch_tokens)

    print("\n[6/7] analysis")
    for label, (model_id, sub) in readers.items():
        rr.run(rr.CLI + ["analyze", "--items", items, "--scored", sub / "scored.jsonl", "--signals", signals,
                         "--out-dir", sub / "report", "--n-boot", 500])

    print("\n[7/7] tables")
    rr.run([sys.executable, "-m", "drybench.report", out, "--quiet"])
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"\nTables: {out / 'tables_synthetic.md'}\nWhole run: {archive}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
