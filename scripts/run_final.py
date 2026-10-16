#!/usr/bin/env python3
"""The confirmation run of the paper: one frozen version of the code, new items, new models, every check.

    python scripts/run_final.py --run runs/final_v1 \
        --dev-items results/dev_v1.0/real/items.jsonl runs/real_t4_n15_v12/items.jsonl [settings]
    python scripts/run_final.py --print-hash      # version and code hash, fixed before the run

Everything was designed and tuned on the development runs (the T4 runs: three models of 0.5–1.7B, synthetic
items of seed 0, the real-text items in --dev-items, every development run). This run checks the results on
what those runs did not see: synthetic items of another seed, real-text items built from other source texts (no
paragraph, MuSiQue question or step, or contract of the development items), and other, larger models; the code
is the version recorded in protocol.json at the first start (rerunning with another version stops, see
docs/PROTOCOL.md). Poddar et al. is not a suite of this run: the development human-label check used every
question of it that could be an item.

A model may be a local folder instead of a Hugging Face id. When every model is one, nothing is fetched from the
hub, and the datasets must already be in --data (squad2/dev-v2.0.json, musique/musique_full_v1.0_dev.jsonl,
cuad/CUADv1.json); they must be the files the development runs used (their config.json records the sha256).

Steps (rerunning resumes):
  1. synthetic reports  → RUN/synthetic   (scripts/run_synthetic.py)
  2. real texts         → RUN/real        (scripts/run_real.py)
  3. human-label samples → RUN/annotation: sample.jsonl / key.jsonl (item by item, both suites, no missing-fact
     items; every reader's filler map, and the judges and Selective Context on it) and abstain_sample.csv
     (answers to missing-fact items, for held-out labels of the abstention rule)
  4. the confirmatory tests without the human labels → RUN/confirmatory.md (drybench confirm; with the labels:
     drybench confirm --run RUN --labels ...)
  5. RUN.zip (everything) and RUN_annotation.zip (what the annotators need)
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
sys.path.insert(0, str(REPO))  # protocol.json records the package these scripts run (run_real.ENV), not another install

import run_real as rr  # noqa: E402

DEFAULT_MODELS = [                                    # three families, 3–14B; none of them is the main reader
    "llama3.2-3b=meta-llama/Llama-3.2-3B-Instruct",  # gated: HF_TOKEN
    "phi4-mini=microsoft/Phi-4-mini-instruct",
    "qwen2.5-14b=Qwen/Qwen2.5-14B-Instruct",
]
# the second and the large reader; both are also evaluated models (their scores of their own answers are secondary)
DEFAULT_READERS2 = ["phi4-mini=microsoft/Phi-4-mini-instruct", "qwen2.5-14b=Qwen/Qwen2.5-14B-Instruct"]
DEFAULT_JUDGES = ["qwen2.5-14b=Qwen/Qwen2.5-14B-Instruct"]
DEFAULT_SELECTIVE = ["qwen2.5-3b=Qwen/Qwen2.5-3B-Instruct"]
# Poddar et al. is left out: every question usable as an item was used by the development human-label check
DEFAULT_SUITES = ["squad2", "musique", "cuad"]
DATA_FILES = {"squad2": "squad2/dev-v2.0.json", "musique": "musique/musique_full_v1.0_dev.jsonl",
              "cuad": "cuad/CUADv1.json"}
PROTOCOL_KEYS = ("version", "code", "scoring", "signals", "seed", "filler_check_seed", "n_per_cell_synthetic",
                 "n_per_cell_real", "real_check_n", "reader", "readers2", "reader_items_per_cell", "models",
                 "variants", "judges", "selective", "suites", "max_new_tokens", "dtype", "reader_dtype",
                 "dev_items_sha256", "annotation", "abstain_n")
# the code that runs: the package and the scripts of a run, the same files in the repository and in the Colab
# notebook (scripts/make_colab.py ships them), so both give the same hash
CODE_SCRIPTS = ("scripts/run_final.py", "scripts/run_real.py", "scripts/run_synthetic.py",
                "scripts/pilot_validation.py")


def code_files() -> list[Path]:
    return sorted(list((REPO / "drybench").rglob("*.py")) + [REPO / f for f in CODE_SCRIPTS])


def code_hash() -> str:
    """sha1 of the package and the run scripts (code_files): the version string alone would not notice a
    rebuilt package with other code."""
    import hashlib

    h = hashlib.sha1()
    for f in code_files():
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_local(model_id: str) -> bool:
    """A model given as a folder (absolute, relative or ~), not as a Hugging Face id."""
    return Path(model_id).expanduser().is_dir() or model_id.startswith(("/", "~", "./", "../"))


def preflight(ids: list[str]) -> None:
    """Every local model folder has its config.json; every hub model can be downloaded with this token (gated
    models fail here, not after hours)."""
    bad, hub = [], []
    for mid in sorted(set(ids)):
        if is_local(mid):
            if not (Path(mid).expanduser() / "config.json").exists():
                bad.append(f"{mid} (no config.json)")
        else:
            hub.append(mid)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hub = []
    for mid in hub:
        try:
            hf_hub_download(mid, "config.json")
        except Exception as e:  # noqa: BLE001 - report every model that fails
            bad.append(f"{mid} ({type(e).__name__})")
    if bad:
        raise SystemExit("Cannot use: " + "; ".join(bad) + ". A local model folder needs its config.json; a hub "
                         "model needs HF_TOKEN and access to it. Fix the path or drop the model.")


def offline(ids: list[str]) -> bool:
    """Every model a local folder: the run makes no calls to the Hugging Face hub (a server without internet)."""
    if ids and all(is_local(m) for m in ids):
        rr.ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return True
    return False


def check_data(data: Path, suites: list[str], dev_items: list[str], need_all: bool) -> None:
    """The dataset files of the suites are there (need_all: nothing may be downloaded) and are the files the
    development runs used (sha256 in the config.json next to their items)."""
    wanted = {DATA_FILES[s] for s in suites if s in DATA_FILES}
    missing = [str(data / f) for f in sorted(wanted) if not (data / f).exists()]
    if missing and need_all:
        raise SystemExit("Datasets not found (nothing is downloaded when every model is a local folder; copy the "
                         "development runs' data folder): " + ", ".join(missing))
    differ = []
    for p in dev_items:
        cfg = Path(p).parent / "config.json"
        recorded = json.loads(cfg.read_text()).get("data") or {} if cfg.exists() else {}
        for path, digest in recorded.items():
            here = data / Path(*Path(path).parts[-2:])  # e.g. squad2/dev-v2.0.json
            if here.relative_to(data).as_posix() in wanted and here.exists() and rr.sha256(here) != digest:
                differ.append(f"{here} (sha256 differs from {cfg})")
    if differ:
        raise SystemExit("Not the dataset files of the development runs, so their source texts may not be found "
                         "and left out: " + "; ".join(differ))


def protocol(out: Path, args: argparse.Namespace, write: bool = True) -> dict:
    """protocol.json: the frozen settings of the run, written at its first start and checked at every rerun.
    A change allowed with --allow-protocol-change is applied and logged under "changes" (v1.2; before, the old
    values stayed and nothing was logged)."""
    import drybench

    cur = {"version": drybench.__version__, "code": code_hash(), "scoring": drybench.SCORING_VERSION,
           "signals": drybench.SIGNALS_VERSION, "seed": args.seed, "filler_check_seed": 2026 + args.seed,
           "n_per_cell_synthetic": args.n_per_cell_synthetic, "n_per_cell_real": args.n_per_cell_real,
           "real_check_n": args.real_check_n, "reader": args.reader, "readers2": rr.label_map(args.readers2),
           "reader_items_per_cell": args.reader_items_per_cell, "models": rr.model_map(args.models),
           "variants": args.variants, "judges": rr.label_map(args.judges),
           "selective": rr.label_map(args.selective), "suites": sorted(args.suites),
           "max_new_tokens": args.max_new_tokens, "dtype": args.dtype, "reader_dtype": args.reader_dtype,
           "decoding": "plain greedy",
           "dev_items": [str(p) for p in args.dev_items],
           "dev_items_sha256": sorted(file_sha256(p) for p in args.dev_items),
           "annotation": {"sources": args.annotation_sources, "items_per_source": args.annotation_items,
                          "per_item": args.annotation_per_item}, "abstain_n": args.abstain_n,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = out / "protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = [k for k in PROTOCOL_KEYS if old.get(k) != cur.get(k)]
        if diff and not args.allow_protocol_change:
            raise SystemExit(f"{path} was started with other {', '.join(diff)} (package v{old.get('version')}, "
                             f"now v{cur['version']}). A confirmation run keeps its protocol: use the same package "
                             "and settings, or a new --run, or --allow-protocol-change (and report it).")
        changes = list(old.get("changes") or [])
        if diff:
            changes.append({"at": cur["started"], "changed": {k: {"from": old.get(k), "to": cur.get(k)} for k in diff}})
            print(f"PROTOCOL CHANGED ({', '.join(diff)}): logged in {path} under 'changes'; report it")
        cur = {**cur, "started": old.get("started", cur["started"]), "changes": changes}
    if write:
        path.write_text(json.dumps(cur, indent=1))
    return cur


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/final_v1")
    ap.add_argument("--data", default=None, help="dataset downloads (default: RUN/../data)")
    ap.add_argument("--dev-items", nargs="+", default=None,
                    help="items.jsonl of every development run on real texts: their source texts are left out")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--reader", default=rr.READER)
    ap.add_argument("--readers2", nargs="*", default=DEFAULT_READERS2, metavar="LABEL=ID")
    ap.add_argument("--reader-items-per-cell", nargs="*", default=["qwen2.5-14b=10"], metavar="LABEL=N",
                    help="a reader that scores only the answers to the first N items per cell ('' = all)")
    ap.add_argument("--judges", nargs="*", default=DEFAULT_JUDGES, metavar="LABEL=ID")
    ap.add_argument("--selective", nargs="*", default=DEFAULT_SELECTIVE, metavar="LABEL=ID")
    ap.add_argument("--variants", nargs="+", default=["default", "concise"])
    ap.add_argument("--seed", type=int, default=1, help="items: the development runs used seed 0")
    ap.add_argument("--n-per-cell-synthetic", type=int, default=30)
    ap.add_argument("--n-per-cell-real", type=int, default=30)
    ap.add_argument("--suites", nargs="+", default=DEFAULT_SUITES,
                    help="real-text suites (Poddar et al. is not held out: see docs/PROTOCOL.md)")
    ap.add_argument("--filler-check-n", type=int, default=5)
    ap.add_argument("--real-check-n", type=int, default=8, help="R3: real items per cell (development: 4)")
    ap.add_argument("--dtype", default="bf16", help="the evaluated models, the other readers, judge, Selective Context")
    ap.add_argument("--reader-dtype", default="fp16", help="the main reader (the development runs read in fp16)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--reader-max-batch-tokens", type=int, default=8192)
    ap.add_argument("--annotation-sources", nargs="+", default=["synthetic", "squad2", "musique", "cuad"])
    ap.add_argument("--annotation-items", type=int, default=8, help="items per source")
    ap.add_argument("--annotation-per-item", type=int, default=6, help="answers per item")
    ap.add_argument("--annotation-min-distinct", type=int, default=3, help="items need this many distinct answers")
    ap.add_argument("--abstain-n", type=int, default=150)
    ap.add_argument("--skip", nargs="*", default=[], choices=["synthetic", "real", "annotation"])
    ap.add_argument("--free-model-cache", action="store_true",
                    help="delete each evaluated model's download after use (needs ~70 GB instead of ~140 GB; models "
                         "are downloaded again for the second suite)")
    ap.add_argument("--allow-protocol-change", action="store_true")
    ap.add_argument("--print-hash", action="store_true",
                    help="print the package version and the code hash (fixed before the run) and stop")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"], help="mock: a smoke test without models")
    args = ap.parse_args(argv)
    if not args.print_hash and not args.dev_items:
        ap.error("--dev-items is required")

    if args.print_hash:
        import drybench

        print(json.dumps({"version": drybench.__version__, "code": code_hash(),
                          "files": [f.relative_to(REPO).as_posix() for f in code_files()]}, indent=1))
        return
    t0 = time.time()
    out = Path(args.run)
    missing = [p for p in args.dev_items if not Path(p).exists()]
    if missing:
        raise SystemExit("development items not found: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    protocol(out, args, write=False)  # a rerun with other settings stops here, before any download
    data = Path(args.data) if args.data else out.parent / "data"
    if args.backend == "hf":
        ids = [v.split("=", 1)[1] for v in args.models + args.readers2 + args.judges + args.selective] + [args.reader]
        preflight(ids)
        local = offline(ids)
        if local:
            print("Every model is a local folder: nothing is fetched from the Hugging Face hub.")
        if "real" not in args.skip:
            check_data(data, args.suites, args.dev_items, need_all=local)
    proto = protocol(out, args)
    rr.write_hardware(out)
    print(f"Confirmation run {out.resolve()} — package v{proto['version']} (code {proto['code']}, scoring "
          f"{proto['scoring']})")
    subset = [s for s in args.reader_items_per_cell if s]
    shared = ["--models", *args.models, "--reader", args.reader, "--readers2", *args.readers2,
              "--variants", *args.variants, "--dtype", args.dtype, "--reader-dtype", args.reader_dtype,
              "--batch-size", args.batch_size,
              "--max-new-tokens", args.max_new_tokens, "--reader-max-batch-tokens", args.reader_max_batch_tokens,
              "--missing-check", "--backend", args.backend]
    shared += (["--reader-items-per-cell", *subset] if subset else [])
    shared += (["--judges", *args.judges] if args.judges else []) + (["--selective", *args.selective] if args.selective else [])
    shared += (["--allow-cpu"] if args.allow_cpu else []) + (["--free-model-cache"] if args.free_model_cache else [])

    if "synthetic" not in args.skip:
        print("\n======== 1/3 synthetic reports ========")
        rr.run([sys.executable, REPO / "scripts" / "run_synthetic.py", "--run", out / "synthetic",
                "--n-per-cell", args.n_per_cell_synthetic, "--seed", args.seed, "--filler-check-n", args.filler_check_n,
                "--filler-check-seed", proto["filler_check_seed"]] + shared)
    if "real" not in args.skip:
        print("\n======== 2/3 real texts ========")
        # the synthetic planted-filler check is T1 of the synthetic suite: not repeated here (R2)
        rr.run([sys.executable, REPO / "scripts" / "run_real.py", "--run", out / "real",
                "--data", data, "--suites", *args.suites,
                "--n-per-cell", args.n_per_cell_real, "--seed", args.seed, "--real-check-n", args.real_check_n,
                "--filler-check-n", 0, "--exclude-items", *args.dev_items, "--require-musique-full",
                "--no-human-check"] + shared)
    if "annotation" not in args.skip:
        print("\n======== 3/3 samples for human labels ========")
        annotation(out, args)
    if (out / "synthetic").exists() or (out / "real").exists():
        print("\n======== confirmatory tests (the human-label tests wait for the labels) ========")
        rr.run(rr.CLI + ["confirm", "--run", out])
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    ann = out / "annotation"
    if ann.exists():
        shutil.make_archive(str(out) + "_annotation", "zip", root_dir=out, base_dir="annotation")
    tables = [p for p in (out / "synthetic" / "tables_synthetic.md", out / "real" / "tables_real.md") if p.exists()]
    print("\nTables: " + ", ".join(map(str, tables)))
    print(f"For the annotators: {ann} (also {out}_annotation.zip)\nWhole run: {archive}")
    print(f"Total time of this session: {(time.time() - t0) / 60:.1f} min")


def annotation(out: Path, args: argparse.Namespace) -> None:
    """Both samples. A reader that scored part of the run scores the sampled answers it missed (kept apart in
    scored_sample.jsonl, so its tables stay on its own part); then the judges and Selective Context on the
    sample. A sample that exists is never replaced by a different one: labels and baseline rows name its
    responses by uid."""
    import tempfile

    ann = out / "annotation"
    runs = [p for p in (out / "synthetic", out / "real") if (p / "scored.jsonl").exists()]
    if not runs:
        print("    nothing scored yet")
        return
    dtype = args.dtype
    try:
        import torch

        dtype = args.dtype if torch.cuda.is_available() else "auto"
    except ImportError:
        dtype = "auto"

    def build() -> None:
        tmp = Path(tempfile.mkdtemp(prefix="sample_", dir=out))
        rr.run(rr.CLI + ["annotation-sample", "--runs", *runs, "--out", tmp, "--sources", *args.annotation_sources,
                         "--items-per-source", args.annotation_items, "--per-item", args.annotation_per_item,
                         "--min-distinct", args.annotation_min_distinct])
        ann.mkdir(exist_ok=True)
        old = ann / "sample.jsonl"
        if old.exists() and old.read_text(encoding="utf-8") != (tmp / "sample.jsonl").read_text(encoding="utf-8"):
            print(f"    WARNING: the sample built now differs from {old} (the runs changed since): the existing "
                  "sample and its key are kept. Delete the annotation folder to draw a new one before anyone labels it.")
        else:
            for f in ("sample.jsonl", "key.jsonl", "sample_stats.json"):
                (tmp / f).replace(ann / f)
        shutil.rmtree(tmp, ignore_errors=True)

    build()
    key = [json.loads(x) for x in (ann / "key.jsonl").read_text().splitlines() if x.strip()]
    rescored = False
    for label, model_id in rr.label_map(args.readers2).items():
        for run in runs:
            todo = {(k["model"], k["variant"], k["item_id"]) for k in key
                    if k["run"] == run.name and k.get(f"status_{label}") is None}  # not scored by this reader
            sub = run / f"reader2_{label}"
            if not todo or not sub.exists():
                continue
            rows = [r for r in map(json.loads, (run / "responses.jsonl").read_text().splitlines())
                    if (r["model"], r.get("variant", "default"), r["item_id"]) in todo]
            part = sub / "responses_sample.jsonl"
            part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"    reader {label} scores {len(rows)} sampled answers of {run.name} it had not scored")
            rr.run(rr.CLI + ["score", "--items", run / "items.jsonl", "--responses", part,
                             "--out", sub / "scored_sample.jsonl", "--reader-backend", args.backend, "--reader", model_id,
                             "--reader-dtype", dtype, "--reader-max-batch-tokens", args.reader_max_batch_tokens,
                             "--mode", "loo", "--ablation-abs-tol", 2.0])
            rescored = True
    if rescored:  # the same seed draws the same sample, now with every reader's map in the key
        build()
    rr.run_baselines(ann, ["sample"], rr.label_map(args.judges), rr.label_map(args.selective), args.backend,
                     dtype, args.reader_max_batch_tokens)
    if not (ann / "abstain_sample.csv").exists():
        rr.run(rr.CLI + ["abstain-sample", "--runs", *runs, "--out", ann, "--n", args.abstain_n])
    page = REPO / "annotation" / "dry_span_labels_offline.html"
    if page.exists():
        shutil.copyfile(page, ann / "labelling_page.html")
    guide = REPO / "docs" / "ANNOTATION.md"
    if guide.exists():
        shutil.copyfile(guide, ann / "ANNOTATION.md")


if __name__ == "__main__":
    main()
