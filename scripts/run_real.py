#!/usr/bin/env python3
"""The benchmark on real texts: SQuAD 2.0, MuSiQue, CUAD contracts and Poddar et al.'s passage
questions, for a GPU server or a Colab T4.

Once, in the repository root:
    python -m pip install -e ".[hf]" accelerate gdown huggingface_hub matplotlib

Run (re-running the same command resumes; a new --run name starts a fresh run):
    python scripts/run_real.py --run runs/real_v1          # defaults: 4 models up to 7B, A100-class GPU
    python scripts/run_real.py --run runs/real_t4 --dtype fp16 --n-per-cell 15 --models LABEL=HF_ID ...
On a Colab T4 use colab/real_texts_colab.ipynb (3 small models, 2 readers, ~3 h).

Steps: download the datasets → a pool of items per suite → leakage filter with the reader
(drops facts the reader answers without a response, and with --exclude-items every item built from a
source text those items use) → checks of each reader where the truth is known: reader check, filler map on
simulated synthetic reports, filler map on real-text responses with planted filler, and agreement with
human span labels (Poddar et al.) → responses and entropy signals of every model → scores with the
reader and the other readers (--readers2 LABEL=ID ...; --reader-items-per-cell LABEL=N scores the
answers to the first N items per cell only) → how each reader reads a missing fact (--missing-check) →
the LLM judge and Selective Context on the checks' responses (--judges / --selective LABEL=ID) →
analysis → runs/<name>/tables_real.md and runs/<name>.zip.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "qwen2.5-0.5b=Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b=Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b=HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-7b=Qwen/Qwen2.5-7B-Instruct",
]
READER = "Qwen/Qwen2.5-3B-Instruct"
READER2 = "microsoft/Phi-3.5-mini-instruct"
SQUAD_URLS = [
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json",
]
MUSIQUE_GDRIVE = "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"  # official musique_v1.0.zip
MUSIQUE_HF_MIRROR = "dgslibisey/MuSiQue"  # MuSiQue-Ans only
CUAD_URLS = ["https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip",  # CUADv1.json inside
             "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"]           # CUAD_v1/CUAD_v1.json
PODDAR_URL = "https://raw.githubusercontent.com/sohampoddar26/LLM-brevity/main/data.jsonl"
SUITES = ["squad2", "musique", "cuad", "poddar"]
# settings that define the items; changing them needs a new run
ITEM_KEYS = ("suites", "n_per_cell", "oversample", "seed", "distractors", "reader", "max_prior", "min_gold_p")

ENV = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
ENV["PYTHONPATH"] = str(REPO) + (os.pathsep + ENV["PYTHONPATH"] if ENV.get("PYTHONPATH") else "")
PROGRESS = re.compile(r"^(?:generated|signals|scored) (\d+)/(\d+)")
CLI = [sys.executable, "-m", "drybench.cli"]


def run(args, check: bool = True) -> int:
    args = [str(a) for a in args]
    shown = ["drybench"] + args[3:] if args[1:3] == ["-m", "drybench.cli"] else args
    print("$", " ".join(shown), flush=True)
    t0 = time.time()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:
            continue
        m = PROGRESS.match(line.strip())  # about ten progress lines per step
        if m and int(m.group(1)) % max(10, int(m.group(2)) // 10) and m.group(1) != m.group(2):
            continue
        print("   ", line, end="", flush=True)
    code = p.wait()
    print(f"    [{time.time() - t0:.0f}s]", flush=True)
    if check and code != 0:
        raise SystemExit(f"step failed with exit code {code}: {' '.join(shown)}")
    return code


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
        return True
    except Exception as e:  # noqa: BLE001 - any network error means: try the next source
        print(f"    could not download {url}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def get_squad(data: Path) -> list[Path]:
    dest = data / "squad2" / "dev-v2.0.json"
    if not dest.exists() and not any(fetch(u, dest) for u in SQUAD_URLS):
        raise SystemExit(f"Download SQuAD 2.0 dev ({SQUAD_URLS[0]}) and save it as {dest}")
    return [dest]


def get_musique(data: Path) -> list[Path]:
    d = data / "musique"
    d.mkdir(parents=True, exist_ok=True)
    full, ans = d / "musique_full_v1.0_dev.jsonl", d / "musique_ans_v1.0_dev.jsonl"
    if not full.exists():
        zpath = d / "musique_v1.0.zip"
        try:
            try:
                import gdown
            except ImportError:
                run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
                importlib.invalidate_caches()
                import gdown
            gdown.download(MUSIQUE_GDRIVE, str(zpath), quiet=True)
            with zipfile.ZipFile(zpath) as z:
                for name in z.namelist():
                    if Path(name).name in (full.name, ans.name):
                        (d / Path(name).name).write_bytes(z.read(name))
        except Exception as e:  # noqa: BLE001
            print(f"    MuSiQue from Google Drive failed: {e}")
        finally:
            zpath.unlink(missing_ok=True)
    if full.exists():
        return [full]
    if not ans.exists():
        try:
            from huggingface_hub import hf_hub_download

            got = Path(hf_hub_download(repo_id=MUSIQUE_HF_MIRROR, filename=ans.name, repo_type="dataset",
                                       local_dir=str(d)))
            if got != ans:
                shutil.copyfile(got, ans)
        except Exception as e:  # noqa: BLE001
            print(f"    MuSiQue from the Hugging Face mirror failed: {e}")
    if ans.exists():
        print("    MuSiQue-Full not found: the MuSiQue missing-fact cell is skipped "
              f"(put {full.name} into {d} to include it)")
        return [ans]
    raise SystemExit(f"Download musique_v1.0.zip ({MUSIQUE_GDRIVE}), unzip it and put {full.name} into {d}")


def get_cuad(data: Path) -> list[Path]:
    d = data / "cuad"
    dest = d / "CUADv1.json"
    if not dest.exists():
        d.mkdir(parents=True, exist_ok=True)
        zpath = d / "cuad.zip"
        for url in CUAD_URLS:
            if not fetch(url, zpath):
                continue
            try:
                with zipfile.ZipFile(zpath) as z:
                    name = next((n for n in z.namelist() if Path(n).name in ("CUADv1.json", "CUAD_v1.json")), None)
                    if name:
                        dest.write_bytes(z.read(name))
            except zipfile.BadZipFile as e:
                print(f"    {url}: {e}")
            zpath.unlink(missing_ok=True)
            if dest.exists():
                break
    if not dest.exists():
        raise SystemExit(f"Download CUAD ({CUAD_URLS[0]}), unzip it and put CUADv1.json into {d}")
    return [dest]


def get_poddar(data: Path) -> list[Path]:
    dest = data / "poddar" / "data.jsonl"
    if not dest.exists() and not fetch(PODDAR_URL, dest):
        raise SystemExit(f"Download Poddar et al.'s data.jsonl ({PODDAR_URL}) and save it as {dest}")
    return [dest]


GETTERS = {"squad2": get_squad, "musique": get_musique, "cuad": get_cuad, "poddar": get_poddar}


def model_map(specs: list[str]) -> dict[str, str]:
    out = {}
    for s in specs:
        label, sep, mid = s.partition("=")
        if not sep or not label or not mid:
            raise SystemExit(f"--models entries look like label=hf_id, got {s!r}")
        out[label] = mid
    return out


def reader_label(model_id: str) -> str:
    return re.sub(r"-instruct$", "", model_id.rstrip("/").split("/")[-1].lower())


def check_versions(out: Path, reader: str, reader2: str | dict | None) -> None:
    """Scoring or signals changed since this run was made: recompute them (every score, or only those
    the new version changed for that reader: drybench.rescore), keep the responses. reader2: the other
    readers as {label: model id} (or one model id, labelled by reader_label)."""
    extra = reader2 if isinstance(reader2, dict) else ({reader_label(reader2): reader2} if reader2 else {})
    got = subprocess.run([sys.executable, "-c", "import drybench, json; print(json.dumps("
                          "{'scoring': drybench.SCORING_VERSION, 'signals': drybench.SIGNALS_VERSION}))"],
                         capture_output=True, text=True, env=ENV)
    if got.returncode:
        raise SystemExit("drybench is not importable: run `python -m pip install -e \".[hf]\"` in " + str(REPO))
    new = json.loads(got.stdout.strip())
    vfile = out / "versions.json"
    old = json.loads(vfile.read_text()) if vfile.exists() else new
    if old.get("scoring") != new["scoring"]:
        print(f"Scoring changed (v{old.get('scoring')} → v{new['scoring']}): rescoring, responses are kept.")
        plans = {}
        for sub in [out] + sorted(out.glob("reader2_*")):
            rd = reader if sub == out else extra.get(sub.name[len("reader2_"):])
            got = subprocess.run(CLI + ["prune-scores", "--dir", str(sub), "--items", str(out / "items.jsonl"),
                                        "--from", str(old.get("scoring"))] + (["--reader", rd] if rd else [])
                                 + ["--json"], capture_output=True, text=True, env=ENV)
            if got.returncode:
                raise SystemExit("prune-scores failed:\n" + got.stdout + got.stderr)
            res = json.loads(got.stdout.strip().splitlines()[-1])
            plans[sub] = res["plan"]
            what = {"all": "every score", "missing": "scores of items with a missing fact",
                    "none": "nothing (this reader's scores did not change)"}[res["plan"]]
            print(f"    {sub.name}: rescoring {what}")
            for line in res["changes"]:
                print(f"      {line}")
        report = out / "filter_report.json"
        if plans.get(out) != "all" and report.exists():  # the leakage filter reads probabilities, not
            rep = json.loads(report.read_text())          # gains: a partial change leaves it as it is
            if rep.get("scoring") == old.get("scoring"):
                report.write_text(json.dumps({**rep, "scoring": new["scoring"]}, indent=1))
    if old.get("signals") != new["signals"]:
        print(f"Signals changed (v{old.get('signals')} → v{new['signals']}): recomputing entropy signals.")
        (out / "signals.jsonl").unlink(missing_ok=True)
    vfile.write_text(json.dumps(new))


def write_hardware(out: Path) -> None:
    """GPU and library versions of every session that worked on the run (hardware.json: sessions)."""
    import platform

    info = {"python": platform.python_version()}
    try:
        got = subprocess.run([sys.executable, "-c", "import drybench; print(drybench.__version__)"],
                             capture_output=True, text=True, env=ENV)
        info["drybench"] = got.stdout.strip() or None
    except OSError:
        pass
    try:
        import torch
        import transformers

        info.update({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "cuda": torch.version.cuda, "torch": torch.__version__,
                     "transformers": transformers.__version__})
    except ImportError:
        info["gpu"] = "cpu"
    path = out / "hardware.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    sessions = old.get("sessions") or ([old] if old else [])
    if not sessions or {k: v for k, v in sessions[-1].items() if k != "started"} != info:
        sessions.append({**info, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    path.write_text(json.dumps({"sessions": sessions}, indent=1))


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def subset_responses(items: Path, resp: Path, dest: Path, n_per_cell: int, cell_of) -> Path:
    """The responses to the first n items of every cell (items.jsonl order), for a reader that scores
    part of the run."""
    from drybench.types import load_items, read_jsonl

    seen: dict = {}
    keep = set()
    for it in load_items(items):
        c = cell_of(it)
        if seen.get(c, 0) < n_per_cell:
            keep.add(it.item_id)
            seen[c] = seen.get(c, 0) + 1
    rows = [r for r in read_jsonl(resp) if r["item_id"] in keep]
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return dest


def run_baselines(out: Path, sets: list[str], judges: dict, selective: dict, backend: str, dtype: str,
                  batch_tokens: int, extra: list | None = None) -> None:
    """The LLM judge(s) and Selective Context on the responses of the checks (drybench baselines)."""
    for method, models in (("judge", judges), ("selective", selective)):
        for label, model_id in models.items():
            run(CLI + ["baselines", "--run", out, "--set", *sets, "--method", method, "--label", label,
                       "--backend", backend, "--model", model_id, "--dtype", dtype,
                       "--max-batch-tokens", batch_tokens] + list(extra or []))


def missing_checks(out: Path, readers: dict, backend: str, dtype: str, extra: list | None = None) -> None:
    """How each reader reads a fact the source does not give (drybench missing-check on the run's answers)."""
    for label, model_id in readers.items():
        dest = out / f"missing_check_{label}.json"
        if not dest.exists():
            run(CLI + ["missing-check", "--run", out, "--reader-backend", backend, "--reader", model_id,
                       "--reader-dtype", dtype, "--out", dest] + list(extra or []))


def free_model_cache(model_id: str, keep: set) -> None:
    """Delete a model's Hugging Face download (not one of `keep`: readers and judges are used again)."""
    if model_id in keep or "/" not in model_id:
        return
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        revs = [r.commit_hash for repo in cache.repos if repo.repo_id == model_id for r in repo.revisions]
        if revs:
            strategy = cache.delete_revisions(*revs)
            print(f"    freed {strategy.expected_freed_size_str} ({model_id})")
            strategy.execute()
    except Exception as e:  # noqa: BLE001 - disk housekeeping must not stop the run
        print(f"    could not free the download of {model_id}: {e}")


def label_map(specs: list[str] | None) -> dict[str, str]:
    return model_map(specs) if specs else {}


def filler_check(args: argparse.Namespace, reader: str, dest: Path, dtype: str) -> None:
    """The filler map on simulated responses with planted filler (truth known for every fragment)."""
    if args.filler_check_n <= 0 or (dest / "summary.json").exists():
        return
    backend = "mock" if args.backend == "mock" else "hf"
    cmd = [sys.executable, REPO / "scripts" / "pilot_validation.py", "--reader-backend", backend, "--out-dir", dest,
           "--n-per-cell", args.filler_check_n, "--greedy-conditions", "none", "--no-plots", "--quiet",
           "--seed", getattr(args, "filler_check_seed", 2026)]
    if backend == "hf":
        cmd += ["--reader", reader, "--reader-dtype", dtype]
    run(cmd)


def real_checks(args: argparse.Namespace, reader: str, sub: Path, items: Path, poddar: Path | None,
                dtype: str) -> None:
    """Checks on real texts where the truth is known: planted filler, and human span labels."""
    common = ["--reader-backend", args.backend, "--reader", reader, "--reader-dtype", dtype,
              "--reader-max-batch-tokens", args.reader_max_batch_tokens]
    if args.real_check_n > 0 and not (sub / "filler_check_real" / "summary.json").exists():
        run(CLI + ["real-filler-check", "--items", items, "--out-dir", sub / "filler_check_real",
                   "--n-per-cell", args.real_check_n, *common])
    if poddar is not None and args.human_check and not (sub / "human_check" / "summary.json").exists():
        run(CLI + ["poddar-check", "--data", poddar, "--out-dir", sub / "human_check", *common])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/real_v1", help="output folder; the same folder resumes")
    ap.add_argument("--data", default="data", help="where the datasets are downloaded")
    ap.add_argument("--suites", nargs="+", default=SUITES, choices=SUITES)
    ap.add_argument("--n-per-cell", type=int, default=30, help="items per cell after the filter")
    ap.add_argument("--oversample", type=int, default=4, help="pool size = n-per-cell × this")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="label=hf_id ...")
    ap.add_argument("--reader", default=READER)
    ap.add_argument("--reader2", default=READER2, help="second reader, or 'none' (label: from the model id)")
    ap.add_argument("--readers2", nargs="*", default=None, metavar="LABEL=ID",
                    help="other readers with their labels (instead of --reader2), e.g. phi3.5-mini=microsoft/"
                         "Phi-3.5-mini-instruct phi-4=microsoft/phi-4")
    ap.add_argument("--reader-items-per-cell", nargs="*", default=[], metavar="LABEL=N",
                    help="another reader scores only the answers to the first N items per cell")
    ap.add_argument("--exclude-items", nargs="+", default=None,
                    help="items of other runs (the development runs): items from the same source texts are left out")
    ap.add_argument("--judges", nargs="*", default=None, metavar="LABEL=ID", help="LLM judge baselines")
    ap.add_argument("--selective", nargs="*", default=None, metavar="LABEL=ID", help="Selective Context baselines")
    ap.add_argument("--missing-check", action="store_true", help="how each reader reads a missing fact")
    ap.add_argument("--variants", nargs="+", default=["default", "concise"], choices=["default", "concise"])
    ap.add_argument("--dtype", default="bf16", help="bf16 on A100/H100; fp16 on T4/V100")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--reader-max-batch-tokens", type=int, default=8192)
    ap.add_argument("--max-prior", type=float, default=0.05)
    ap.add_argument("--min-gold-p", type=float, default=0.5)
    ap.add_argument("--distractors", type=int, default=3, help="MuSiQue distractor paragraphs per item")
    ap.add_argument("--sampled", type=int, default=0, help=">0: also sampled answer entropy (N samples)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="pass enable_thinking=false to chat templates (Qwen3-style reasoning models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filler-check-n", type=int, default=5,
                    help="synthetic items per cell for the filler-map check of each reader (0 = skip)")
    ap.add_argument("--filler-check-seed", type=int, default=2026,
                    help="item seed of that check (the development runs used 2026)")
    ap.add_argument("--free-model-cache", action="store_true",
                    help="delete an evaluated model's download after its responses and signals (saves disk; "
                         "a later suite downloads it again)")
    ap.add_argument("--real-check-n", type=int, default=4,
                    help="real items per cell for the planted-filler check of each reader (0 = skip)")
    ap.add_argument("--no-human-check", dest="human_check", action="store_false",
                    help="skip the comparison with Poddar et al.'s human span labels")
    ap.add_argument("--require-musique-full", action="store_true",
                    help="stop when MuSiQue-Full cannot be downloaded (without it the MuSiQue missing-fact cell is "
                         "skipped); the confirmation run sets it")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"], help="mock: a smoke test without models")
    args = ap.parse_args()

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
    models = model_map(args.models)
    if args.readers2 is not None:
        extra = label_map(args.readers2)
    else:
        extra = {} if args.reader2.lower() == "none" else {reader_label(args.reader2): args.reader2}
    subset = {k: int(v) for k, v in label_map(args.reader_items_per_cell).items()}
    judges, selective = label_map(args.judges), label_map(args.selective)

    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    print("Outputs:", out.resolve())
    cfg_path = out / "config.json"
    cfg = {"suites": sorted(args.suites), "n_per_cell": args.n_per_cell, "oversample": args.oversample,
           "seed": args.seed, "distractors": args.distractors, "reader": args.reader, "max_prior": args.max_prior,
           "min_gold_p": args.min_gold_p, "readers2": extra, "reader_items_per_cell": subset, "models": models,
           "variants": args.variants, "judges": judges, "selective": selective,
           "exclude_items": [str(p) for p in args.exclude_items or []]}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = [k for k in ITEM_KEYS if old.get(k) != cfg[k]]
        if diff:
            raise SystemExit(f"{out} was made with other item settings ({', '.join(diff)}); "
                             "use a new --run name or the old settings.")
        moved = {k: (old["models"][k], v) for k, v in models.items() if old.get("models", {}).get(k) not in (None, v)}
        if moved:
            raise SystemExit(f"{out}: model labels already used for other models: "
                             + "; ".join(f"{k} was {a}, now {b}" for k, (a, b) in moved.items())
                             + " — use new labels or a new --run name.")
        cfg["models"] = {**old.get("models", {}), **models}
        if old.get("exclude_items", []) != cfg["exclude_items"] and old.get("exclude_items") is not None:
            raise SystemExit(f"{out} was made with other --exclude-items; use a new --run name or the old ones.")
        cfg.update({k: old[k] for k in ("data", "musique_full") if k in old})  # recorded by the first session
    cfg_path.write_text(json.dumps(cfg, indent=1))
    write_hardware(out)
    check_versions(out, args.reader, extra)

    items, resp, signals, scored = out / "items.jsonl", out / "responses.jsonl", out / "signals.jsonl", out / "scored.jsonl"
    template = ["--chat-template-kwargs", json.dumps({"enable_thinking": False})] if args.no_thinking else []

    print("\n[1/8] data")
    data = Path(args.data)
    paths = {s: GETTERS[s](data) for s in args.suites}
    poddar = None
    if args.human_check:
        poddar = paths["poddar"][0] if "poddar" in paths else get_poddar(data)[0]
    for s, ps in paths.items():
        print(f"    {s}: {', '.join(str(p) for p in ps)}")
    full = "musique" not in paths or any(p.name == "musique_full_v1.0_dev.jsonl" for p in paths["musique"])
    if not full and args.require_musique_full:
        raise SystemExit("MuSiQue-Full is needed (--require-musique-full): put musique_full_v1.0_dev.jsonl into "
                         f"{data / 'musique'} and run again")
    files = {str(p): sha256(p) for ps in paths.values() for p in ps}
    if poddar is not None:
        files[str(poddar)] = sha256(poddar)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("data") and cfg["data"] != files:
        print("    note: the dataset files differ from those this run started with (pools already built are kept)")
    cfg.update({"data": cfg.get("data") or files, "musique_full": full})
    cfg_path.write_text(json.dumps(cfg, indent=1))

    print("\n[2/8] items: pool per suite, then the leakage filter")
    report = out / "filter_report.json"
    scoring = json.loads((out / "versions.json").read_text())["scoring"]
    stale = items.exists() and (not report.exists() or json.loads(report.read_text()).get("scoring") != scoring)
    if not items.exists() or stale:
        pools = []
        for s in args.suites:
            pool = out / f"pool_{s}.jsonl"
            if not pool.exists():
                run(CLI + ["real-items", "--source", s, "--path", *paths[s], "--out", pool,
                           "--n-per-cell", args.n_per_cell * args.oversample, "--seed", args.seed,
                           "--distractors", args.distractors])
            pools.append(pool)
        if stale:  # the reader's questions changed: keep the items in use, top up the cells that came out short
            print("    scoring changed since the items were filtered: re-checking the rest of the pools")
        run(CLI + ["filter-items", "--items", *pools, "--out", items, "--report", report,
                   "--n-per-cell", args.n_per_cell, "--max-prior", args.max_prior, "--min-gold-p", args.min_gold_p,
                   "--reader-backend", args.backend, "--reader", args.reader, "--reader-dtype", dtype,
                   "--reader-max-batch-tokens", args.reader_max_batch_tokens] + (["--keep", items] if stale else [])
            + (["--exclude-items", *args.exclude_items] if args.exclude_items else []))
    print(f"    {sum(1 for _ in open(items))} items")

    print("\n[3/8] checks of the reader where the truth is known: reader check, filler map on synthetic and "
          "real-text responses with planted filler, human span labels")
    if not (out / "reader_check.json").exists():
        run(CLI + ["check-reader", "--items", items, "--reader-backend", args.backend, "--reader", args.reader,
                   "--reader-dtype", dtype, "--n", 40, "--out", out / "reader_check.json"])
    filler_check(args, args.reader, out / "filler_check", dtype)
    real_checks(args, args.reader, out, items, poddar, dtype)

    print("\n[4/8] responses + entropy signals")
    for label, model_id in models.items():
        run(CLI + ["generate", "--items", items, "--out", resp, "--backend", args.backend, "--model", model_id,
                   "--model-name", label, "--variant", *args.variants, "--dtype", dtype,
                   "--batch-size", args.batch_size, "--max-new-tokens", args.max_new_tokens, *template])
        run(CLI + ["signals", "--items", items, "--responses", resp, "--out", signals, "--backend", args.backend,
                   "--model", model_id, "--model-name", label, "--dtype", dtype, "--sampled", args.sampled,
                   *template])
        if args.free_model_cache:
            free_model_cache(model_id, {args.reader, *extra.values(), *judges.values(), *selective.values()})

    print("\n[5/8] scores")
    run(CLI + ["score", "--items", items, "--responses", resp, "--out", scored, "--reader-backend", args.backend,
               "--reader", args.reader, "--reader-dtype", dtype,
               "--reader-max-batch-tokens", args.reader_max_batch_tokens, "--mode", "loo", "--ablation-abs-tol", 2.0])

    subs = [out]
    if not extra:
        print("\n[6/8] other readers: none")
    for label, reader2 in extra.items():
        print(f"\n[6/8] reader {label}: {reader2}")
        r2 = out / f"reader2_{label}"
        r2.mkdir(exist_ok=True)
        if not (r2 / "reader_check.json").exists():
            run(CLI + ["check-reader", "--items", items, "--reader-backend", args.backend, "--reader", reader2,
                       "--reader-dtype", dtype, "--n", 40, "--out", r2 / "reader_check.json"])
        answers = resp
        if subset.get(label):
            from drybench.realdata import cell_key

            answers = subset_responses(items, resp, r2 / "responses_scored.jsonl", subset[label], cell_key)
            print(f"    scores the answers to the first {subset[label]} items per cell")
        run(CLI + ["score", "--items", items, "--responses", answers, "--out", r2 / "scored.jsonl",
                   "--reader-backend", args.backend, "--reader", reader2, "--reader-dtype", dtype,
                   "--reader-max-batch-tokens", args.reader_max_batch_tokens, "--mode", "loo",
                   "--ablation-abs-tol", 2.0])
        filler_check(args, reader2, r2 / "filler_check", dtype)
        real_checks(args, reader2, r2, items, poddar, dtype)
        subs.append(r2)
    if args.missing_check:
        print("\n[6/8] how each reader reads a missing fact")
        missing_checks(out, {"main": args.reader, **extra}, args.backend, dtype)
    if judges or selective:
        print("\n[6/8] baselines: LLM judge and Selective Context on the checks' responses")
        sets = ["planted_real"] + (["human_poddar"] if poddar is not None and args.human_check else [])
        run_baselines(out, sets, judges, selective, args.backend, dtype, args.reader_max_batch_tokens,
                      ["--poddar-data", poddar] if poddar is not None else [])

    print("\n[7/8] analysis")
    for sub in subs:
        run(CLI + ["analyze", "--items", items, "--scored", sub / "scored.jsonl", "--signals", signals,
                   "--out-dir", sub / "report", "--n-boot", 500])

    print("\n[8/8] tables")
    run([sys.executable, "-m", "drybench.report", out, "--quiet"])
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(f"\nTables: {out / 'tables_real.md'}\nWhole run: {archive}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
