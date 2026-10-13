# Filler in LLM answers — the confirmation run for the paper (Colab A100, package v__VERSION__)
# 1) Runtime → Change runtime type → A100 GPU (High-RAM).   2) Colab secrets: HF_TOKEN with access to
#    meta-llama/Llama-3.1-8B-Instruct and google/gemma-2-9b-it (or drop them from MODELS).   3) Run this cell.
# Both suites (synthetic reports, real texts), 7 models × 2 prompts, three readers, every check where the truth is
# known, the LLM judge and Selective Context, and the two samples for human labels. With the defaults about 8–12 h
# on an A100. The run lives in Google Drive: after a disconnect, rerun the cell; it resumes. The code is frozen: the
# run records the package version and a hash of its code in protocol.json and refuses other code or settings.
# The package is embedded at the bottom of the cell; nothing to upload.

# ---- settings ----
RUN_NAME = "final_v1"                # new name = fresh run; same name = resume
# items of the v1.2 development run on real texts: their source texts are left out (the v1.0 items come with
# the package: results/dev_v1.0/real/items.jsonl)
DEV_ITEMS = ["/content/drive/MyDrive/drybench_runs/real_t4_n15_v12/items.jsonl"]
MODELS = {                           # label: Hugging Face id of an evaluated (instruct) model
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",       # the development models
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",           # one family across sizes
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",  # gated
    "gemma2-9b": "google/gemma-2-9b-it",                # gated
}
READER = "Qwen/Qwen2.5-3B-Instruct"  # the main reader of the development runs; keep it out of MODELS
READERS2 = {"phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",  # the development runs' second reader
            "phi-4": "microsoft/phi-4"}                         # a 14B reader of the same family
READER_ITEMS_PER_CELL = {"phi-4": 10}  # phi-4 scores the answers to 10 items per cell (+ the annotation sample)
JUDGES = {"phi-4": "microsoft/phi-4"}  # LLM judge baseline: the same model as the big reader
SELECTIVE = {"qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"}  # Selective Context baseline: the main reader's model
N_PER_CELL_SYNTHETIC = 30            # 15 cells → 450 items
N_PER_CELL_REAL = 30                 # 21 cells → up to 630 items (Poddar et al.'s cells come out smaller)
SEED = 1                             # the development runs used 0
ANNOTATION_ITEMS = 6                 # items per source (synthetic, SQuAD, MuSiQue, CUAD) …
ANNOTATION_PER_ITEM = 8              # … × answers per item ≈ 190 answers to label
ABSTAIN_N = 150                      # answers to missing-fact items for held-out labels of the abstention rule
FREE_MODEL_CACHE = True              # delete each evaluated model's download after use: ~75 GB of disk instead of
                                     # ~140 GB, at the cost of downloading the models again for the second suite
DTYPE = "bf16"
BATCH = 32                           # generation batch (halved automatically on out-of-memory)
MAX_NEW_TOKENS = 400
USE_DRIVE = True                     # keep the run in Google Drive (survives disconnects)
RUN_ROOT = "/content/drybench_runs"  # used when USE_DRIVE = False
# ---- end settings ----

import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ENV = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PIP_BREAK_SYSTEM_PACKAGES="1")


def run(args):
    args = [str(a) for a in args]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"failed with exit code {p.returncode} — rerun the cell to resume")


def install():
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    pkg = (base / f"drybench_pkg_{digest}").resolve()
    if not (pkg / ".installed").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
        run([sys.executable, "-m", "pip", "install", "-q", str(pkg), "transformers>=4.46", "accelerate",
             "matplotlib", "gdown"])
        (pkg / ".installed").touch()
    return pkg


def hf_token():
    try:
        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        if tok:
            ENV["HF_TOKEN"] = tok
            return True
    except Exception:
        pass
    return False


def main():
    t0 = time.time()
    if USE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive/drybench_runs")
    else:
        root = Path(RUN_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    out = root / RUN_NAME
    missing = [p for p in DEV_ITEMS if not Path(p).exists()]
    if missing:
        raise SystemExit("Development items not found (restore the development run to Drive or fix DEV_ITEMS): "
                         + ", ".join(missing))
    free = shutil.disk_usage("/content" if Path("/content").exists() else ".").free / 1e9
    need = 75 if FREE_MODEL_CACHE else 140
    advice = "drop a large model" if FREE_MODEL_CACHE else "set FREE_MODEL_CACHE = True or drop a large model"
    print(f"Free disk: {free:.0f} GB" + (f"  ← the models need ~{need} GB at a time: {advice}"
                                          if free < need + 10 else ""))
    gated = [m for m in MODELS.values() if m.startswith(("meta-llama/", "google/"))]
    if not hf_token() and gated:
        raise SystemExit("No HF_TOKEN in Colab secrets, but MODELS has gated models: " + ", ".join(gated))
    pkg = install()

    def pairs(d):
        return [f"{k}={v}" for k, v in d.items()]

    dev_items = [pkg / "results" / "dev_v1.0" / "real" / "items.jsonl", *DEV_ITEMS]
    cmd = [sys.executable, pkg / "scripts" / "run_final.py", "--run", out, "--data", root / "data",
           "--dev-items", *dev_items, "--models", *pairs(MODELS), "--reader", READER,
           "--readers2", *pairs(READERS2), "--reader-items-per-cell", *pairs(READER_ITEMS_PER_CELL),
           "--judges", *pairs(JUDGES), "--selective", *pairs(SELECTIVE), "--seed", SEED,
           "--n-per-cell-synthetic", N_PER_CELL_SYNTHETIC, "--n-per-cell-real", N_PER_CELL_REAL,
           "--annotation-items", ANNOTATION_ITEMS, "--annotation-per-item", ANNOTATION_PER_ITEM,
           "--abstain-n", ABSTAIN_N, "--dtype", DTYPE, "--batch-size", BATCH, "--max-new-tokens", MAX_NEW_TOKENS]
    run(cmd + (["--free-model-cache"] if FREE_MODEL_CACHE else []))
    try:
        from IPython.display import Markdown, display

        for t in (out / "synthetic" / "tables_synthetic.md", out / "real" / "tables_real.md"):
            if t.exists():
                display(Markdown(t.read_text()))
    except ImportError:
        pass
    print(f"\nRun folder: {out}\nFor the annotators: {out}/annotation ({out}_annotation.zip)\nWhole run: {out}.zip\n"
          f"Confirmatory tests without the human labels: {out}/confirmatory.md")
    print(f"Total time of this session: {(time.time() - t0) / 60:.1f} min")


PKG_B64 = "__PKG_B64__"

main()
