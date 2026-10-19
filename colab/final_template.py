# Filler in LLM answers — the confirmation run for the paper (an A100 server or Colab A100, package v__VERSION__)
# Both suites (synthetic reports, real texts), 3 models × 2 prompts, three readers, every check where the truth is
# known, the LLM judge and Selective Context, and the two samples for human labels; about 6–10 h on one A100 40 GB.
# Rerunning the cell resumes. The code is frozen: the run records the package version and a hash of its code in
# protocol.json and refuses other code or settings (docs/PROTOCOL.md).
# A server: RUN_ROOT holds the datasets (data/squad2, data/musique, data/cuad); LOCAL maps every model to its folder,
# and then nothing is downloaded; PIP = False installs nothing (the package runs from its folder with the machine's
# numpy, pandas, scipy, torch, transformers).
# Colab: USE_DRIVE = True, PIP = True, LOCAL = {}, FREE_MODEL_CACHE = True, HF_TOKEN in the Colab secrets (Llama).
# The package is embedded as PKG_B64 (notebook: the hidden cell "Package"); nothing to upload.

# ---- settings ----
RUN_NAME = "final_v1"                # new name = fresh run; same name = resume
RUN_ROOT = "~/drybench_runs"         # the run and data/ (USE_DRIVE: drybench_runs in Google Drive)
USE_DRIVE = False                    # Colab: True keeps everything in Google Drive (survives disconnects)
# the source texts of the development items are left out; both items files come with the package
# (results/dev_v1.0/real, results/dev_v1.2/real_t4_n15_v12); more items files: paths in RUN_ROOT or absolute
DEV_ITEMS = []
MODELS = {                           # label: Hugging Face id of an evaluated (instruct) model
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}
READER = "Qwen/Qwen2.5-3B-Instruct"  # the main reader of the development runs; not an evaluated model
READERS2 = {"phi4-mini": "microsoft/Phi-4-mini-instruct",  # the second reader
            "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct"}    # the large reader
READER_ITEMS_PER_CELL = {"qwen2.5-14b": 10}  # it scores the answers to 10 items per cell (+ the annotation sample)
JUDGES = {"qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct"}   # LLM judge: the same model as the large reader
SELECTIVE = {"qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"}  # Selective Context: the main reader's model
LOCAL = {}                           # Hugging Face id → local model folder, e.g. {"Qwen/Qwen2.5-3B-Instruct":
                                     # "/models/Qwen2.5-3B-Instruct"}; every model local: no hub calls
N_PER_CELL_SYNTHETIC = 30            # 15 cells → 450 items
N_PER_CELL_REAL = 30                 # 18 cells → 540 items
REAL_CHECK_N = 8                     # planted filler in real texts (R3): items per cell
SEED = 1                             # the development runs used 0
ANNOTATION_ITEMS = 8                 # items per source (synthetic, SQuAD, MuSiQue, CUAD) …
ANNOTATION_PER_ITEM = 6              # … × answers per item ≈ 190 answers to label
ABSTAIN_N = 150                      # answers to missing-fact items for held-out labels of the abstention rule
DTYPE = "bf16"                       # evaluated models, the other readers, the judge, Selective Context
READER_DTYPE = "fp16"                # the main reader, as in the development runs
BATCH = 32                           # generation batch (halved automatically on out-of-memory)
MAX_NEW_TOKENS = 400
GPU = None                           # a server with several GPUs: the one to use, e.g. "0"
PIP = False                          # True (Colab): install what is missing; False: install nothing
FREE_MODEL_CACHE = False             # Colab: True deletes each downloaded evaluated model after use (~75 GB of
                                     # disk instead of ~100 GB; the models are downloaded again for the 2nd suite)
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
    if "PKG_B64" not in globals():
        raise SystemExit('Run the cell "Package" first (its code is hidden), then this cell.')
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    pkg = (base / f"drybench_pkg_{digest}").resolve()
    if not (pkg / ".installed").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
        try:
            if not PIP:
                raise RuntimeError("PIP = False")
            run([sys.executable, "-m", "pip", "install", "-q", "--retries", "1", "--timeout", "20", str(pkg),
                 "transformers>=4.46", "accelerate", "matplotlib", "gdown"])
        except RuntimeError as e:  # nothing installed: the libraries this machine has, the package from its folder
            have = subprocess.run([sys.executable, "-c", "import numpy, pandas, scipy, torch, transformers"],
                                  capture_output=True, text=True, env=ENV)
            if have.returncode:
                raise SystemExit(f"Not installed ({e}), and a library is missing:\n" + have.stderr[-800:])
            print("nothing installed: the package runs from", pkg)
        (pkg / ".installed").touch()
    ENV["PYTHONPATH"] = os.pathsep.join([str(pkg)] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    return pkg


def hf_token():
    try:
        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        if tok:
            ENV["HF_TOKEN"] = tok
    except Exception:
        pass  # a server: HF_TOKEN from the environment, if set


def model(mid):
    return os.path.expanduser(LOCAL.get(mid, mid))


def main():
    t0 = time.time()
    if USE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive/drybench_runs")
    else:
        root = Path(RUN_ROOT).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    out = root / RUN_NAME
    dev = [root / Path(p).expanduser() for p in DEV_ITEMS]  # an absolute path stays as it is
    missing = [str(p) for p in dev if not p.exists()]
    if missing:
        raise SystemExit("Development items not found (fix DEV_ITEMS): "
                         + ", ".join(missing))
    if GPU is not None:
        ENV["CUDA_VISIBLE_DEVICES"] = str(GPU)
    ids = [model(m) for m in [*MODELS.values(), READER, *READERS2.values(), *JUDGES.values(), *SELECTIVE.values()]]
    hub = sorted({m for m in ids if not Path(m).is_dir()})
    if hub:
        hf_token()
        gated = [m for m in hub if m.startswith(("meta-llama/", "google/"))]
        if gated and not ENV.get("HF_TOKEN"):
            raise SystemExit("No HF_TOKEN (Colab secrets or the environment), but these models are gated: "
                             + ", ".join(gated) + ". Or give their local folders in LOCAL.")
        free = shutil.disk_usage(root).free / 1e9
        need = 75 if FREE_MODEL_CACHE else 100
        print(f"Downloaded from the Hugging Face hub: {', '.join(hub)}. Free disk: {free:.0f} GB"
              + (f"  ← the models need ~{need} GB" if free < need + 10 else ""))
    else:
        print("Every model is a local folder: nothing is downloaded.")
    pkg = install()

    def pairs(d):
        return [f"{k}={model(v)}" for k, v in d.items()]

    packaged = [pkg / "results" / f / "items.jsonl" for f in ("dev_v1.0/real", "dev_v1.2/real_t4_n15_v12")]
    cmd = [sys.executable, pkg / "scripts" / "run_final.py", "--run", out, "--data", root / "data",
           "--dev-items", *packaged, *dev,
           "--models", *pairs(MODELS), "--reader", model(READER), "--readers2", *pairs(READERS2),
           "--reader-items-per-cell", *[f"{k}={v}" for k, v in READER_ITEMS_PER_CELL.items()],
           "--judges", *pairs(JUDGES), "--selective", *pairs(SELECTIVE), "--seed", SEED,
           "--n-per-cell-synthetic", N_PER_CELL_SYNTHETIC, "--n-per-cell-real", N_PER_CELL_REAL,
           "--real-check-n", REAL_CHECK_N, "--annotation-items", ANNOTATION_ITEMS,
           "--annotation-per-item", ANNOTATION_PER_ITEM, "--abstain-n", ABSTAIN_N, "--dtype", DTYPE,
           "--reader-dtype", READER_DTYPE, "--batch-size", BATCH, "--max-new-tokens", MAX_NEW_TOKENS]
    run(cmd + (["--free-model-cache"] if FREE_MODEL_CACHE else []))
    try:
        from IPython.display import Markdown, display

        for t in (out / "synthetic" / "tables_synthetic.md", out / "real" / "tables_real.md"):
            if t.exists():
                display(Markdown(t.read_text().replace("$", r"\$")))  # "$10.00" is not LaTeX
    except ImportError:
        pass
    print(f"\nRun folder: {out}\nFor the annotators: {out}/annotation ({out}_annotation.zip)\nWhole run: {out}.zip\n"
          f"Confirmatory tests without the human labels: {out}/confirmatory.md")
    print(f"Total time of this session: {(time.time() - t0) / 60:.1f} min")


PKG_B64 = "__PKG_B64__"

main()
