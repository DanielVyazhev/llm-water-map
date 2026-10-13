# Filler in LLM answers on real texts — SQuAD 2.0, MuSiQue, CUAD contracts, Poddar et al. — Colab T4
# 1) Runtime → Change runtime type → T4 GPU.   2) Run this cell. With the defaults (15 items per cell,
#    ~300 items, 3 models × 2 prompts, 2 readers and all checks) it takes about 3 hours on a T4;
#    N_PER_CELL = 10 takes about 2 hours.
# Rerunning the cell resumes where it stopped. With USE_DRIVE = True the run lives in Google Drive,
# so a disconnected session loses nothing: reconnect, rerun this cell.
# The package (v__VERSION__) is embedded at the bottom of the cell; nothing to upload.

# ---- settings ----
N_PER_CELL = 15                      # items per cell after the leakage filter (21 cells)
SUITES = ["squad2", "musique", "cuad", "poddar"]
MODELS = {                           # label: Hugging Face id of an evaluated (instruct) model
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",  # gated: needs HF_TOKEN in Colab secrets
}
READER = "Qwen/Qwen2.5-3B-Instruct"         # reads the responses; keep it out of MODELS
READER2 = "microsoft/Phi-3.5-mini-instruct"  # second reader for agreement; "none" saves ~45 min
VARIANTS = ["default", "concise"]    # "concise" adds "Be concise." to the system prompt
FILLER_CHECK_N = 3                    # synthetic reports per cell: filler map with known filler
REAL_CHECK_N = 4                     # real items per cell: planted filler in real-text responses
HUMAN_CHECK = True                   # deletions vs Poddar et al.'s human span labels
MAX_NEW_TOKENS = 320
DTYPE = "fp16"                       # T4 has no bf16
BATCH = 16                           # generation batch (halved automatically on out-of-memory)
USE_DRIVE = True                     # keep the run in Google Drive (survives disconnects)
RUN_ROOT = "/content/drybench_runs"  # used when USE_DRIVE = False
RUN_NAME = "real_t4_n15_v12"         # new name = fresh run; same name = resume
NO_THINKING = False                  # True for reasoning models with a thinking switch (Qwen3)
EXTRA_ARGS = []                      # more options for scripts/run_real.py, e.g. ["--oversample", "6"]
# ---- end settings ----

import base64
import hashlib
import io
import os
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
    except Exception:
        pass  # public models do not need a token


def main():
    t0 = time.time()
    if USE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive/drybench_runs")
    else:
        root = Path(RUN_ROOT)
    out = root / RUN_NAME
    root.mkdir(parents=True, exist_ok=True)
    pkg = install()
    hf_token()
    cmd = [sys.executable, pkg / "scripts" / "run_real.py", "--run", out, "--data", root / "data",
           "--suites", *SUITES, "--n-per-cell", N_PER_CELL,
           "--models", *[f"{k}={v}" for k, v in MODELS.items()], "--reader", READER, "--reader2", READER2,
           "--variants", *VARIANTS, "--dtype", DTYPE, "--batch-size", BATCH, "--max-new-tokens", MAX_NEW_TOKENS,
           "--filler-check-n", FILLER_CHECK_N, "--real-check-n", REAL_CHECK_N, "--reader-max-batch-tokens", 4096]
    if not HUMAN_CHECK:
        cmd.append("--no-human-check")
    if NO_THINKING:
        cmd.append("--no-thinking")
    run(cmd + list(EXTRA_ARGS))
    try:
        from IPython.display import Markdown, display

        display(Markdown((out / "tables_real.md").read_text()))
    except ImportError:
        pass
    print(f"\nRun folder: {out}\nArchive: {out}.zip (download it from the Files panel or run the next cell)")
    print(f"Total time of this session: {(time.time() - t0) / 60:.1f} min")


PKG_B64 = "__PKG_B64__"

main()
