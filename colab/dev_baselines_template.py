# Filler in LLM answers — the LLM judge and Selective Context on the development runs (Colab A100, package v__VERSION__)
# 1) Runtime → Change runtime type → A100 GPU.   2) The two development runs of this package version in Google
#    Drive (drybench_runs/<SYN_RUN>, drybench_runs/<REAL_RUN>), made with the T4 notebooks.   3) Run this cell.
# Writes RUN/baselines/<method>-<label>/<set>.jsonl into each run (planted filler in synthetic reports; planted
# filler in real texts and Poddar et al.'s human labels), remakes the runs' tables, writes the development values
# of the confirmatory tests (docs/PROTOCOL.md) and zips everything into drybench_runs/<ZIP_NAME>.zip. About 1–2 h;
# rerunning resumes. The package is embedded as PKG_B64 (notebook: the hidden cell "Package"); nothing to upload.
# On a server instead of Colab: USE_DRIVE = False, RUN_ROOT = the folder with both runs, and JUDGES / SELECTIVE may
# name local model folders (then nothing is fetched from the Hugging Face hub); without internet the package runs
# from its folder with the libraries the server has (numpy, pandas, scipy, torch, transformers).

# ---- settings ----
SYN_RUN = "t4_n20_v12"               # the synthetic development run (T4 notebook drybench_colab.ipynb)
REAL_RUN = "real_t4_n15_v12"         # the real-text development run (T4 notebook real_texts_colab.ipynb)
JUDGES = {"phi-4": "microsoft/phi-4"}                   # LLM judge: the confirmation run's 14B reader (a hub id
                                                        # or a local model folder)
SELECTIVE = {"qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct"}  # Selective Context: the main reader's model
PODDAR_DATA = None                   # Poddar et al.'s data.jsonl: only for human-check rows written before v1.0
DTYPE = "bf16"
MAX_BATCH_TOKENS = 8192
BACKEND = "hf"                       # "mock": a dry run without models
ZIP_NAME = "dev_v12_results"         # both runs with the baselines, tables and the confirmatory statistics
USE_DRIVE = True                     # False on a server: the runs are in RUN_ROOT
RUN_ROOT = "/content/drybench_runs"  # the folder with SYN_RUN and REAL_RUN when USE_DRIVE = False
GPU = None                           # a server with several GPUs: the one to use, e.g. "1"
PIP = True                           # False: install nothing; the package runs from its folder with the
                                     # libraries this machine has
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
                 "transformers>=4.46", "accelerate", "matplotlib"])
        except RuntimeError as e:  # nothing installed: the libraries this machine has, the package from its folder
            libs = "numpy, pandas, scipy" + (", torch, transformers" if BACKEND == "hf" else "")
            have = subprocess.run([sys.executable, "-c", f"import {libs}"], capture_output=True, text=True,
                                  env=ENV)
            if have.returncode:
                raise SystemExit(f"Not installed ({e}), and a library is missing:\n" + have.stderr[-800:])
            print("nothing installed: the package runs from", pkg)
        (pkg / ".installed").touch()
    ENV["PYTHONPATH"] = os.pathsep.join([str(pkg)] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    return pkg


def local_models():
    """Every model a local folder: no calls to the Hugging Face hub (a server without internet)."""
    paths = [os.path.expanduser(m) for m in list(JUDGES.values()) + list(SELECTIVE.values())]
    if BACKEND == "hf" and paths and all(Path(m).is_dir() for m in paths):
        ENV.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        print("local models:", ", ".join(paths))


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
        root = Path(RUN_ROOT).expanduser()
    syn, real = root / SYN_RUN, root / REAL_RUN
    need = {syn / "filler_check" / "rows.jsonl": "synthetic run (planted filler, T1)",
            real / "filler_check_real" / "rows.jsonl": "real-text run (planted filler, R3)"}
    missing = [f"{what}: {p}" for p, what in need.items() if not p.exists()]
    if missing:
        raise SystemExit("Not found (the runs of the T4 notebooks: in Google Drive, or on a server copied into "
                         "RUN_ROOT; or fix SYN_RUN / REAL_RUN):\n  " + "\n  ".join(missing))
    if GPU is not None:
        ENV["CUDA_VISIBLE_DEVICES"] = str(GPU)
    pkg = install()
    hf_token()
    local_models()
    cli = [sys.executable, "-m", "drybench.cli"]
    sets = {syn: ["planted_synthetic"],
            real: ["planted_real"] + (["human_poddar"] if (real / "human_check" / "rows.jsonl").exists() else [])}
    extra = ["--poddar-data", PODDAR_DATA] if PODDAR_DATA else []
    for method, models in (("judge", JUDGES), ("selective", SELECTIVE)):
        for label, model_id in models.items():
            for out, names in sets.items():
                print(f"\n======== {method} {label} on {out.name}: {', '.join(names)} ========")
                run(cli + ["baselines", "--run", out, "--set", *names, "--method", method, "--label", label,
                           "--backend", BACKEND, "--model", os.path.expanduser(model_id), "--dtype", DTYPE,
                           "--max-batch-tokens", MAX_BATCH_TOKENS]
                    + (extra if "human_poddar" in names else []))
    print("\n======== tables ========")
    for out in (syn, real):
        run([sys.executable, "-m", "drybench.report", out, "--quiet"])
    dev = root / f"{ZIP_NAME}_confirmatory.md"
    run(cli + ["confirm", "--synthetic", syn, "--real", real, "--out", dev])
    stage = Path("/content/dev_zip") if Path("/content").exists() else Path.cwd() / "dev_zip"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for out in (syn, real):
        shutil.copytree(out, stage / out.name, ignore=shutil.ignore_patterns("*.zip"))
    shutil.copyfile(dev, stage / dev.name)
    archive = shutil.make_archive(str(root / ZIP_NAME), "zip", root_dir=stage)
    shutil.rmtree(stage, ignore_errors=True)
    try:
        from IPython.display import Markdown, display

        display(Markdown(dev.read_text().replace("$", r"\$")))  # "$10.00" is not LaTeX
    except ImportError:
        pass
    print(f"\nTables: {syn / 'tables_synthetic.md'}, {real / 'tables_real.md'}\nDevelopment values of the "
          f"confirmatory tests: {dev}\nEverything: {archive}")
    print(f"Total time of this session: {(time.time() - t0) / 60:.1f} min")


PKG_B64 = "__PKG_B64__"

main()
