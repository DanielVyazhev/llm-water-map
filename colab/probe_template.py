# Filler in LLM answers — which of this machine's models the benchmark can use, and in which role (a server with
# local models; ~2–5 min per model). For every folder in MODELS: does it load, does it answer a report request in
# plain text with the facts, does it read the facts of a gold answer (reader), does it label an opener and the facts
# right (judge), how much GPU memory it takes. Nothing is downloaded or installed: the models are local folders and
# the package (v__VERSION__) runs from its folder with this machine's libraries (numpy, pandas, scipy, torch,
# transformers). Writes OUT (one JSON line per model) and prints a table. The package is embedded as PKG_B64
# (notebook: the hidden cell "Package").

# ---- settings ----
MODELS = []                          # local model folders
SHA256 = []                          # folders whose weight files to hash (the main reader: to compare with the hub)
DTYPE = "bf16"
GPU = None                           # a machine with several GPUs: the one to use, e.g. "1"
OUT = "~/drybench_probe.jsonl"
# ---- end settings ----

import base64
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")


def extract():
    if "PKG_B64" not in globals():
        raise SystemExit('Run the cell "Package" first (its code is hidden), then this cell.')
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    pkg = (Path.cwd() / f"drybench_pkg_{digest}").resolve()
    if not (pkg / "drybench").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
    return pkg


def main():
    if not MODELS:
        raise SystemExit("MODELS is empty: list the local model folders.")
    if GPU is not None:
        ENV["CUDA_VISIBLE_DEVICES"] = str(GPU)
    pkg = extract()
    out = Path(OUT).expanduser()
    if out.exists():
        out.unlink()  # a fresh table each time
    cmd = [sys.executable, str(pkg / "scripts" / "probe_models.py"), *MODELS, "--out", str(out), "--dtype", DTYPE]
    for path in SHA256:
        cmd += ["--sha256", path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"the probe failed with exit code {p.returncode}")


PKG_B64 = "__PKG_B64__"

main()
