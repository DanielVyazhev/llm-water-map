# Missing facts: which question lets the reader tell an answer that SAYS a fact is not stated from an
# answer that only does not mention it? v0.7.2's question did not (CUAD: 19 of 21 correct abstentions
# deleted as filler). Round 1 (hint, plain, silence, yesno): yesno worked for Qwen, plain for Phi, none
# for both. Round 2 compares plain, yesno and three new wordings (yesno2, contains, choice) on the real
# run's missing-fact answers (Google Drive) and the synthetic missing-fact items, for the main reader,
# Phi-3.5 and a candidate second reader from another family (IBM Granite 3.3 2B), with the usual reader
# check for each. GPU (T4), ~30 min; reads the run, changes nothing in it.
# The drybench package (v__VERSION__) is embedded as PKG_B64 (notebook: the hidden cell "Package"); nothing to upload.

# ---- settings ----
RUN = "/content/drive/MyDrive/drybench_runs/real_t4_n15_v12"  # the real-text run
READERS = {"main": "Qwen/Qwen2.5-3B-Instruct",
           "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
           "granite-3.3-2b": "ibm-granite/granite-3.3-2b-instruct"}
STYLES = ["plain", "yesno", "yesno2", "contains", "choice"]
SYNTHETIC_N = 20       # synthetic items per cell (their gold answers; 2 missing-fact cells)
DTYPE = "fp16"         # T4 has no bf16
BACKEND = "hf"
# ---- end settings ----

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ENV = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PIP_BREAK_SYSTEM_PACKAGES="1")


def run(args, show=True):
    args = [str(a) for a in args]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if show and "Warning: You are sending unauthenticated requests" not in line:
            print(line, end="", flush=True)
    if p.wait():
        print("".join(lines[-30:]))
        raise RuntimeError(f"failed with exit code {p.returncode}")


def install():
    if "PKG_B64" not in globals():
        raise SystemExit('Run the cell "Package" first (its code is hidden), then this cell.')
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    pkg = (base / f"drybench_pkg_{digest}").resolve()
    if not (pkg / ".installed").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
        run([sys.executable, "-m", "pip", "install", "-q", str(pkg), "transformers>=4.46", "accelerate"])
        (pkg / ".installed").touch()
    return pkg


def main():
    run_dir = Path(RUN)
    if str(run_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive

        drive.mount("/content/drive")
    if not (run_dir / "items.jsonl").exists():
        raise SystemExit(f"No run in {run_dir}: check RUN.")
    pkg = install()
    sys.path.insert(0, str(pkg))  # the extracted package, importable in this kernel without a restart
    from drybench.validate import missing_check_summary

    chat = ["--chat"] if BACKEND == "mock" else []
    summaries = {}
    for label, model in READERS.items():
        print(f"\n=== reader {label}: {model}")
        rc = run_dir / f"missing_check_reader_{label}.json"
        run([sys.executable, "-m", "drybench.cli", "check-reader", "--items", run_dir / "items.jsonl", "--n", 40,
             "--reader-backend", BACKEND, "--reader", model, "--reader-dtype", DTYPE, "--out", rc], show=False)
        r = json.loads(rc.read_text())
        print(f"reader check: AUC present vs absent {r['auc_present_vs_absent']:.3f}, gold unreadable "
              f"{r['reader_fail_rate']:.3f}, wrong-statement detection {r.get('misleading_check_active', float('nan')):.3f}")
        out = run_dir / f"missing_check_{label}.json"
        run([sys.executable, "-m", "drybench.cli", "missing-check", "--run", run_dir, "--synthetic-n", SYNTHETIC_N,
             "--styles", *STYLES, "--reader-backend", BACKEND, "--reader", model, "--reader-dtype", DTYPE,
             "--reader-max-batch-tokens", 4096, "--out", out] + chat)
        summaries[label] = missing_check_summary(json.loads(out.read_text()))
    main_label, others = next(iter(READERS)), list(READERS)[1:]
    print("\n=== which question passes for which reader (abstentions kept ≥ 0.9, gold and abstentions read ≥ 0.8, "
          "answers without one read ≤ 0.2, p with no message ≤ 0.05)")
    for style in STYLES:
        print(f"{style}: " + ", ".join(f"{lab} {'yes' if summaries[lab][style]['pass'] else 'no'}" for lab in READERS))
    def score(v):  # abstentions kept + gold read + abstentions read − answers without one read as one
        val = lambda key: 0.0 if v[key] != v[key] else v[key]  # noqa: E731
        return val("kept") + val("gold_read") + val("abstaining_read") - val("other_read")

    both = sorted(((min(score(summaries[main_label][s]), score(summaries[o][s])), s, o) for o in others for s in STYLES
                   if summaries[main_label][s]["pass"] and summaries[o][s]["pass"]), reverse=True)
    if both:
        print("\nPasses for the main reader and a second reader: "
              + "; ".join(f"{s} with {o} (worse of the two: {sc:.2f} of 3)" for sc, s, o in both) + ".")
    else:
        print("\nNo question passes for the main reader and a second reader together.")
    print("Results are saved next to the run: missing_check_<reader>.json, missing_check_reader_<reader>.json.")


PKG_B64 = "__PKG_B64__"

main()
