# Filler in LLM answers, synthetic reports — full run on a Colab T4 GPU
# 1) Runtime → Change runtime type → T4 GPU.   2) Run this cell (~55 min with the defaults:
#    300 items × 3 models × 2 prompts; N_PER_CELL = 6 takes ~20 min).
# Re-running the cell resumes: every step appends to JSONL files and skips finished rows.
# A newer version of this cell on an old run rescores it and reuses the generated responses.
# The drybench package (v__VERSION__) is embedded as PKG_B64 (notebook: the hidden cell "Package"); nothing to upload.

# ---- settings ----
N_PER_CELL = 20                      # items per knob cell; 15 cells → 300 items
MODELS = {                           # label: Hugging Face id of an evaluated (instruct) model
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",  # optional 4th model (3.8B), ~+25 min
}
READER = "Qwen/Qwen2.5-3B-Instruct"  # reads the responses; keep it out of MODELS
VARIANTS = ["default", "concise"]    # "concise" adds "Be concise." to the system prompt (H3)
FILLER_CHECK_N = 3                    # simulated items per cell to test the filler map with this reader (0 = skip)
MAX_NEW_TOKENS = 320
SCORE_MODE = "loo"                   # "greedy" is exact but several times slower
SAMPLED = 0                          # >0: also sampled answer entropy from N samples (slower)
DTYPE = "fp16"                       # T4 has no bf16; use "fp32" if a step reports NaN logits
BATCH = 16                           # generation batch size
USE_DRIVE = True                     # keep outputs in Google Drive (survives disconnects); False: /content
RUN_ROOT = "/content/drybench_runs"
RUN_NAME = "t4_n20_v12"              # new name = fresh run; same name = resume
ALLOW_CPU = False
# ---- end settings ----

import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ENV = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
PROGRESS = re.compile(r"^(?:generated|signals|scored) (\d+)/(\d+)")


def run(args, check=True):
    args = [str(a) for a in args]
    shown = ["drybench"] + args[3:] if args[1:3] == ["-m", "drybench.cli"] else args
    print("$", " ".join(shown), flush=True)
    t0 = time.time()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:
            continue
        m = PROGRESS.match(line.strip())  # keep about ten progress lines per step
        if m and int(m.group(1)) % max(10, int(m.group(2)) // 10) and m.group(1) != m.group(2):
            continue
        print("   ", line, end="", flush=True)
    code = p.wait()
    print(f"    [{time.time() - t0:.0f}s]", flush=True)
    if check and code != 0:
        raise RuntimeError(f"step failed with exit code {code}: {' '.join(args)}")
    return code


def install():
    if "PKG_B64" not in globals():
        raise SystemExit('Run the cell "Package" first (its code is hidden), then this cell.')
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    pkg = (base / f"drybench_pkg_{digest}").resolve()
    marker = pkg / ".installed"
    if not marker.exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
        ENV["PIP_BREAK_SYSTEM_PACKAGES"] = "1"  # harmless on Colab; needed on PEP 668 systems
        run([sys.executable, "-m", "pip", "install", "-q", str(pkg), "transformers>=4.46", "accelerate",
             "matplotlib"])
        marker.touch()
    return pkg


def hf_token():
    try:
        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        if tok:
            ENV["HF_TOKEN"] = tok
    except Exception:
        pass  # public models do not need a token


def num(x, spec="+.3f"):
    return "n/a" if x != x else format(x, spec)  # x != x is True only for NaN


def fmt_ci(p, s, spec="+.3f"):
    return f"{num(p[s], spec)} [{num(p[s + '_lo'], spec)}, {num(p[s + '_hi'], spec)}]"


def verdict(p, s, up, down, flat):
    lo, hi = p[s + "_lo"], p[s + "_hi"]
    if lo != lo or hi != hi:
        return "n/a"
    return up if lo > 0 else (down if hi < 0 else flat)


def paired_ci(x, ratio=False, n_boot=2000, seed=0):
    """Mean of paired differences (or geometric mean of ratios) with a bootstrap 95% CI."""
    import numpy as np

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "n/a"
    boots = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(axis=1)
    m, lo, hi = x.mean(), *np.quantile(boots, [0.025, 0.975])
    if ratio:
        return f"×{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def show_results(out, report):
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 200)
    rc = json.loads((out / "reader_check.json").read_text())
    ok_reader = rc["auc_present_vs_absent"] >= 0.95 and rc["reader_fail_rate"] <= 0.1
    print("\n================ RESULTS ================")
    print(f"Reader check: AUC present vs absent = {rc['auc_present_vs_absent']:.3f}, "
          f"cannot read gold answer in {100 * rc['reader_fail_rate']:.0f}% of items "
          f"→ {'reader OK' if ok_reader else 'WEAK READER: treat W with caution'}")
    active = rc.get("misleading_check_active", float("nan"))
    if active == active:
        print(f"Wrong-statement detection works for {100 * active:.0f}% of questions "
              f"(reader answers 'unknown' to an empty message with median p = "
              f"{rc['p_unknown_empty_median']:.2f}).")
    else:
        print("Wrong-statement detection is off: the reader has no chat template.")

    wc = out / "filler_check" / "summary.json"
    if wc.exists():
        ws = json.loads(wc.read_text())
        names = {"loo": "tolerance 0.5 nats (default)", "loo_t1": "tolerance 1 nat",
                 "loo_t2": "tolerance 2 nats", "loo_global": "one budget for all facts"}
        tab = pd.DataFrame({lab: {"no fact lost": ws[k]["safety_all_ok"], "filler precision": ws[k]["filler_precision"],
                                  "filler recall": ws[k]["filler_recall"], "MAE vs true W": ws[k]["mae"]}
                            for k, lab in names.items() if k in ws}).T
        print(f"\nFiller-map check: {ws['n_responses']} simulated responses with known filler, read by this reader:")
        print(tab.round(3).to_string())
        kept = {k.replace("filler:", ""): v for k, v in ws["loo"]["kept_share_by_label"].items()
                if k.startswith("filler:") and v > 0.05}
        if kept:
            print("  filler the default setting leaves in: " + ", ".join(
                f"{k} {100 * v:.0f}%" for k, v in sorted(kept.items(), key=lambda kv: -kv[1])))

    groups = pd.read_csv(report / "groups.csv", index_col=0)
    print("\nPer model × prompt:\n"
          "  W = filler share of tokens (uninformative + repeated facts); W_wrong = wrong or confusing statements "
          "(errors, not filler);\n"
          "  κ = share of the requested information delivered; facts = share of requested facts the reader "
          "gets right; bits/tok = density;\n"
          "  overhead = tokens per delivered bit relative to the minimal correct answer (1.0 = as lean)\n"
          "           = ×removable (sentences that can go) × ×wordiness (kept tokens per delivered bit vs the gold answer)")
    cols = {"n": "n", "share_ok": "scored", "tokens_mean": "tokens", "W_mean": "W",
            "W_misleading_mean": "W_wrong", "kappa_mean": "κ", "facts_mean": "facts",
            "density_mean": "bits/tok", "overhead_gmean": "overhead", "removable_gmean": "×removable",
            "wordiness_gmean": "×wordiness"}
    print(groups[[c for c in cols if c in groups]].rename(columns=cols).round(3).to_string())
    if "W_ablation_mean" in groups and groups["W_ablation_mean"].notna().any():
        diff = (groups["W_ablation_mean"] - groups["W_mean"]).abs().max()
        print(f"Tolerance check: with a looser tolerance (2 nats instead of 0.5) mean W changes by at most "
              f"{diff:.3f} across groups.")

    prof_path = report / "profiles.csv"
    prof = pd.read_csv(prof_path, index_col=0) if prof_path.exists() and prof_path.stat().st_size > 5 else None
    if prof is None or prof.empty:
        print("\nNo profiles: too few scored responses with entropy signals.")
        return
    print("\nProfiles (point [95% CI]). α = filler on easy items the model is sure about; "
          "β = filler vs uncertainty (β_ctrl: within the same item design);\n"
          "γ, γ_filler = substance / filler tokens per requested fact (plain items):")
    for g, p in prof.iterrows():
        print(f"  {g}: n={int(p['n'])}  α={num(p['alpha'], '.3f')} (n={int(p['alpha_n'])})  "
              f"β={fmt_ci(p, 'beta')}  β_ctrl={fmt_ci(p, 'beta_ctrl')}\n"
              f"  {' ' * len(g)}  γ={fmt_ci(p, 'gamma', '+.2f')}  γ_filler={fmt_ci(p, 'gamma_filler', '+.2f')}")

    print("\nH2 — does filler follow uncertainty while substance follows the number of facts?")
    for g, p in prof.iterrows():
        print(f"  {g}: filler vs uncertainty: {verdict(p, 'beta', 'grows', 'falls', 'no significant trend')} "
              f"(same design: {verdict(p, 'beta_ctrl', 'grows', 'falls', 'no significant trend')}); "
              f"substance vs k: {verdict(p, 'gamma', 'grows', 'falls', 'no significant trend')}; "
              f"filler vs k: {verdict(p, 'gamma_filler', 'grows', 'falls', 'no significant trend')}")

    it = pd.read_csv(report / "item_table.csv")
    if {"default", "concise"} <= set(VARIANTS):
        print("\nH3 — 'Be concise.' vs default on the same items (concise − default, mean [95% CI]; "
              "overhead as a ratio):")
        piv = it.pivot_table(index=["model", "item_id"], columns="variant", aggfunc="first", dropna=False,
                             values=["tokens", "W", "W_misleading", "kappa", "facts", "overhead"])
        rows = {}
        for m, g in piv.groupby(level=0):
            def d(c, op=np.subtract):
                if (c, "concise") not in g.columns or (c, "default") not in g.columns:
                    return np.array([])
                return op(g[(c, "concise")].to_numpy(float), g[(c, "default")].to_numpy(float))

            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log(d("overhead", np.divide))
            rows[m] = {"Δtokens": paired_ci(d("tokens")), "ΔW": paired_ci(d("W")),
                       "ΔW_wrong": paired_ci(d("W_misleading")), "Δκ": paired_ci(d("kappa")),
                       "Δfacts": paired_ci(d("facts")), "overhead": paired_ci(log_ratio, ratio=True)}
        print(pd.DataFrame(rows).T.to_string())

    att_path = report / "attribution.csv"
    if att_path.exists():
        att = pd.read_csv(att_path, index_col=0)
        ws_path = report / "filler_spans.csv"
        if ws_path.exists():
            att["filler spans"] = pd.read_csv(ws_path).groupby("group").size()
        print("\nH1 — share of filler tokens by entropy cell:")
        print(att.round(3).to_string())
        enr_path = report / "attribution_enrichment.csv"
        if enr_path.exists():
            print("    the same, relative to the share of ALL response tokens in the cell "
                  "(> 1: filler is over-represented there):")
            print(pd.read_csv(enr_path, index_col=0).round(2).to_string())

    try:
        from IPython.display import Image, display

        for f in ("fig_filler_vs_entropy.png", "fig_tokens_vs_k.png", "fig_attribution.png"):
            if (report / f).exists():
                display(Image(filename=str(report / f)))
    except ImportError:
        pass
    print(f"\nAll tables and figures: {report}")


def main():
    t_start = time.time()
    try:
        import torch

        gpu = torch.cuda.is_available()
        print("GPU:", torch.cuda.get_device_name(0) if gpu else "none")
    except ImportError:
        gpu = False
    if not gpu and not ALLOW_CPU:
        raise SystemExit("No GPU. In Colab: Runtime → Change runtime type → T4 GPU, then rerun.")
    dtype = DTYPE if gpu else "auto"
    if USE_DRIVE:
        from google.colab import drive

        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive/drybench_runs")
    else:
        root = Path(RUN_ROOT)
    out = root / RUN_NAME
    out.mkdir(parents=True, exist_ok=True)
    print("Outputs:", out)
    pkg = install()
    hf_token()
    cli = [sys.executable, "-m", "drybench.cli"]
    items, resp = out / "items.jsonl", out / "responses.jsonl"
    signals, scored, report = out / "signals.jsonl", out / "scored.jsonl", out / "report"

    # Scoring or signals changed since these outputs were made? Recompute them, keep responses.
    got = subprocess.run([sys.executable, "-c", "import drybench, json; print(json.dumps("
                          "{'scoring': drybench.SCORING_VERSION, 'signals': drybench.SIGNALS_VERSION}))"],
                         capture_output=True, text=True, env=ENV).stdout.strip()
    if not got:
        raise RuntimeError("drybench is not importable after installation; rerun the cell")
    new = json.loads(got)
    vfile, legacy = out / "versions.json", out / "scoring_version.txt"
    if vfile.exists():
        old = json.loads(vfile.read_text())
    else:  # runs made before versions.json: signals last changed with scoring version 2
        s_old = legacy.read_text().strip() if legacy.exists() else "1"
        old = {"scoring": s_old, "signals": "2" if int(s_old) >= 2 else "1"}
    if old["scoring"] != new["scoring"] and scored.exists():
        # every score, or only those the new version changed for this reader (drybench.rescore)
        got = subprocess.run([str(a) for a in cli + ["prune-scores", "--dir", out, "--items", items, "--reader",
                                                     READER, "--from", old["scoring"], "--json"]],
                             capture_output=True, text=True, env=ENV)
        if got.returncode:
            raise RuntimeError("prune-scores failed:\n" + got.stdout + got.stderr)
        res = json.loads(got.stdout.strip().splitlines()[-1])
        if res["plan"] in ("all", "synthetic"):  # v1.2.1 ("synthetic"): the answer forms of synthetic items
            print(f"Scoring changed (v{old['scoring']} → v{new['scoring']}): updating answer forms of the "
                  "items, recomputing the reader checks and filler maps; responses and signals are reused.")
            if items.exists():
                run(cli + ["items", "--out", items, "--update-probes"])
        else:
            what = ("the answers to items with a missing fact" if res["plan"] == "missing"
                    else "nothing for this reader")
            print(f"Scoring changed (v{old['scoring']} → v{new['scoring']}): rescoring {what}; "
                  "everything else is kept.")
        for line in res["changes"]:
            print("    " + line)
    if old["signals"] != new["signals"] and signals.exists():
        print(f"Signals changed (v{old['signals']} → v{new['signals']}): recomputing entropy signals.")
        signals.unlink()
        shutil.rmtree(report, ignore_errors=True)
    vfile.write_text(json.dumps(new))
    legacy.unlink(missing_ok=True)

    print("\n[1/6] items")
    if not items.exists():
        run(cli + ["items", "--out", items, "--n-per-cell", N_PER_CELL])
    n_items = sum(1 for _ in open(items))
    print(f"    {n_items} items")

    print("\n[2/6] reader check")
    if not (out / "reader_check.json").exists():
        run(cli + ["check-reader", "--items", items, "--reader-backend", "hf", "--reader", READER,
                   "--reader-dtype", dtype, "--n", 30, "--out", out / "reader_check.json"])

    print("\n[3/6] filler-map check on simulated responses with known filler")
    wc = out / "filler_check"
    if FILLER_CHECK_N and not (wc / "summary.json").exists():
        run([sys.executable, pkg / "scripts" / "pilot_validation.py", "--reader-backend", "hf",
             "--reader", READER, "--reader-dtype", dtype, "--out-dir", wc, "--n-per-cell", FILLER_CHECK_N,
             "--greedy-conditions", "none", "--no-plots", "--quiet"])

    print("\n[4/6] generate responses + entropy signals")
    for label, model_id in MODELS.items():
        run(cli + ["generate", "--items", items, "--out", resp, "--backend", "hf", "--model", model_id,
                   "--model-name", label, "--variant", *VARIANTS, "--dtype", dtype,
                   "--batch-size", BATCH, "--max-new-tokens", MAX_NEW_TOKENS])
        run(cli + ["signals", "--items", items, "--responses", resp, "--out", signals, "--backend", "hf",
                   "--model", model_id, "--model-name", label, "--dtype", dtype, "--sampled", SAMPLED])

    print("\n[5/6] filler maps (reader scoring)")
    run(cli + ["score", "--items", items, "--responses", resp, "--out", scored, "--reader-backend", "hf",
               "--reader", READER, "--reader-dtype", dtype, "--reader-max-batch-tokens", 4096,
               "--mode", SCORE_MODE, "--ablation-abs-tol", 2.0])

    print("\n[6/6] analysis")
    run(cli + ["analyze", "--items", items, "--scored", scored, "--signals", signals,
               "--out-dir", report, "--n-boot", 500])
    show_results(out, report)
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    return out


PKG_B64 = "__PKG_B64__"

RUN_DIR = str(main())  # the cells below find the run here
