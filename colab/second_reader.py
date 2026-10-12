# Filler in LLM answers — score the same responses with a second reader (Colab T4, ~45 min).
# Does the benchmark depend on the reader? Run after the main cell, in the same runtime
# (or with the run folder restored). Results go to <RUN>/reader2_<LABEL2>/.

# ---- settings ----
RUN = "auto"                                  # the main cell's folder (Google Drive or /content), or a path
READER2 = "microsoft/Phi-3.5-mini-instruct"  # another model family than the main reader
LABEL2 = "phi3.5-mini"
DTYPE = "fp16"                                # T4 has no bf16
FILLER_CHECK_N = 3                             # simulated items per cell for the filler-map check (0 = skip)
FULL_RUN = False                              # True: allow scoring every response from scratch (~60 min on a T4)
# ---- end settings ----

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENV = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
PROGRESS = re.compile(r"^(?:generated|signals|scored) (\d+)/(\d+)")


def run(args):
    args = [str(a) for a in args]
    shown = ["drybench"] + args[3:] if args[1:3] == ["-m", "drybench.cli"] else args
    print("$", " ".join(shown), flush=True)
    t0 = time.time()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    for line in p.stdout:
        m = PROGRESS.match(line.strip())
        if m and int(m.group(1)) % max(10, int(m.group(2)) // 10) and m.group(1) != m.group(2):
            continue
        print("   ", line, end="", flush=True)
    code = p.wait()
    print(f"    [{time.time() - t0:.0f}s]", flush=True)
    if code:
        raise RuntimeError(f"step failed with exit code {code}: {' '.join(args)}")


def ci(x, ratio=False, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "n/a"
    b = np.random.default_rng(0).choice(x, (n, len(x))).mean(axis=1)
    m, lo, hi = x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)
    if ratio:
        return f"×{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def compare(out, r2):
    pd.set_option("display.width", 200)
    print("\n================ MAIN READER vs", LABEL2, "================")
    checks = {"main": out / "reader_check.json", LABEL2: r2 / "reader_check.json"}
    print(pd.DataFrame({k: (lambda v: {"AUC present vs absent": v["auc_present_vs_absent"],
                                       "gold unreadable": v["reader_fail_rate"],
                                       "wrong-statement detection": v.get("misleading_check_active", np.nan),
                                       "nats per fact": v["gain_present_median"]})(json.loads(p.read_text()))
                        for k, p in checks.items()}).round(3).to_string())

    rows = {}
    for k, path in (("main", out / "filler_check" / "summary.json"), (LABEL2, r2 / "filler_check" / "summary.json")):
        if path.exists():
            ws = json.loads(path.read_text())
            for key, name in (("loo", "τ=0.5"), ("loo_t1", "τ=1"), ("loo_t2", "τ=2")):
                if key in ws:
                    rows[f"{k}, {name}"] = {"no fact lost": ws[key]["safety_all_ok"],
                                            "precision": ws[key]["filler_precision"],
                                            "recall": ws[key]["filler_recall"], "MAE": ws[key]["mae"]}
    if rows:
        print("\nFiller-map check on simulated responses with known filler:")
        print(pd.DataFrame(rows).T.round(3).to_string())

    a = pd.read_csv(out / "report" / "item_table.csv")
    b = pd.read_csv(r2 / "report" / "item_table.csv")
    m = a.merge(b, on=["model", "variant", "item_id"], suffixes=("_1", "_2"))
    both = m[(m["status_1"] == "ok") & (m["status_2"] == "ok")]
    print(f"\nPer-response agreement, Spearman ({len(both)} responses scored by both; "
          "pooled / mean within model × prompt; W_ablation = W with the looser tolerance):")
    agree = {}
    for c in ("W", "W_ablation", "W_misleading", "kappa", "facts", "overhead"):
        if f"{c}_1" not in both or f"{c}_2" not in both:
            continue
        pooled = both[f"{c}_1"].corr(both[f"{c}_2"], method="spearman")
        within = np.array([d[f"{c}_1"].corr(d[f"{c}_2"], method="spearman")
                           for _, d in both.groupby(["model", "variant"])], float)
        agree[c] = {"pooled": pooled, "within": np.nanmean(within) if np.isfinite(within).any() else np.nan}
    print(pd.DataFrame(agree).T.round(3).to_string())

    sa = pd.read_csv(out / "report" / "span_table.csv")
    sb = pd.read_csv(r2 / "report" / "span_table.csv")
    s = sa.merge(sb, on=["model", "variant", "item_id", "span_idx"], suffixes=("_1", "_2"))
    w1 = s["kind_1"].isin(["uninformative", "duplicate", "removed"]).to_numpy()
    w2 = s["kind_2"].isin(["uninformative", "duplicate", "removed"]).to_numpy()
    po = float((w1 == w2).mean())
    pe = w1.mean() * w2.mean() + (1 - w1.mean()) * (1 - w2.mean())
    print(f"\nSpan-level filler map: same decision for {100 * po:.1f}% of {len(s)} spans, "
          f"Cohen's κ = {(po - pe) / (1 - pe):.2f} (filler share {w1.mean():.3f} vs {w2.mean():.3f})")

    ga = pd.read_csv(out / "report" / "groups.csv", index_col=0)
    gb = pd.read_csv(r2 / "report" / "groups.csv", index_col=0)
    cols = [c for c in ("W_mean", "W_misleading_mean", "kappa_mean", "facts_mean", "overhead_gmean")
            if c in ga and c in gb]
    print("\nPer model × prompt:")
    print(pd.concat({"main": ga[cols], LABEL2: gb[cols]}, axis=1).round(3).to_string())
    for c in ("W_mean", "overhead_gmean", "facts_mean"):
        if c in cols:
            print(f"  ranking of the {len(ga)} groups by {c}: Kendall τ = {ga[c].corr(gb[c], method='kendall'):.2f}")

    print(f"\nH3 with {LABEL2} as the reader (concise − default; overhead as a ratio):")
    piv = b.pivot_table(index=["model", "item_id"], columns="variant", values=["W", "facts", "overhead"],
                        aggfunc="first", dropna=False)
    for mdl, g in piv.groupby(level=0):
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(g[("overhead", "concise")] / g[("overhead", "default")])
        print(f"  {mdl}: ΔW {ci(g[('W', 'concise')] - g[('W', 'default')])}  "
              f"Δfacts {ci(g[('facts', 'concise')] - g[('facts', 'default')])}  overhead {ci(lr, ratio=True)}")
    print(f"\nAll tables: {r2 / 'report'}")


def find_run(run, name="t4_n20"):
    """RUN = "auto": the main cell's folder in this session (RUN_DIR), else the run in Google Drive,
    else the one in /content; a path is used as given."""
    if run != "auto":
        return Path(run)
    if "RUN_DIR" in globals():
        return Path(globals()["RUN_DIR"])
    name = globals().get("RUN_NAME", name)
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception:
            pass  # not in Colab, or Drive declined: look in /content
    on_drive = Path("/content/drive/MyDrive/drybench_runs") / name
    return on_drive if on_drive.exists() else Path("/content/drybench_runs") / name


def main():
    t_start = time.time()
    try:
        from google.colab import userdata

        tok = userdata.get("HF_TOKEN")
        if tok:
            ENV["HF_TOKEN"] = tok
    except Exception:
        pass  # public models do not need a token
    out = find_run(RUN)
    print("Run:", out)
    need = [out / "items.jsonl", out / "responses.jsonl", out / "signals.jsonl", out / "report" / "item_table.csv"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit("Run the main cell first; not found: " + ", ".join(missing))
    r2 = out / f"reader2_{LABEL2}"
    if not (r2 / "scored.jsonl").exists() and not FULL_RUN:
        raise SystemExit(f"No second-reader scores in {r2}: scoring every response takes ~60 min on a T4. "
                         "Set FULL_RUN = True to start it (or restore the folder from a saved run).")
    r2.mkdir(exist_ok=True)
    cli = [sys.executable, "-m", "drybench.cli"]
    version = subprocess.run([sys.executable, "-c", "import drybench; print(drybench.SCORING_VERSION)"],
                             capture_output=True, text=True, env=ENV).stdout.strip()
    main_v = json.loads((out / "versions.json").read_text()).get("scoring") if (out / "versions.json").exists() else None
    if main_v != version:
        raise SystemExit("The main run was scored by another drybench version; run the main cell first.")
    v2 = r2 / "versions.json"
    old_v = json.loads(v2.read_text()).get("scoring") if v2.exists() else (
        "older" if (r2 / "scored.jsonl").exists() else version)
    if old_v != version:  # every score, or only those the new version changed for this reader
        got = subprocess.run([str(a) for a in cli + ["prune-scores", "--dir", r2, "--items", out / "items.jsonl",
                                                     "--reader", READER2, "--from", old_v, "--json"]],
                             capture_output=True, text=True, env=ENV)
        if got.returncode:
            raise RuntimeError("prune-scores failed:\n" + got.stdout + got.stderr)
        res = json.loads(got.stdout.strip().splitlines()[-1])
        what = {"all": "every response", "missing": "the answers to items with a missing fact",
                "none": "nothing (this reader's scores did not change)"}[res["plan"]]
        print(f"Scoring changed since the last second-reader run (v{old_v} → v{version}): rescoring {what}.")
        for line in res["changes"]:
            print("    " + line)
    v2.write_text(json.dumps({"scoring": version}))
    try:
        import torch

        dtype = DTYPE if torch.cuda.is_available() else "auto"
    except ImportError:
        dtype = "auto"

    print("[1/4] reader check")
    if not (r2 / "reader_check.json").exists():
        run(cli + ["check-reader", "--items", out / "items.jsonl", "--reader-backend", "hf", "--reader", READER2,
                   "--reader-dtype", dtype, "--n", 30, "--out", r2 / "reader_check.json"])

    print("\n[2/4] filler-map check on simulated responses with known filler")
    scripts = sorted(glob.glob("/content/drybench_pkg_*/scripts/pilot_validation.py")
                     + glob.glob("drybench_pkg_*/scripts/pilot_validation.py"), key=os.path.getmtime)
    if FILLER_CHECK_N and scripts and not (r2 / "filler_check" / "summary.json").exists():
        run([sys.executable, scripts[-1], "--reader-backend", "hf", "--reader", READER2, "--reader-dtype", dtype,
             "--out-dir", r2 / "filler_check", "--n-per-cell", FILLER_CHECK_N, "--greedy-conditions", "none",
             "--no-plots", "--quiet"])

    print("\n[3/4] filler maps of the same responses")
    run(cli + ["score", "--items", out / "items.jsonl", "--responses", out / "responses.jsonl",
               "--out", r2 / "scored.jsonl", "--reader-backend", "hf", "--reader", READER2, "--reader-dtype", dtype,
               "--reader-max-batch-tokens", 4096, "--mode", "loo", "--ablation-abs-tol", 2.0])

    print("\n[4/4] analysis")
    run(cli + ["analyze", "--items", out / "items.jsonl", "--scored", r2 / "scored.jsonl",
               "--signals", out / "signals.jsonl", "--out-dir", r2 / "report", "--n-boot", 500])
    compare(out, r2)
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")


main()
