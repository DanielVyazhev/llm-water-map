# Filler in LLM answers, synthetic reports — all tables for the paper from a finished run
# (CPU, ~1 min; no GPU needed). Run after the main and second-reader cells, or in a new session after
# the restore cell. Writes <RUN>/paper/ (tables.md, CSVs, a blind sample for human labels) and
# downloads it as a zip. The tables are built by the same code as the real-text tables (same columns).
# The drybench package (v__VERSION__) is embedded at the bottom; nothing to install.

# ---- settings ----
RUN = "auto"                           # the main cell's folder (Google Drive or /content), or a path
N_ANNOTATE = 30                        # responses per model × prompt for human labels (0 = none)
# ---- end settings ----

import base64
import hashlib
import io
import json
import random
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import pandas as pd


def load_package():
    """Extract the embedded package next to the run and import it from there (no pip)."""
    digest = hashlib.sha1(PKG_B64.encode()).hexdigest()[:10]
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    pkg = (base / f"drybench_src_{digest}").resolve()
    if not (pkg / "drybench").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(PKG_B64))).extractall(pkg)
    for name in [m for m in sys.modules if m == "drybench" or m.startswith("drybench.")]:
        del sys.modules[name]  # an older copy imported earlier in this session
    sys.path.insert(0, str(pkg))


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
    load_package()
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty bootstrap slices in tiny groups
    from drybench.report import _readers, _results, synthetic_tables
    from drybench.types import load_items

    run = find_run(RUN)
    print("Run:", run)
    if not (run / "scored.jsonl").exists():
        raise SystemExit(f"No scored run in {run}: run the main cell (or the restore cell) first.")
    out = run / "paper"
    out.mkdir(exist_ok=True)
    text = synthetic_tables(run)
    (out / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    items = load_items(run / "items.jsonl")
    _, R = _readers(run, items)
    for r in R:
        _results(r["items"]).to_csv(out / f"results_{r['label']}.csv")

    # blind sample for human labels
    if N_ANNOTATE:
        rng = random.Random(0)
        by_id = {i.item_id: i for i in items}
        resp = {(j["model"], j.get("variant", "default"), j["item_id"]): j["response"]
                for j in map(json.loads, open(run / "responses.jsonl"))}
        spans = {r["label"]: r["spans"] for r in R}
        main_ok = R[0]["items"][R[0]["items"]["status"] == "ok"].copy()
        picked = []
        for g, d in main_ok.groupby("group"):
            rows = d.sample(frac=1.0, random_state=0)
            per = max(1, N_ANNOTATE // max(1, rows["design"].nunique()))
            take = rows.groupby("design").head(per)
            if len(take) < N_ANNOTATE:
                take = pd.concat([take, rows.drop(take.index).head(N_ANNOTATE - len(take))])
            picked += take.head(N_ANNOTATE)[["model", "variant", "item_id"]].values.tolist()
        rng.shuffle(picked)
        sample, key = [], []
        for n, (mdl, var, iid) in enumerate(picked, 1):
            uid = f"r{n:03d}"
            sp0 = spans[R[0]["label"]]
            rows = sp0[(sp0["model"] == mdl) & (sp0["variant"] == var) & (sp0["item_id"] == iid)].sort_values("span_idx")
            sample.append({"uid": uid, "document": by_id[iid].document, "request": by_id[iid].request,
                           "response": resp[(mdl, var, iid)],
                           "spans": [{"i": int(i), "text": t} for i, t in zip(rows["span_idx"], rows["text"])]})
            k = {"uid": uid, "model": mdl, "variant": var, "item_id": iid}
            for lab, sp_ in spans.items():
                x = sp_[(sp_["model"] == mdl) & (sp_["variant"] == var) & (sp_["item_id"] == iid)].sort_values("span_idx")
                k[f"kinds_{lab}"] = x["kind"].tolist()
            key.append(k)
        ann = out / "annotation"
        ann.mkdir(exist_ok=True)
        (ann / "sample.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sample))
        (ann / "key.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in key))
        print(f"\nAnnotation sample: {len(sample)} responses, "
              f"{sum(len(x['spans']) for x in sample)} spans → {ann}")

    archive = shutil.make_archive(str(run / f"{run.name}_paper"), "zip", root_dir=out)
    print("\nSaved:", archive)
    try:
        from google.colab import files

        files.download(archive)
    except Exception:
        pass


PKG_B64 = "__PKG_B64__"

main()
