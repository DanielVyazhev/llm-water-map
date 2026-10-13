# Filler in LLM answers — details (CPU only, ~30 s). Run in a new cell after the main cell has finished;
# at the end it zips the run folder and downloads it.
import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path

RUN = "auto"  # the main cell's folder (Google Drive or /content), or a path


def find_run(run, name="t4_n20_v12"):
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


RUN = find_run(RUN)
print("Run:", RUN)
R = RUN / "report"
it = pd.read_csv(R / "item_table.csv")
sp = pd.read_csv(R / "span_table.csv")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
it["group"] = it["model"] + " | " + it["variant"]
sp["group"] = sp["model"] + " | " + sp["variant"]
sc = it[it["status"] == "ok"].copy()
rng = np.random.default_rng(0)


def ci(x, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return "n/a"
    b = rng.choice(x, (n, len(x))).mean(axis=1)
    return f"{x.mean():+.3f} [{np.quantile(b, 0.025):+.3f}, {np.quantile(b, 0.975):+.3f}]"


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


first = json.loads(open(RUN / "scored.jsonl").readline())
ab = first.get("ablation") or {}
alt = f"tolerance {ab.get('abs_tol')} nats" if ab.get("abs_tol") not in (None, first.get("abs_tol")) \
    else f"δ = {ab.get('delta')}"
print(f"1) Filler under two tolerances: {first.get('abs_tol')} nats (default) vs {alt}")
rows = {}
for g, d in sc.groupby("group"):
    rows[g] = {"W": d["W"].mean(), "W (alternative)": d["W_ablation"].mean(),
               "responses changed": ((d["W_ablation"] - d["W"]).abs() > 1e-9).mean()}
print(pd.DataFrame(rows).T.round(3).to_string())
piv = it.pivot_table(index=["model", "item_id"], columns="variant", values=["W", "W_ablation"],
                     aggfunc="first", dropna=False)
print("   H3 ΔW (concise − default) under each tolerance:")
for m, g in piv.groupby(level=0):
    print(f"   {m}: default {ci(g[('W', 'concise')] - g[('W', 'default')])}; "
          f"alternative {ci(g[('W_ablation', 'concise')] - g[('W_ablation', 'default')])}")
wc = RUN / "filler_check" / "summary.json"
if wc.exists():
    ws = json.loads(wc.read_text())
    print("   simulated filler left in (share of spans of each type) by tolerance:")
    kept = pd.DataFrame({name: ws[k]["kept_share_by_label"]
                         for k, name in (("loo", "0.5 nats"), ("loo_t1", "1 nat"), ("loo_t2", "2 nats"))
                         if k in ws})
    print(kept.round(2).to_string())

print("\n2) Where the filler is, by item design (mean W; W_wrong below):")


def design(r):
    if not r["answerable"]:
        return "missing fact"
    if r["ambiguous"]:
        return "ambiguous"
    if r["hops"] == 2:
        return "2-hop+explain" if r["explain"] else "2-hop"
    return "k=1" if r["k"] == 1 else ("k=2-3" if r["k"] <= 3 else "k=4-6")


sc["design"] = sc.apply(design, axis=1)
order = ["k=1", "k=2-3", "k=4-6", "missing fact", "ambiguous", "2-hop", "2-hop+explain"]
for col in ("W", "W_misleading"):
    t = sc.pivot_table(index="group", columns="design", values=col, aggfunc="mean")
    if not t.empty:
        print(t.reindex(columns=[c for c in order if c in t.columns]).round(3).to_string(), "\n")

print("3) Does the generating model's own span entropy flag filler? AUC of h_span, filler vs kept spans")
print("   (0.5 = no signal; 'within' compares spans of the same response; length AUC is a baseline):")
sp["filler"] = sp["kind"].isin(["uninformative", "duplicate", "removed"])
res = {}
for g, d in sp[sp["kind"] != "misleading"].groupby("group"):
    within = [auc(r.loc[r["filler"], "h_span"], r.loc[~r["filler"], "h_span"])
              for _, r in d.groupby("item_id") if r["filler"].any() and (~r["filler"]).any()]
    res[g] = {"filler spans": int(d["filler"].sum()), "AUC h_span": auc(d.loc[d["filler"], "h_span"],
                                                                       d.loc[~d["filler"], "h_span"]),
              "AUC h_span within": float(np.nanmean(within)) if within else float("nan"),
              "responses": len(within),
              "AUC length": auc(d.loc[d["filler"], "tokens"], d.loc[~d["filler"], "tokens"])}
print(pd.DataFrame(res).T.round(3).to_string())

print("\n4) Completeness check: requested facts the reader could read from the gold answer —")
print("   'decided' = reader answers correctly (p ≥ 0.5); 'near-full info, undecided' = ≥ 90% of the gold's")
print("   information but p < 0.5 (often a format mismatch, e.g. '23' vs '23 containers'):")
items = {j["item_id"]: j for j in map(json.loads, open(RUN / "items.jsonl"))}
resp = {(j["model"], j.get("variant", "default"), j["item_id"]): j["response"]
        for j in map(json.loads, open(RUN / "responses.jsonl"))}
recs = []
for s in map(json.loads, open(RUN / "scored.jsonl")):
    probes = [p for p in items[s["item_id"]]["probes"] if p.get("required", True)]
    for p, I, p_resp, p_gold in zip(probes, s["per_probe"], s.get("per_probe_p") or [],
                                    s.get("gold_per_probe_p") or []):
        if "not stated" in p["answers"] or p_gold < 0.5:
            continue
        f = np.log(s.get("prior_floor") or 1e-300)
        prior = max(np.log(max(p_resp, 1e-300)), f) - I  # floored log p(answer | empty message)
        I_gold = max(np.log(p_gold), f) - prior
        recs.append({"group": f"{s['model']} | {s['variant']}", "key": (s["model"], s["variant"], s["item_id"]),
                     "answer": p["answers"][0], "p": p_resp, "share_info": I / I_gold if I_gold > 0 else np.nan})
pr = pd.DataFrame(recs)
if not pr.empty:
    pr["state"] = np.select([pr["p"] >= 0.5, pr["share_info"] >= 0.9, pr["share_info"] >= 0.1],
                            ["decided", "near-full info, undecided", "partial"], "absent")
    print(pr.pivot_table(index="group", columns="state", values="p", aggfunc="size", fill_value=0)
          .pipe(lambda t: t.div(t.sum(axis=1), axis=0)).round(3).to_string())
    ex = pr[pr["state"] == "near-full info, undecided"]
    for _, r in ex.sample(min(6, len(ex)), random_state=0).iterrows():
        print(f"\n   {r['group']}: expected '{r['answer']}', p = {r['p']:.2f}")
        print("     response:", resp.get(r["key"], "")[:300].replace("\n", " / "))

print("\n5) Deleted spans by kind (count; tokens), with examples for the default prompt:")
dele = sp[sp["kind"] != "kept"]
print(pd.concat([dele.pivot_table(index="group", columns="kind", values="tokens", aggfunc="size", fill_value=0),
                 dele.pivot_table(index="group", columns="kind", values="tokens", aggfunc="sum", fill_value=0)
                 .add_suffix(" tok")], axis=1).to_string())
for kind in ("uninformative", "duplicate", "misleading"):
    for grp, g in dele[(dele["kind"] == kind) & dele["group"].str.endswith("default")].groupby("group"):
        print(f"\n  {grp} — {kind} ({len(g)} spans):")
        for t in g.sample(min(4, len(g)), random_state=0)["text"]:
            print("     ·", str(t)[:150])

print("\n6) Save this run (Colab storage is wiped when the runtime is recycled):")
archive = shutil.make_archive(str(RUN), "zip", root_dir=RUN.parent, base_dir=RUN.name)
print("   ", archive)
try:
    from google.colab import files

    files.download(archive)
except Exception:
    pass
