# Missing-fact answers: what each reader's filler map did with the fragments that say the fact is not
# stated, and κ with a contradicted fact counted as 0 (v0.7.4). CPU, a few seconds, no package needed.
# Works on any run folder: the real-text run on Google Drive, or a synthetic run restored from its zip.
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("/content/drive/MyDrive/drybench_runs/real_t4_n15")  # synthetic: Path("/content/drybench_runs/t4_n20")
if str(RUN).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
    from google.colab import drive

    drive.mount("/content/drive")
ABSTAIN = re.compile(
    r"\b(?:not (?:stated|specified|mentioned|provided|given|available|included|listed|indicated|disclosed|found|"
    r"known|clear)|no (?:information|mention|details?|data)|(?:does|do|did|is|are|was|were) not (?:say|state|mention|"
    r"specify|provide|include|contain|indicate|give)|(?:doesn't|don't|didn't|isn't|aren't) (?:say|state|mention|"
    r"specify|provide|include|contain|indicate|give)|unknown|cannot be determined|can't be determined|"
    r"unable to determine)\b", re.I)
KINDS = ("kept", "uninformative", "removed", "duplicate", "misleading")

items = {j["item_id"]: j for j in map(json.loads, open(RUN / "items.jsonl"))}
missing = {k: v["knobs"].get("source", "synthetic") for k, v in items.items()
           if v["knobs"].get("answerable", True) is False}
kappa_rows = []
for sub in [RUN] + sorted(p for p in RUN.glob("reader2_*") if (p / "scored.jsonl").exists()):
    label = "main" if sub == RUN else sub.name.replace("reader2_", "")
    rows = [json.loads(line) for line in open(sub / "scored.jsonl")]
    stats = defaultdict(lambda: {"kinds": Counter(), "answers": 0, "none_kept": 0, "costs": [], "examples": []})
    for r in rows:
        if r["item_id"] not in missing or r["status"] != "ok":
            continue
        ab = [sp for sp in r["spans"] if ABSTAIN.search(sp["text"])]
        if not ab:
            continue
        st = stats[missing[r["item_id"]]]
        st["answers"] += 1
        st["none_kept"] += not any(sp["kept"] for sp in ab)
        for sp in ab:
            kind = sp.get("kind") or ("kept" if sp["kept"] else "removed")
            st["kinds"][kind] += 1
            if kind in ("uninformative", "removed"):
                st["costs"].append(sp.get("loo_cost", np.nan))
                if len(st["examples"]) < 2:
                    q = next((p["question"] for p in items[r["item_id"]]["probes"] if p.get("value") == "not stated"), "")
                    left = " ".join(" ".join(s["text"].split()) for s in r["spans"] if s["kept"])
                    st["examples"].append(f"      [{r['model']} | {r.get('variant', 'default')}] {q}\n"
                                          f"        deleted: «{' '.join(sp['text'].split())}»\n"
                                          f"        left: «{left[:240]}»")
    print(f"\n=== reader {label}: fragments that say the fact is not stated, in missing-fact answers")
    for source, st in sorted(stats.items()):
        n = sum(st["kinds"].values())
        shares = ", ".join(f"{k} {st['kinds'][k] / n:.2f}" for k in KINDS if st["kinds"][k])
        cost = f"; median cost to delete alone {np.nanmedian(st['costs']):.2f} nats" if st["costs"] else ""
        print(f"  {source}: {n} fragments in {st['answers']} answers — {shares}; answers left without one: "
              f"{st['none_kept']}/{st['answers']}{cost}")
        if st["examples"]:
            print("\n".join(st["examples"]))
    for r in rows:
        if r["status"] in ("ok", "no_signal") and r.get("info_gold") and r.get("per_probe"):
            kappa_rows.append({"reader": label, "group": f"{r['model']} | {r.get('variant', 'default')}",
                               "κ as stored": r.get("kappa"),
                               "κ, contradicted fact = 0": sum(max(x, 0.0) for x in r["per_probe"]) / r["info_gold"]})
k = pd.DataFrame(kappa_rows).groupby(["group", "reader"])[["κ as stored", "κ, contradicted fact = 0"]].mean().unstack()
print("\nκ (mean over the answers the reader can judge, all suites):")
print(k.round(3).to_string())
