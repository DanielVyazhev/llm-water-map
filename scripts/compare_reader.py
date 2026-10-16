#!/usr/bin/env python3
"""Does a reader reproduce a run's scores? The main reader on another machine, in another dtype or from another
copy of its weights (a local folder) against the scores of the development runs.

    python scripts/compare_reader.py RUN [RUN ...] --reader MODEL [--dtype fp16 bf16] [--n 40] [--out DIR]

Draws --n responses of each run (seed 0), scores them again with --reader in each dtype, with the settings of a run
(loo, ablation at 2 nats), and compares with the run's own scores (scored.jsonl): the status (a filler map or none),
the information read per requested fact (nats; delivered at ≥ 0.5), the kept fragments and the removable share W.
Writes DIR/summary.md, summary.json and rows.csv; rerunning reuses the scores already made.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DELIVERED = 0.5  # nats: a fact the response delivers (drybench.filler)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(row: dict) -> tuple:
    return row["model"], row.get("variant", "default"), row["item_id"]


def sample(run: Path, n: int, seed: int = 0) -> list[dict]:
    """n responses of the run that its reader scored, the same ones every time."""
    scored = {key(r) for r in read_jsonl(run / "scored.jsonl")}
    resp = [r for r in read_jsonl(run / "responses.jsonl") if key(r) in scored]
    resp.sort(key=key)
    return random.Random(seed).sample(resp, min(n, len(resp)))


def score(run: Path, responses: Path, out: Path, reader: str, dtype: str, backend: str, batch_tokens: int) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO)] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    if Path(reader).expanduser().is_dir():
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cmd = [sys.executable, "-m", "drybench.cli", "score", "--items", run / "items.jsonl", "--responses", responses,
           "--out", out, "--reader-backend", backend, "--reader", reader, "--reader-dtype", dtype,
           "--reader-max-batch-tokens", batch_tokens, "--mode", "loo", "--ablation-abs-tol", 2.0]
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    if p.returncode:
        raise SystemExit(f"scoring failed ({run.name}, {dtype}):\n" + (p.stderr or p.stdout)[-2000:])


def compare(old: dict, new: dict) -> dict:
    """One response: the run's score against the new one."""
    a, b = old.get("per_probe") or [], new.get("per_probe") or []
    same_facts = bool(a) and len(a) == len(b)
    d_info = [abs(x - y) for x, y in zip(a, b)] if same_facts else []
    row = {"status_old": old["status"], "status_new": new["status"], "same_status": old["status"] == new["status"],
           "max_d_info": max(d_info) if d_info else None,
           "same_delivered": (all((x >= DELIVERED) == (y >= DELIVERED) for x, y in zip(a, b))
                              if same_facts else None),
           "W_old": old.get("W"), "W_new": new.get("W")}
    if old["status"] == new["status"] == "ok":
        row["same_kept"] = [s["kept"] for s in old["spans"]] == [s["kept"] for s in new["spans"]]
        row["d_W"] = abs(old["W"] - new["W"])
    return row


def _q(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def summarize(rows: list[dict]) -> dict:
    both = [r for r in rows if "same_kept" in r]
    info = [r["max_d_info"] for r in rows if r["max_d_info"] is not None]
    deliv = [r["same_delivered"] for r in rows if r["same_delivered"] is not None]
    return {"responses": len(rows), "same_status": sum(r["same_status"] for r in rows),
            "with_map_in_both": len(both), "same_kept": sum(r["same_kept"] for r in both),
            "mean_d_W": sum(r["d_W"] for r in both) / len(both) if both else None,
            "max_d_W": max((r["d_W"] for r in both), default=None),
            "same_delivered": sum(deliv), "compared_delivered": len(deliv),
            "median_max_d_info": _q(info, 0.5), "p95_max_d_info": _q(info, 0.95), "max_d_info": max(info, default=None)}


def settings_diff(old: list[dict], new: list[dict]) -> list[str]:
    """Scoring settings that differ between the run and the new scores (the reader and the dtype aside)."""
    a = old[0].get("settings") or {} if old else {}
    b = new[0].get("settings") or {} if new else {}
    return sorted(k for k in set(a) | set(b) if k not in ("reader", "dtype") and a.get(k) != b.get(k))


def _fmt(v, spec: str) -> str:
    return "–" if v is None else format(v, spec)


def markdown(summary: dict, reader: str) -> str:
    lines = [f"# Reader against the runs' scores\n\nReader: {reader}. Per response: the status (filler map or none), "
             "the kept fragments, the removable share W, and per requested fact the information read (|Δ| in nats; "
             f"delivered at ≥ {DELIVERED} nats).\n",
             "| run | dtype | responses | same status | same kept fragments (both with a map) | mean / max |ΔW| | "
             "same delivered facts | per-fact |Δ| nats: median / 95% / max | settings differ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for run, by_dtype in summary.items():
        for dtype, s in by_dtype.items():
            lines.append(f"| {run} | {dtype} | {s['responses']} | {s['same_status']} | {s['same_kept']} of "
                         f"{s['with_map_in_both']} | {_fmt(s['mean_d_W'], '.3f')} / {_fmt(s['max_d_W'], '.3f')} | "
                         f"{s['same_delivered']} of {s['compared_delivered']} | {_fmt(s['median_max_d_info'], '.3f')} "
                         f"/ {_fmt(s['p95_max_d_info'], '.3f')} / {_fmt(s['max_d_info'], '.3f')} | "
                         f"{', '.join(s['settings_differ']) or 'none'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders with items.jsonl, responses.jsonl and scored.jsonl")
    ap.add_argument("--reader", required=True, help="a Hugging Face id or a local model folder")
    ap.add_argument("--dtype", nargs="+", default=["fp16"])
    ap.add_argument("--n", type=int, default=40, help="responses per run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reader_equivalence")
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    reader = os.path.expanduser(args.reader)
    summary: dict = {}
    table: list[dict] = []
    for run in map(Path, args.runs):
        run = run.expanduser()
        dest = out / run.name
        dest.mkdir(parents=True, exist_ok=True)
        resp = dest / "responses.jsonl"
        chosen = sample(run, args.n, args.seed)
        resp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
        old = {key(r): r for r in read_jsonl(run / "scored.jsonl")}
        for dtype in args.dtype:
            new_path = dest / f"scored_{dtype}.jsonl"
            print(f"{run.name}: {len(chosen)} responses, reader {reader}, {dtype}", flush=True)
            score(run, resp, new_path, reader, dtype, args.backend, args.max_batch_tokens)
            new = read_jsonl(new_path)
            rows = []
            for r in new:
                if key(r) in old:
                    rows.append({"run": run.name, "dtype": dtype, "model": r["model"], "variant": r.get("variant"),
                                 "item_id": r["item_id"], **compare(old[key(r)], r)})
            s = summarize(rows)
            s["settings_differ"] = settings_diff([old[key(r)] for r in new if key(r) in old], new)
            summary.setdefault(run.name, {})[dtype] = s
            table += rows
    (out / "summary.json").write_text(json.dumps({"reader": reader, "runs": summary}, indent=1))
    if table:
        with open(out / "rows.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
            w.writeheader()
            w.writerows(table)
    text = markdown(summary, reader)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"written: {out / 'summary.md'}, {out / 'rows.csv'}")


if __name__ == "__main__":
    main()
