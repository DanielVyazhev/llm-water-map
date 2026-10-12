#!/usr/bin/env python3
"""Chat LLMs (GPT, Claude, ...) as extra annotators of the blind annotation sample, by the same guidelines
as the human annotators (drybench/guidelines.py).

  python scripts/chat_annotators.py make    annotation/sample.jsonl --out chat/ [--parts 3]
      → chat/prompt.md and chat/part_1of3.json … (paste the prompt into a new chat, attach one part per message)
  python scripts/chat_annotators.py collect annotation/sample.jsonl --model gpt chat/gpt_*.txt --out chat/gpt.jsonl
      → one labels row per response (the schema `drybench annotation-report --labels` reads), and what is missing

Chat interfaces do not fix the model version or sampling: for the paper, say so, or use an API with a fixed
version. The reproducible judge baseline is `drybench baselines --method judge` (an open model, the same
guidelines, label probabilities instead of free text).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from drybench.guidelines import EXAMPLES, GUIDE, LABELS, guidelines_text  # noqa: E402


def prompt() -> str:
    ex = EXAMPLES[0]
    frags = [{"i": k, "text": t} for k, (t, _, _) in enumerate(ex["spans"])]
    inp = {"uid": "example", "document": "\n".join(ex["document"]), "request": ex["request"], "fragments": frags}
    outp = {"uid": "example", "labels": [{"i": k, "label": lab} for k, (_, lab, _) in enumerate(ex["spans"])]}
    why = " ".join(f"Fragment {k}: {w}." for k, (_, lab, w) in enumerate(ex["spans"]) if lab != "needed")
    return f"""You are one of several independent annotators in a research study on unnecessary text ("filler") in chatbot answers. Label the fragments of every response in the attached JSON file.

INPUT. The file is a JSON array. Each element is one response:
- "uid": id of the response
- "document": the source text the chatbot was asked about; it is the only source of truth
- "request": what the user asked about the source text
- "fragments": the response split into fragments (sentences or list lines), in order; "i" is the fragment's index

{GUIDE["intro"]}

{guidelines_text()}

You may open the file however you need, but decide every label yourself by reading the fragment, the request and the source text: do not assign labels with code, scripts or keyword rules.

EXAMPLE
Input element:
{json.dumps(inp, ensure_ascii=False)}
Output element:
{json.dumps(outp, ensure_ascii=False)}
({why})

OUTPUT. Reply with a JSON array only (no text before or after it, no code fences), one object per response, in the input order:
[{{"uid": "...", "labels": [{{"i": 0, "label": "..."}}, ...]}}, ...]
Include every "uid" of the file exactly once and every fragment "i" of each response; "label" is one of {", ".join(f'"{lab}"' for lab in LABELS)}. Optional keys per fragment: "also_needed": true (only with "wrong") and "unsure": true. If the answer does not fit in one message, stop after a complete response object; I will write "continue" and you go on with the next response.
"""


def read_sample(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if not {"uid", "document", "request", "spans"} <= set(r):
            raise SystemExit(f"{path}: every line needs uid, document, request, spans (is this key.jsonl? use sample.jsonl)")
    return rows


def cmd_make(args: argparse.Namespace) -> None:
    rows = read_sample(Path(args.sample))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.md").write_text(prompt(), encoding="utf-8")
    items = [{"uid": r["uid"], "document": r["document"], "request": r["request"],
              "fragments": [{"i": int(s["i"]), "text": s["text"]} for s in r["spans"]]} for r in rows]
    n = max(1, args.parts)
    size = -(-len(items) // n)
    for k in range(n):
        part = items[k * size:(k + 1) * size]
        if part:
            (out / f"part_{k + 1}of{n}.json").write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"part_{k + 1}of{n}.json: {len(part)} responses, {sum(len(x['fragments']) for x in part)} fragments")
    print(f"prompt: {out / 'prompt.md'}")


def _objects(text: str) -> list[dict]:
    """Every {"uid": ..., "labels": [...]} object in a pasted reply (fences, prose and splits tolerated)."""
    text = re.sub(r"```(?:json)?", "", text)
    dec, out, pos = json.JSONDecoder(), [], 0
    start = re.compile(r"\{\s*\"uid\"")
    while True:
        m = start.search(text, pos)
        if not m:
            return out
        try:
            obj, end = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            pos = m.end()
            continue
        if isinstance(obj, dict) and "labels" in obj:
            out.append(obj)
        pos = end


def cmd_collect(args: argparse.Namespace) -> None:
    rows = {r["uid"]: r for r in read_sample(Path(args.sample))}
    got: dict[str, dict] = {}
    for f in args.replies:
        for obj in _objects(Path(f).read_text(encoding="utf-8")):
            if obj.get("uid") in rows:
                got[obj["uid"]] = obj  # a later reply for the same uid wins
    records, problems = [], []
    for uid, r in rows.items():
        idx = [int(s["i"]) for s in r["spans"]]
        obj = got.get(uid)
        if obj is None:
            problems.append(f"{uid}: missing")
            continue
        lab = {}
        for e in obj.get("labels", []):
            try:
                lab[int(e["i"])] = e
            except (KeyError, TypeError, ValueError):
                continue
        labels = [lab.get(i, {}).get("label") if lab.get(i, {}).get("label") in LABELS else None for i in idx]
        if any(v is None for v in labels):
            problems.append(f"{uid}: {sum(v is None for v in labels)} of {len(idx)} fragments without a valid label")
        records.append({"uid": uid, "annotator": args.model, "span_i": idx, "labels": labels,
                        "also_needed": [i for i in idx if lab.get(i, {}).get("also_needed") is True
                                        and lab.get(i, {}).get("label") == "wrong"],
                        "unsure": [i for i in idx if lab.get(i, {}).get("unsure") is True],
                        "complete": all(v is not None for v in labels)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    done = sum(x["complete"] for x in records)
    print(f"{args.model}: {done}/{len(rows)} responses fully labelled → {args.out}")
    for p in problems[:30]:
        print("  ", p)
    if len(problems) > 30:
        print(f"   … and {len(problems) - 30} more")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="prompt and input parts from annotation/sample.jsonl")
    m.add_argument("sample")
    m.add_argument("--out", default="chat")
    m.add_argument("--parts", type=int, default=3)
    m.set_defaults(fn=cmd_make)
    c = sub.add_parser("collect", help="parse pasted replies into labels")
    c.add_argument("sample")
    c.add_argument("replies", nargs="+", help="text files with the model's replies (any number, any order)")
    c.add_argument("--model", required=True, help="annotator name, e.g. gpt or claude")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
