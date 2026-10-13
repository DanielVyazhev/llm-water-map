#!/usr/bin/env python3
"""Every reader call behind the filler map of one response, with the numbers the scores do not store.

  python scripts/trace_response.py RUN --model qwen2.5-0.5b --variant default --item c01-0017 \
      --reader Qwen/Qwen2.5-3B-Instruct --out trace.json          # GPU: a few seconds
  python scripts/trace_response.py RUN ... --reader-backend mock   # smoke test without a model

For each requested fact it prints ln p(correct answer) for: no message, the whole response, the
response without each fragment, the response without each prefix of the deletion queue, and each
deleted fragment read alone with its own wording of a fact accepted (plus the probability of a wrong
answer), then the decisions, including what keeping the first statement of a fact changed. The
decisions come from drybench's own dry_score, so they match the run when the reader and the
package version match. A response without a filler map (no_signal: the reader reads no requested fact
from it; reader_fail: it cannot read the gold answer) stops after the facts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the drybench next to this script


def build_reader(args):
    from drybench.reader import Reader, ReaderConfig

    if args.reader_backend == "mock":
        from drybench.backends.mock import MockBackend

        return Reader(MockBackend(), ReaderConfig(use_chat=True))
    from drybench.backends.hf import HFBackend

    return Reader.for_backend(HFBackend(args.reader, dtype=args.dtype))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder with items.jsonl and responses.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--item", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--reader-backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default=None, help="write the trace as JSON here")
    args = ap.parse_args(argv)

    from drybench.forms import own_wording_probes, with_own_wording
    from drybench.segment import join_spans, segment
    from drybench.types import load_items, read_jsonl
    from drybench.filler import _feasible, dry_score, misleading_excuse, wrong_answer_mass

    run = Path(args.run)
    item = {it.item_id: it for it in load_items(run / "items.jsonl")}[args.item]
    response = next(r["response"] for r in read_jsonl(run / "responses.jsonl")
                    if r["item_id"] == args.item and r["model"] == args.model and r.get("variant", "default") == args.variant)
    reader = build_reader(args)
    res = dry_score(reader, item, response, mode="loo")
    spans = segment(response)
    # v1.1: the trace reads the facts with the same accepted answers as dry_score (the response's own wording)
    item, added = with_own_wording(reader, item, join_spans(spans))
    if any(added):
        print("accepted in the response's own words: " + "; ".join(
            f"{p.question} → {', '.join(a)}" for p, a in zip(item.required_probes, added) if a))
    probes = item.required_probes
    floor = reader.log_floors(probes)  # per fact: a missing fact asked as yes/no has its own floor
    n = len(spans)

    def lp(messages):
        return reader.probe_logprobs(item.request, messages, probes)

    def info(lps):
        return np.maximum(lps, floor) - np.maximum(prior, floor)

    prior = reader.prior_logprobs(item.request, probes)
    full = lp([join_spans(spans)])[0]  # the message the filler map reads
    I_full = info(full[None])[0]
    reader_name = args.reader if args.reader_backend == "hf" else "mock"
    head = {"item": args.item, "model": args.model, "variant": args.variant, "reader": reader_name,
            "request": item.request, "status": res.status,
            "facts": [{"question": p.question, "answer": p.value, "ln_p_no_message": float(prior[j]),
                       "ln_p_full": float(full[j]), "p_full": math.exp(full[j]), "information_full": float(I_full[j]),
                       "floor_ln": float(floor[j])} for j, p in enumerate(probes)]}
    print(f"{args.model} | {args.variant} | {args.item}   reader: {reader_name}\nrequest: {item.request}")
    for f in head["facts"]:
        print(f"fact: {f['question']} → {f['answer']}\n  ln p(no message) = {f['ln_p_no_message']:.4f} "
              f"(floor {f['floor_ln']:.4f});  ln p(whole answer) = {f['ln_p_full']:.6f} (p = {f['p_full']:.6f});  "
              f"information = {f['information_full']:.4f} nats")
    if res.status != "ok":
        print(f"\nno filler map ({res.status}): " + ("the reader reads no requested fact from the response"
                                                    if res.status == "no_signal" else
                                                    "the reader cannot read the gold answer"))
        if args.out:
            Path(args.out).write_text(json.dumps(head, indent=1, ensure_ascii=False))
            print(f"written: {args.out}")
        return
    without = lp([join_spans(spans, [j != i for j in range(n)]) for i in range(n)])
    I_without = info(without)
    cost = I_full[None, :] - I_without
    candidate = _feasible(I_without, I_full, res.delta, res.abs_tol, res.constraint)
    tol = 1e-3 * abs(float(I_full.sum()))
    lengths = [max(1, reader.backend.count_tokens(s.text)) for s in spans]
    order = sorted((i for i in range(n) if candidate[i]),
                   key=lambda i: (round(cost[i].sum() / tol) if tol else cost[i].sum(), -lengths[i]))
    # the deletion search: the longest feasible prefix of the queue, then skip the span that ended it
    # and go on with the rest (each round checked jointly with everything deleted so far)
    checks, done, queue, rnd = [], [], list(order), 0
    while queue:
        rnd += 1
        prefixes = [done + queue[: r + 1] for r in range(len(queue))]
        pre_lp = lp([join_spans(spans, [j not in set(p) for j in range(n)]) for p in prefixes])
        pre_ok = _feasible(info(pre_lp), I_full, res.delta, res.abs_tol, res.constraint)
        checks += [(rnd, pfx, pre_lp[r], bool(pre_ok[r])) for r, pfx in enumerate(prefixes)]
        ok = np.flatnonzero(pre_ok)
        r_best = int(ok.max()) if len(ok) else -1
        if r_best >= 0:
            done = done + queue[: r_best + 1]
        queue = queue[r_best + 2:]
    removed = [i for i, s in enumerate(res.spans) if not s.kept]
    search = set(done)  # the queue search alone; dry_score then keeps the first statement of a fact
    keep_first = {"restored": sorted(i + 1 for i in search - set(removed)),
                  "deleted": sorted(i + 1 for i in set(removed) - search)}
    # each deleted fragment read alone with its own wording of a fact accepted, as classify_removed reads it
    alone_info, p_wrong, p_wrong_empty = [], [], []
    for i in removed:
        ps, _ = own_wording_probes(reader, item.request, probes, spans[i].text)
        alone_info.append(reader.per_probe_information(item.request, [spans[i].text], ps)[0])
        both = wrong_answer_mass(reader, item.request, [spans[i].text, ""], ps)
        p_wrong.append(both[0])
        p_wrong_empty.append(both[1])

    trace = {
        **head, "tolerance_nats": res.abs_tol, "tie_resolution_nats": tol,
        "fragments": [{"i": i + 1, "text": s.text, "tokens": lengths[i],
                       "ln_p_without": [float(x) for x in without[i]], "p_without": [math.exp(x) for x in without[i]],
                       "information_without": [float(x) for x in I_without[i]], "cost": [float(x) for x in cost[i]],
                       "candidate": bool(candidate[i]), "kept": res.spans[i].kept, "kind": res.spans[i].kind,
                       "step": None if res.spans[i].removal_rank is None else res.spans[i].removal_rank + 1}
                      for i, s in enumerate(spans)],
        "queue": [i + 1 for i in order], "keep_first": keep_first,
        "joint_checks": [{"round": rnd, "removed": [i + 1 for i in pfx], "ln_p": [float(x) for x in lps],
                          "information": [float(x) for x in info(lps[None, :])[0]], "passes": passes}
                         for rnd, pfx, lps, passes in checks],
        "read_alone": [{"i": i + 1, "information": [float(x) for x in alone_info[k]],
                        "p_wrong_answer": [float(x) for x in p_wrong[k]],
                        "p_wrong_answer_no_message": [float(x) for x in p_wrong_empty[k]],
                        "excuse": misleading_excuse(spans[i].text, item), "kind": res.spans[i].kind}
                       for k, i in enumerate(removed)],
        "result": {"tokens": res.tokens, "tokens_kept": res.tokens_kept, "W": res.W, "W_filler": res.W_filler,
                   "W_misleading": res.W_misleading, "information_kept": res.info_kept, "facts": res.facts},
    }

    print("\nremove one fragment:")
    for f in trace["fragments"]:
        print(f"  [{f['i']}] cost {', '.join(f'{c:+.5f}' for c in f['cost'])} nats  candidate={f['candidate']}  "
              f"→ {'kept' if f['kept'] else 'deleted at step ' + str(f['step']) + ' as ' + f['kind']} | {f['text'][:80]}")
    print(f"\nqueue: {trace['queue']}")
    for c in trace["joint_checks"]:
        print(f"  round {c['round']}, remove {c['removed']}: information "
              f"{', '.join(f'{x:.4f}' for x in c['information'])} → {'passes' if c['passes'] else 'fails'}")
    if keep_first["restored"] or keep_first["deleted"]:
        print(f"keeping the first statement of a fact: restored {keep_first['restored']}, "
              f"deleted {keep_first['deleted']}")
    print("\ndeleted fragments read alone:")
    for a in trace["read_alone"]:
        print(f"  [{a['i']}] information alone {', '.join(f'{x:.3f}' for x in a['information'])}; p(wrong answer) "
              f"{', '.join(f'{x:.3f}' for x in a['p_wrong_answer'])} (no message: "
              f"{', '.join(f'{x:.3f}' for x in a['p_wrong_answer_no_message'])}); excuse: {a['excuse']} → {a['kind']}")
    r = trace["result"]
    print(f"\ntokens {r['tokens']} → {r['tokens_kept']}; removable {r['W']:.3f}; information kept {r['information_kept']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(trace, indent=1, ensure_ascii=False))
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
