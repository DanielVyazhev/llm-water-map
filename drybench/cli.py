"""Command-line pipeline.

  drybench items     --out items.jsonl [--n-per-cell 30] [--seed 0]
  drybench generate  --items items.jsonl --backend hf --model Qwen/Qwen2.5-7B-Instruct --out resp.jsonl
  drybench score     --items items.jsonl --responses resp.jsonl --reader-backend hf \
                     --reader Qwen/Qwen2.5-7B-Instruct --out scored.jsonl
  drybench signals   --items items.jsonl --responses resp.jsonl --backend hf --model <same model> \
                     --model-name <label used in generate> --out signals.jsonl [--sampled 10]
  drybench analyze   --items items.jsonl --scored scored.jsonl --signals signals.jsonl --out-dir report/
  drybench check-reader --items items.jsonl --reader-backend hf --reader <model>
  drybench missing-check --run RUN_DIR [--synthetic-n 20] --reader-backend hf --reader <model>

Real texts (SQuAD 2.0, MuSiQue, CUAD, Poddar et al.): build an oversampled pool, then keep n items
per cell whose facts the reader cannot answer without a response and can read from the gold answer:
  drybench real-items   --source squad2 --path dev-v2.0.json --n-per-cell 90 --out pool.jsonl
  drybench real-items   --source musique --path musique_full_v1.0_dev.jsonl --n-per-cell 90 --out pool2.jsonl
  drybench filter-items --items pool.jsonl pool2.jsonl --n-per-cell 30 --reader-backend hf \
                        --reader <model> --out items.jsonl --report filter_report.json
Checks on real texts (truth known):
  drybench real-filler-check --items items.jsonl --out-dir filler_check_real --reader-backend hf --reader <model>
  drybench poddar-check     --data data.jsonl --out-dir poddar_check --reader-backend hf --reader <model>
Baselines on a check's responses (or on the annotation sample), written to RUN/baselines/<method>-<label>/:
  drybench baselines --run RUN --set planted_real human_poddar --method judge --label phi-4 --backend hf --model microsoft/phi-4
  drybench baselines --run RUN --set planted_real --method selective --label qwen2.5-3b --backend hf --model Qwen/Qwen2.5-3B-Instruct
Human labels of the benchmark's own answers:
  drybench annotation-sample --runs RUN_SYN RUN_REAL --out annotation/      # sample.jsonl + key.jsonl
  drybench abstain-sample    --runs RUN_SYN RUN_REAL --out annotation/      # abstain_sample.csv + key
  drybench annotation-report --sample annotation/ --labels span_labels_*.json [--abstain-csv filled.csv]

All writers append and skip rows already present, so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from drybench.backends import make_backend
from drybench.baselines import judge_filler
from drybench.entropy import answer_entropy_first_token, answer_entropy_sampled, response_span_entropies
from drybench.items import evaluated_model_messages, generate_items
from drybench.reader import Reader
from drybench.segment import segment
from drybench.types import load_items, read_jsonl, write_jsonl
from drybench.filler import dry_score


def _backend_args(p: argparse.ArgumentParser, prefix: str = "") -> None:
    dash = f"--{prefix}-" if prefix else "--"
    name = prefix or "model"
    p.add_argument(f"{dash}backend", default="hf", choices=["hf", "onnx-gpt2", "mock"])
    p.add_argument(f"--{name}", default=None, help="model id/path (hf) or assets dir (onnx-gpt2)")
    p.add_argument(f"{dash}device", default=None)
    p.add_argument(f"{dash}dtype", default="auto")
    p.add_argument(f"{dash}max-batch-tokens", type=int, default=8192)
    p.add_argument(f"{dash}chat-template-kwargs", default=None,
                   help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": false}\'')


def _make(args: argparse.Namespace, prefix: str = ""):
    key = (prefix + "_") if prefix else ""
    kind = getattr(args, f"{key}backend")
    path = getattr(args, prefix or "model")
    kw = {}
    if kind == "hf":
        ctk = getattr(args, f"{key}chat_template_kwargs")
        kw = dict(device=getattr(args, f"{key}device"), dtype=getattr(args, f"{key}dtype"),
                  max_batch_tokens=getattr(args, f"{key}max_batch_tokens"),
                  chat_template_kwargs=json.loads(ctk) if ctk else None)
    elif kind == "onnx-gpt2":
        kw = dict(max_batch_tokens=min(getattr(args, f"{key}max_batch_tokens"), 4096))
    return make_backend(kind, path, **kw)


def _device_name(backend) -> str:
    """The GPU a backend runs on (stored with every timing)."""
    dev = str(getattr(backend, "device", "") or "")
    if dev.startswith("cuda"):
        try:
            import torch

            return torch.cuda.get_device_name(torch.device(dev))
        except Exception:  # noqa: BLE001 - only a label
            return dev
    return dev or type(backend).__name__.replace("Backend", "").lower()


def _done_keys(path: str, fields: tuple[str, ...]) -> set[tuple]:
    if not Path(path).exists():
        return set()
    return {tuple(r.get(f, "default") for f in fields) for r in read_jsonl(path)}


def _append(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_items(args: argparse.Namespace) -> None:
    if args.update_probes:  # keep the items, take accepted answers from the current generator
        from drybench.items import refresh_probes

        items = refresh_probes(load_items(args.out), seed=args.seed)
        n = write_jsonl(args.out, (it.to_dict() for it in items))
        print(f"updated answer forms of {n} items in {args.out}")
        return
    items = generate_items(args.n_per_cell, seed=args.seed)
    n = write_jsonl(args.out, (it.to_dict() for it in items))
    print(f"wrote {n} items to {args.out}")


def cmd_real_items(args: argparse.Namespace) -> None:
    from drybench.realdata import build_real_items

    items, stats = build_real_items(args.source, args.path, args.n_per_cell, seed=args.seed,
                                    distractors=args.distractors)
    n = write_jsonl(args.out, (it.to_dict() for it in items))
    print(f"wrote {n} {args.source} items to {args.out}")
    for key, st in stats.items():
        short = "" if st["built"] >= st["wanted"] else "  <- fewer than asked"
        print(f"  {key} {json.dumps(st['cell'])}: {st['built']}/{st['wanted']}{short}")


def cmd_filter_items(args: argparse.Namespace) -> None:
    from drybench.realdata import filter_items

    from drybench import SCORING_VERSION

    pool = [it for path in args.items for it in load_items(path)]
    if args.exclude_items:  # e.g. the items of the development runs: a held-out confirmation run
        from drybench.realdata import source_key

        used = {source_key(it) for path in args.exclude_items for it in load_items(path)}
        n0 = len(pool)
        pool = [it for it in pool if source_key(it) not in used]
        print(f"left out {n0 - len(pool)} pool items whose source text (paragraph, question set, contract, "
              f"labelled question) is used by {', '.join(args.exclude_items)}")
    keep = load_items(args.keep) if args.keep and Path(args.keep).exists() else []
    reader = Reader.for_backend(_make(args, "reader"))
    kept, stats = filter_items(reader, pool, n_per_cell=args.n_per_cell, max_prior=args.max_prior,
                               min_gold_p=args.min_gold_p, keep=keep)
    n = write_jsonl(args.out, (it.to_dict() for it in kept))
    print(f"kept {n} of {len(pool)} pool items -> {args.out}")
    for key, st in stats.items():
        short = "" if args.n_per_cell is None or st["kept"] >= args.n_per_cell else "  <- fewer than asked"
        prev = f"{st['kept_before']} from before + " if st.get("kept_before") else ""
        print(f"  {key}: kept {st['kept']} ({prev}checked {st['checked']} of {st['pool']}: reader knew a fact in "
              f"{st['known']}, could not read the gold answer in {st['unreadable']}){short}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"reader": args.reader, "max_prior": args.max_prior,
                                                 "min_gold_p": args.min_gold_p, "n_per_cell": args.n_per_cell,
                                                 "scoring": SCORING_VERSION, "cells": stats}, indent=1))


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def _generate_safe(backend, ctxs: list[str], args: argparse.Namespace) -> list[str]:
    """Generate a batch; on a CUDA out-of-memory error, split it in two and retry (long documents).

    The retry runs after the except block has exited: while it runs, the exception's traceback
    keeps the failed call's frames (and their KV cache) alive, so nothing could be freed."""
    gen_batch = getattr(backend, "generate_batch", None)
    try:
        if gen_batch:
            return gen_batch(ctxs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed)
        return [backend.generate(c, args.max_new_tokens, args.temperature, seed=args.seed) for c in ctxs]
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
        if not _is_oom(e) or len(ctxs) == 1:
            raise
    import gc

    gc.collect()
    torch = getattr(backend, "torch", None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    half = len(ctxs) // 2
    print(f"out of memory with a batch of {len(ctxs)}: retrying as {half} + {len(ctxs) - half}", file=sys.stderr)
    return _generate_safe(backend, ctxs[:half], args) + _generate_safe(backend, ctxs[half:], args)


def cmd_generate(args: argparse.Namespace) -> None:
    items = load_items(args.items)
    done = _done_keys(args.out, ("model", "variant", "item_id"))
    label = args.model_name or args.model
    todo = [(v, it) for v in args.variant for it in items if (label, v, it.item_id) not in done]
    if not todo:
        print(f"generate: nothing to do for {label}", file=sys.stderr)
        return
    backend = _make(args)
    ctx = {(v, it.item_id): backend.format_chat(evaluated_model_messages(it, v)) for v, it in todo}
    todo.sort(key=lambda t: len(ctx[(t[0], t[1].item_id)]))  # similar lengths per batch: less padding
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start: start + args.batch_size]
        outs = _generate_safe(backend, [ctx[(v, it.item_id)] for v, it in chunk], args)
        for (v, it), text in zip(chunk, outs):
            _append(args.out, {"item_id": it.item_id, "model": label, "variant": v,
                               "response": text.strip()})
        print(f"generated {min(start + args.batch_size, len(todo))}/{len(todo)} [{label}]", file=sys.stderr)


def cmd_score(args: argparse.Namespace) -> None:
    items = {it.item_id: it for it in load_items(args.items)}
    done = _done_keys(args.out, ("model", "variant", "item_id"))
    rows = [r for r in read_jsonl(args.responses)
            if (r["model"], r.get("variant", "default"), r["item_id"]) not in done]
    if not rows:
        print("score: nothing to do", file=sys.stderr)
        return
    reader = Reader.for_backend(_make(args, "reader"))
    device = _device_name(reader.backend)
    for n, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        res = dry_score(reader, items[r["item_id"]], r["response"], delta=args.delta,
                        mode=args.mode, segment_mode=args.segment, constraint=args.constraint,
                        abs_tol=args.abs_tol)
        row = {"item_id": r["item_id"], "model": r["model"], "variant": r.get("variant", "default"),
               **res.to_dict(), "seconds": round(time.perf_counter() - t0, 4), "device": device}
        if args.ablation_delta is not None or args.ablation_abs_tol is not None:
            # a second tolerance on the same response; cheap, the reader cache holds most messages
            d = args.delta if args.ablation_delta is None else args.ablation_delta
            t = args.abs_tol if args.ablation_abs_tol is None else args.ablation_abs_tol
            t0 = time.perf_counter()
            ab = dry_score(reader, items[r["item_id"]], r["response"], delta=d, mode=args.mode,
                           segment_mode=args.segment, constraint=args.constraint, abs_tol=t,
                           with_kappa=False)
            row["ablation"] = {"delta": d, "abs_tol": t, "W": ab.W, "W_filler": ab.W_filler,
                               "W_misleading": ab.W_misleading, "tokens_kept": ab.tokens_kept,
                               "seconds": round(time.perf_counter() - t0, 4)}
        if args.judge:
            row["judge_W"] = judge_filler(reader.backend, items[r["item_id"]].request, r["response"])
        _append(args.out, row)
        if n % 10 == 0:
            print(f"scored {n}/{len(rows)}", file=sys.stderr)


def cmd_signals(args: argparse.Namespace) -> None:
    items = {it.item_id: it for it in load_items(args.items)}
    label = args.model_name or args.model
    done = _done_keys(args.out, ("model", "variant", "item_id"))
    rows = [r for r in read_jsonl(args.responses)
            if r["model"] == label and (label, r.get("variant", "default"), r["item_id"]) not in done]
    if not rows:
        print(f"signals: nothing to do for {label!r}", file=sys.stderr)
        return
    backend = _make(args)
    h_first_cache: dict[str, list[float]] = {}  # H_ans does not depend on the prompt variant
    for n, r in enumerate(rows, 1):
        it = items[r["item_id"]]
        variant = r.get("variant", "default")
        ctx = backend.format_chat(evaluated_model_messages(it, variant))
        spans = segment(r["response"], args.segment)
        h_span = response_span_entropies(backend, ctx, r["response"], spans) if spans else []
        if it.item_id not in h_first_cache:
            h_first_cache[it.item_id] = answer_entropy_first_token(backend, it)
        row = {"item_id": it.item_id, "model": label, "variant": variant,
               "H_first": h_first_cache[it.item_id],
               "h_span": [None if h != h else float(h) for h in h_span]}
        if args.sampled:
            row["H_sampled"] = answer_entropy_sampled(backend, it, n=args.sampled)
        _append(args.out, row)
        if n % 10 == 0:
            print(f"signals {n}/{len(rows)}", file=sys.stderr)


def cmd_check_reader(args: argparse.Namespace) -> None:
    from drybench.validate import check_reader

    reader = Reader.for_backend(_make(args, "reader"))
    res = check_reader(reader, load_items(args.items), n=args.n)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_missing_check(args: argparse.Namespace) -> None:
    from drybench.reader import ReaderConfig
    from drybench.validate import format_missing_check, missing_cases, missing_question_check

    run = Path(args.run) if args.run else None
    items = load_items(run / "items.jsonl") if run else []
    scored = list(read_jsonl(run / "scored.jsonl")) if run and (run / "scored.jsonl").exists() else []
    if args.synthetic_n:
        items = items + [it for it in generate_items(n_per_cell=args.synthetic_n, seed=0)
                         if not it.knobs.get("answerable", True)]
    cases = missing_cases(items, scored, max_per_kind=args.max_per_kind)
    backend = _make(args, "reader")
    reader = Reader(backend, ReaderConfig(use_chat=True)) if args.chat else Reader.for_backend(backend)
    res = missing_question_check(reader, cases, args.styles, show=args.show)
    print(format_missing_check(res))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_prune_scores(args: argparse.Namespace) -> None:
    from drybench import SCORING_VERSION
    from drybench.rescore import plan, prune

    new = args.to or SCORING_VERSION
    what = plan(args.old, new, args.reader)
    changes = prune(args.dir, args.items, what)
    if args.json:
        print(json.dumps({"plan": what, "changes": changes}))
        return
    print(f"{args.dir}: scoring v{args.old} → v{new}, rescore {what}")
    for line in changes:
        print("  " + line)


def cmd_real_filler_check(args: argparse.Namespace) -> None:
    from drybench.realcheck import planted_check

    items = [it for path in args.items for it in load_items(path)]
    reader = Reader.for_backend(_make(args, "reader"))
    s = planted_check(reader, items, args.out_dir, n_per_cell=args.n_per_cell)
    o = s.get("overall") or {}
    if o:
        lo, hi = o["recall_pooled_ci"]
        print(f"{s['n_scored_ok']} of {s['n_responses']} simulated responses scored; no fact lost in "
              f"{100 * o['no_fact_lost']:.1f}%; precision {o['precision_pooled']:.3f}; planted filler deleted "
              f"{o['recall_pooled']:.3f} [{lo:.3f}, {hi:.3f}]; wrong statements deleted "
              f"{s['wrong']['deleted']:.2f}, labelled wrong {s['wrong']['labelled_wrong']:.2f}")
    print(f"written: {Path(args.out_dir) / 'summary.json'}")


def cmd_poddar_check(args: argparse.Namespace) -> None:
    from drybench.poddar import labelled_cases, load_poddar
    from drybench.realcheck import human_label_check

    cases = labelled_cases(load_poddar(args.data))[: args.max_cases]
    reader = Reader.for_backend(_make(args, "reader"))
    s = human_label_check(reader, cases, args.out_dir, abs_tol=args.abs_tol)
    o = s.get("overall") or {}
    if o.get("fragments"):
        f = o["fragments"]
        print(f"{s['n_scored_ok']} of {s['n_responses']} labelled responses scored; fragments the annotators "
              f"marked as not minimal: precision {f['precision']:.3f}, recall {f['recall']:.3f}, "
              f"Cohen's κ {f['cohen_kappa']:.2f}; minimal answer kept in {100 * o['minimal_answer_kept']:.1f}%")
    print(f"written: {Path(args.out_dir) / 'summary.json'}")


def _baseline_cases(args: argparse.Namespace, set_name: str) -> list[dict]:
    from drybench.compare import human_cases, planted_cases

    root = Path(args.run)
    if set_name == "sample":
        from drybench.annotate import sample_cases

        return sample_cases(root)
    if set_name == "human_poddar":
        docs, resp = {}, {}
        if args.poddar_data:  # rows written before v1.0 carry neither the passage nor the response
            from drybench.poddar import labelled_cases, load_poddar

            for c in labelled_cases(load_poddar(args.poddar_data)):
                docs[c["item"].item_id], resp[c["item"].item_id] = c["item"].document, c["response"]
        cases = human_cases(root, docs)
        for c in cases:
            if c["id"] in resp:
                c["response"] = resp[c["id"]]
        return cases
    # planted_real: the run's own items; the synthetic check generates its items (same ids, another seed)
    items = ({it.item_id: it for it in load_items(root / "items.jsonl")}
             if set_name == "planted_real" and (root / "items.jsonl").exists() else None)
    cases = planted_cases(root, set_name, items)
    if items:  # the tables leave out items with a missing fact
        from drybench.rescore import has_missing

        cases = [c for c in cases if c["group"] not in items or not has_missing(items[c["group"]])]
    return cases


def cmd_baselines(args: argparse.Namespace) -> None:
    from drybench.compare import baseline_path, response_hash

    backend = None
    for set_name in args.set:
        cases = _baseline_cases(args, set_name)[: args.max_cases]
        out = baseline_path(Path(args.run), args.method, args.label, set_name)
        done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
        todo = [c for c in cases if c["id"] not in done and c["fragments"]]
        if args.method == "judge" and any(not c["document"] for c in todo):
            raise SystemExit(f"{set_name}: the judge needs the source texts; these rows do not carry them "
                             "(human_poddar: pass --poddar-data with Poddar et al.'s data.jsonl)")
        if not todo:
            print(f"baselines {args.method}-{args.label} on {set_name}: nothing to do", file=sys.stderr)
            continue
        if backend is None:
            backend = _make(args)
        for n, c in enumerate(todo, 1):
            t0 = time.perf_counter()
            if args.method == "judge":
                from drybench.judge import judge_fragments

                res = judge_fragments(backend, c["document"], c["request"], c["fragments"], c["source_label"])
                row = {"id": c["id"], "labels": [x["label"] for x in res], "p": [x["p"] for x in res],
                       "lp": [x["lp"] for x in res], "mass": [x["mass"] for x in res]}
            else:
                from drybench.selective import fragment_self_information

                res = fragment_self_information(backend, c["request"], c["response"], c["fragments"])
                row = {"id": c["id"], "self_info": [x["self_info"] for x in res], "tokens": [x["tokens"] for x in res]}
            row.update({"seconds": round(time.perf_counter() - t0, 4), "device": _device_name(backend),
                        "hash": response_hash(c["response"])})
            _append(str(out), row)
            if n % 25 == 0 or n == len(todo):
                print(f"baselines {args.method}-{args.label} {set_name} {n}/{len(todo)}", file=sys.stderr)


def cmd_annotation_sample(args: argparse.Namespace) -> None:
    from drybench.annotate import build_sample, write_sample

    sample, key, stats = build_sample(args.runs, args.sources, items_per_source=args.items_per_source,
                                      per_item=args.per_item, min_distinct=args.min_distinct, seed=args.seed)
    write_sample(args.out, sample, key, stats)
    print(f"{stats['responses']} responses to {stats['items']} items ({stats['fragments']} fragments) -> "
          f"{Path(args.out) / 'sample.jsonl'}; by source {stats['by_source']}")


def cmd_abstain_sample(args: argparse.Namespace) -> None:
    from drybench.annotate import build_abstain, write_abstain

    rows, key, stats = build_abstain(args.runs, n=args.n, seed=args.seed)
    write_abstain(args.out, rows, key, stats)
    print(f"{stats['answers']} answers to missing-fact items -> {Path(args.out) / 'abstain_sample.csv'}; "
          f"by source {stats['by_source']}")


def cmd_annotation_report(args: argparse.Namespace) -> None:
    from drybench.annotate import abstain_check, report

    text = report(args.sample, args.labels, primary=args.primary) if args.labels else ""
    if args.abstain_csv:
        res = abstain_check(args.abstain_csv, Path(args.sample) / "abstain_key.jsonl")
        lines = ["", "## The abstention rule on held-out labels", "",
                 f"{res['labelled']} answers labelled ({res['labels']}); A or B = says the fact is not given."]
        for name in [k for k in res if isinstance(res[k], dict) and "precision" in res[k]]:
            v = res[name]
            lines.append(f"- {name}: {v['n']} answers, {v['say_it']} say it; precision {v['precision']:.3f} "
                         f"({v['tp']}/{v['tp'] + v['fp']}), recall {v['recall']:.3f} ({v['tp']}/{v['tp'] + v['fn']})")
        if res["disagreements"]:
            lines.append("- rule and label disagree on: " + ", ".join(res["disagreements"]))
        if res["unreadable_labels"]:
            lines.append("- labels not in A/B/V/O: " + ", ".join(res["unreadable_labels"]))
        text += "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else Path(args.sample) / "labels_report.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out}")


def cmd_analyze(args: argparse.Namespace) -> None:
    from drybench.analysis import analyze, build_tables, group_summary

    items = load_items(args.items)
    item_df, span_df = build_tables(items, read_jsonl(args.scored), read_jsonl(args.signals),
                                    h_key=args.h_key)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    item_df.to_csv(out / "item_table.csv", index=False)
    span_df.to_csv(out / "span_table.csv", index=False)
    res = analyze(item_df, span_df, n_boot=args.n_boot)
    res["groups"] = group_summary(item_df)
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv")
    summary = {name: json.loads(df.to_json(orient="index")) for name, df in res.items()
               if name in ("profiles", "attribution", "attribution_enrichment", "groups")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    try:
        from drybench.plots import report_figures

        figs = report_figures(item_df, res, out)
        print("figures:", ", ".join(p.name for p in figs))
    except ImportError:
        print("matplotlib not installed: skipping figures")
    print(f"wrote {', '.join(sorted(res))} to {out}")
    if "profiles" in res and not res["profiles"].empty:
        cols = ["n", "alpha", "beta", "beta_ctrl", "gamma", "gamma_filler", "W_mean", "kappa_mean"]
        print(res["profiles"][cols].round(3).to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="drybench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("items", help="generate closed-world items")
    a.add_argument("--out", required=True)
    a.add_argument("--n-per-cell", type=int, default=30)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--update-probes", action="store_true",
                   help="rewrite the probes of the existing --out items (documents must match)")
    a.set_defaults(fn=cmd_items)

    ri = sub.add_parser("real-items", help="items from SQuAD 2.0, MuSiQue, CUAD or Poddar et al. (a pool to filter)")
    ri.add_argument("--source", required=True, choices=["squad2", "musique", "cuad", "poddar"])
    ri.add_argument("--path", nargs="+", required=True,
                    help="dataset file(s): SQuAD 2.0 dev json; MuSiQue jsonl (Full has unanswerable ones); "
                         "CUADv1.json; Poddar et al. data.jsonl")
    ri.add_argument("--out", required=True)
    ri.add_argument("--n-per-cell", type=int, default=90, help="pool size per cell (filter keeps fewer)")
    ri.add_argument("--seed", type=int, default=0)
    ri.add_argument("--distractors", type=int, default=3, help="MuSiQue distractor paragraphs per item")
    ri.set_defaults(fn=cmd_real_items)

    fi = sub.add_parser("filter-items", help="drop real-text items the reader can answer without a response")
    fi.add_argument("--items", nargs="+", required=True, help="pool file(s) from real-items")
    fi.add_argument("--out", required=True)
    fi.add_argument("--report", default=None, help="json with per-cell counts")
    fi.add_argument("--n-per-cell", type=int, default=30)
    fi.add_argument("--max-prior", type=float, default=0.05,
                    help="drop an item if the reader answers one of its facts from an empty message with p >= this")
    fi.add_argument("--min-gold-p", type=float, default=0.5,
                    help="drop an item if the reader reads one of its facts from the gold answer with p < this")
    fi.add_argument("--keep", default=None,
                    help="items kept by an earlier filter (e.g. the current items.jsonl): kept as they are; "
                         "cells are topped up from the pool")
    fi.add_argument("--exclude-items", nargs="+", default=None,
                    help="items of other runs (e.g. the development runs): pool items from the same source text "
                         "are left out")
    _backend_args(fi, "reader")
    fi.set_defaults(fn=cmd_filter_items)

    g = sub.add_parser("generate", help="generate responses with an evaluated model")
    g.add_argument("--items", required=True)
    g.add_argument("--out", required=True)
    _backend_args(g)
    g.add_argument("--model-name", default=None, help="label stored with the responses")
    g.add_argument("--variant", nargs="+", default=["default"], choices=["default", "concise"],
                   help="one or more prompt variants (model is loaded once)")
    g.add_argument("--max-new-tokens", type=int, default=512)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("score", help="score responses with a reader model (filler map)")
    s.add_argument("--items", required=True)
    s.add_argument("--responses", required=True)
    s.add_argument("--out", required=True)
    _backend_args(s, "reader")
    s.add_argument("--delta", type=float, default=0.0,
                   help="relative tolerance (fraction of each probe's gain); 0 = absolute only")
    s.add_argument("--ablation-delta", type=float, default=None,
                   help="also score with this relative tolerance and store it under 'ablation'")
    s.add_argument("--ablation-abs-tol", type=float, default=None,
                   help="also score with this absolute tolerance (nats) and store it under 'ablation'")
    s.add_argument("--mode", default="greedy", choices=["greedy", "loo"])
    s.add_argument("--constraint", default="per_probe", choices=["per_probe", "global"])
    s.add_argument("--abs-tol", type=float, default=0.5, help="per-probe slack in nats")
    s.add_argument("--segment", default="sentence", choices=["sentence", "clause"])
    s.add_argument("--judge", action="store_true", help="also compute the LLM-judge filler baseline")
    s.set_defaults(fn=cmd_score)

    sg = sub.add_parser("signals", help="entropy signals of the evaluated model")
    sg.add_argument("--items", required=True)
    sg.add_argument("--responses", required=True)
    sg.add_argument("--out", required=True)
    _backend_args(sg)
    sg.add_argument("--model-name", default=None, help="must match the label used in generate")
    sg.add_argument("--sampled", type=int, default=0, help="N samples for sampled H_ans (0 = off)")
    sg.add_argument("--segment", default="sentence", choices=["sentence", "clause"])
    sg.set_defaults(fn=cmd_signals)

    cr = sub.add_parser("check-reader", help="can the reader read the facts at all?")
    cr.add_argument("--items", required=True)
    cr.add_argument("--out", default=None)
    cr.add_argument("--n", type=int, default=30)
    _backend_args(cr, "reader")
    cr.set_defaults(fn=cmd_check_reader)

    mc = sub.add_parser("missing-check", help="how the reader reads a missing fact, per way of asking it")
    mc.add_argument("--run", default=None, help="run folder: its missing-fact items and scored answers")
    mc.add_argument("--synthetic-n", type=int, default=0, help="also the synthetic missing-fact items (n per cell)")
    mc.add_argument("--styles", nargs="+", default=["plain", "yesno", "yesno2", "contains", "choice"])
    mc.add_argument("--show", type=int, default=1, help="greedy replies for this many abstaining answers")
    mc.add_argument("--max-per-kind", type=int, default=150)
    mc.add_argument("--chat", action="store_true", help="chat reader whatever the backend (tests)")
    mc.add_argument("--out", default=None)
    _backend_args(mc, "reader")
    mc.set_defaults(fn=cmd_missing_check)

    rw = sub.add_parser("real-filler-check", help="filler map on real-text responses with planted filler (truth known)")
    rw.add_argument("--items", nargs="+", required=True, help="filtered real items (items.jsonl)")
    rw.add_argument("--out-dir", required=True)
    rw.add_argument("--n-per-cell", type=int, default=4, help="items per cell; 5 responses each")
    _backend_args(rw, "reader")
    rw.set_defaults(fn=cmd_real_filler_check)

    pc = sub.add_parser("poddar-check", help="filler map vs human span labels (Poddar et al. 2025)")
    pc.add_argument("--data", required=True, help="Poddar et al. data.jsonl")
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--abs-tol", type=float, default=0.5)
    pc.add_argument("--max-cases", type=int, default=None)
    _backend_args(pc, "reader")
    pc.set_defaults(fn=cmd_poddar_check)

    ps = sub.add_parser("prune-scores", help="remove the scores a new scoring version makes stale (all, or "
                        "only those it changed), so that the next steps redo them")
    ps.add_argument("--dir", required=True, help="a run folder or its reader2_* folder")
    ps.add_argument("--items", required=True)
    ps.add_argument("--reader", default=None, help="the folder's reader (HF id); unknown: rescore all")
    ps.add_argument("--from", dest="old", required=True, help="scoring version the folder was scored with")
    ps.add_argument("--to", default=None, help="default: this package's SCORING_VERSION")
    ps.add_argument("--json", action="store_true", help="print {plan, changes} as JSON")
    ps.set_defaults(fn=cmd_prune_scores)

    bl = sub.add_parser("baselines", help="LLM judge or Selective Context on a check's responses or the annotation sample")
    bl.add_argument("--run", required=True, help="a run folder (or its reader2_* folder); for --set sample, the "
                                                 "annotation folder")
    bl.add_argument("--set", nargs="+", required=True, choices=["planted_synthetic", "planted_real", "human_poddar",
                                                                  "sample"])
    bl.add_argument("--method", required=True, choices=["judge", "selective"])
    bl.add_argument("--label", required=True, help="name of this baseline in the tables, e.g. phi-4")
    bl.add_argument("--poddar-data", default=None, help="Poddar et al.'s data.jsonl (human_poddar rows before v1.0)")
    bl.add_argument("--max-cases", type=int, default=None)
    _backend_args(bl)
    bl.set_defaults(fn=cmd_baselines)

    asmp = sub.add_parser("annotation-sample", help="blind sample of the runs' answers for human labels, item by item")
    asmp.add_argument("--runs", nargs="+", required=True, help="run folders (synthetic and/or real texts)")
    asmp.add_argument("--out", required=True, help="folder for sample.jsonl and key.jsonl")
    asmp.add_argument("--sources", nargs="+", default=None, help="e.g. synthetic squad2 musique cuad (default: all)")
    asmp.add_argument("--items-per-source", type=int, default=6)
    asmp.add_argument("--per-item", type=int, default=8, help="answers per item (different models / prompts)")
    asmp.add_argument("--min-distinct", type=int, default=3, help="items need at least this many distinct answers")
    asmp.add_argument("--seed", type=int, default=0)
    asmp.set_defaults(fn=cmd_annotation_sample)

    absm = sub.add_parser("abstain-sample", help="answers to missing-fact items for held-out labels of the abstention rule")
    absm.add_argument("--runs", nargs="+", required=True)
    absm.add_argument("--out", required=True)
    absm.add_argument("--n", type=int, default=150)
    absm.add_argument("--seed", type=int, default=0)
    absm.set_defaults(fn=cmd_abstain_sample)

    arp = sub.add_parser("annotation-report", help="human labels vs each other, the filler map and the baselines")
    arp.add_argument("--sample", required=True, help="the annotation folder (sample.jsonl, key.jsonl, baselines/)")
    arp.add_argument("--labels", nargs="*", default=[], help="label exports of the labelling page and/or "
                                                            "chat annotators' JSONL")
    arp.add_argument("--primary", default=None, help="annotator whose labels are the truth (default: the most complete)")
    arp.add_argument("--abstain-csv", default=None, help="abstain_sample.csv with the label column filled in")
    arp.add_argument("--out", default=None)
    arp.set_defaults(fn=cmd_annotation_report)

    an = sub.add_parser("analyze", help="profiles and attribution")
    an.add_argument("--items", required=True)
    an.add_argument("--scored", required=True)
    an.add_argument("--signals", required=True)
    an.add_argument("--out-dir", required=True)
    an.add_argument("--h-key", default="H_first", choices=["H_first", "H_sampled"])
    an.add_argument("--n-boot", type=int, default=500)
    an.set_defaults(fn=cmd_analyze)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
