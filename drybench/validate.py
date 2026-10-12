"""Reader sanity check: can the reader read the facts at all?

On a sample of items it scores (a) the gold minimal answer and (b) the same facts with one
fact omitted. A usable reader gives large gains for present facts, ~0 for omitted ones
(AUC close to 1) and rarely fails to read the gold answer. For chat readers it also checks
that an empty message is answered 'unknown', which the 'misleading' span label relies on.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

import numpy as np

from drybench.reader import Reader
from drybench.simulate import simulate
from drybench.types import Item


def check_reader(reader: Reader, items: Sequence[Item], n: int = 30, seed: int = 0,
                 min_info: float = 1.0) -> dict:
    rng = random.Random(seed)
    sample = list(items) if len(items) <= n else rng.sample(list(items), n)
    present, absent, fails = [], [], 0
    for it in sample:
        probes = it.required_probes
        gold = reader.per_probe_information(it.request, [it.gold_minimal], probes)[0]
        fails += int(gold.sum() < min_info)
        present += gold.tolist()
        for j, p in enumerate(probes[:2]):
            others = " ".join(q.question for q in probes if q is not p).lower()
            if any(re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", others) for a in p.answers):
                continue  # a chained fact (multi-hop): the next question names it, so it cannot be absent
            sim = simulate(it, [], omit=[p.fact_id], rng=random.Random(seed))
            absent.append(float(reader.per_probe_information(it.request, [sim.text], probes)[0][j]))
    unk_empty, other_empty = [], []
    if reader.config.use_chat:
        from drybench.filler import wrong_answer_mass

        for it in sample:
            p_other = wrong_answer_mass(reader, it.request, [""], it.required_probes)[0]
            unk_empty += (1.0 - p_other).tolist()  # gold or an absence reply ('unknown', ...)
            other_empty += p_other.tolist()
    # missing facts: does the reader read 'not stated' from the gold, but not from nothing and not from
    # the gold without its 'not stated' sentence (silence is not an abstention)?
    miss_prior, miss_gold, miss_silent = [], [], []
    if reader.config.use_chat:
        from drybench.reader import is_missing

        for it in [it for it in items if any(is_missing(p) for p in it.required_probes)][:60]:
            silent = _silent(it.gold_minimal)
            for p in it.required_probes:
                if is_missing(p):
                    lp = reader.probe_logprobs(it.request, ["", it.gold_minimal] + ([silent] if silent else []),
                                               [p])[:, 0]
                    miss_prior.append(float(np.exp(lp[0])))
                    miss_gold.append(float(np.exp(lp[1])))
                    if silent:
                        miss_silent.append(float(np.exp(lp[2])))
    pres, abs_ = np.array(present), np.array(absent)
    auc = float(np.mean(pres[:, None] > abs_[None, :])) if len(pres) and len(abs_) else float("nan")
    return {
        "n_items": len(sample),
        "reader_fail_rate": fails / len(sample) if sample else float("nan"),
        "gain_present_median": float(np.median(pres)) if len(pres) else float("nan"),
        "gain_present_p10": float(np.quantile(pres, 0.1)) if len(pres) else float("nan"),
        "gain_absent_median": float(np.median(abs_)) if len(abs_) else float("nan"),
        "gain_absent_p90": float(np.quantile(abs_, 0.9)) if len(abs_) else float("nan"),
        "auc_present_vs_absent": auc,
        # share of questions for which 'misleading' spans can be detected (empty message → 'unknown')
        "p_unknown_empty_median": float(np.median(unk_empty)) if unk_empty else float("nan"),
        "misleading_check_active": float(np.mean(np.array(other_empty) < 0.5)) if other_empty
        else float("nan"),
        "n_missing": len(miss_prior),
        "missing_prior_p_median": float(np.median(miss_prior)) if miss_prior else float("nan"),
        "missing_prior_p_max": float(np.max(miss_prior)) if miss_prior else float("nan"),
        "missing_gold_p_median": float(np.median(miss_gold)) if miss_gold else float("nan"),
        "missing_silent_p_median": float(np.median(miss_silent)) if miss_silent else float("nan"),
        "missing_silent_p_max": float(np.max(miss_silent)) if miss_silent else float("nan"),
        "missing_style": reader.config.missing_style if reader.config.use_chat else None,
    }


# ---------------------------------------------------------------- missing facts: which question

def _silent(text: str) -> str:
    """The text without its sentences that say a fact is not stated."""
    from drybench.reader import ABSTAIN
    from drybench.segment import segment

    return " ".join(s.text for s in segment(text) if not ABSTAIN.search(s.text)).strip()


def missing_cases(items: Sequence[Item], scored: Sequence[dict] = (), max_per_kind: int = 150,
                  seed: int = 0) -> list[dict]:
    """Messages that test how a reader reads a missing fact:

    gold               the gold answer (full) and the gold answer without its 'not stated' sentence
                       (silent; empty for a one-fact item)
    abstaining answer  a scored answer to a missing-fact item with fragments that say the fact is not
                       stated (full) and the same answer without them (silent: what the filler map
                       reads when it deletes them)
    other answer       a scored answer to a missing-fact item that never says so (full only); a
                       reader that reads 'not stated' from it reads silence as an abstention
    """
    from drybench.reader import ABSTAIN, is_missing

    by_id = {it.item_id: it for it in items}
    miss = {it.item_id: next((p for p in it.required_probes if is_missing(p)), None) for it in items}
    cases = [{"kind": "gold", "item": it, "probe": miss[it.item_id], "full": it.gold_minimal,
              "silent": _silent(it.gold_minimal)} for it in items if miss[it.item_id] is not None]
    for r in scored:
        probe = miss.get(r["item_id"])
        if probe is None or r.get("status") != "ok":
            continue
        texts = [sp["text"] for sp in r["spans"]]
        ab = [bool(ABSTAIN.search(t)) for t in texts]
        case = {"item": by_id[r["item_id"]], "probe": probe, "full": " ".join(texts),
                "model": r.get("model"), "variant": r.get("variant", "default")}
        if any(ab):
            cases.append({**case, "kind": "abstaining answer",
                          "silent": " ".join(t for t, a in zip(texts, ab) if not a)})
        else:
            cases.append({**case, "kind": "other answer", "silent": None})
    rng, out = random.Random(seed), []
    for kind in ("gold", "abstaining answer", "other answer"):
        group = [c for c in cases if c["kind"] == kind]
        out += group if len(group) <= max_per_kind else rng.sample(group, max_per_kind)
    return out


def missing_question_check(reader: Reader, cases: Sequence[dict], styles: Sequence[str],
                           abs_tol: float = 0.5, show: int = 0) -> dict:
    """For each way of asking a missing fact (reader.MISSING_STYLES), how the reader reads each case:
    p(accepted reply) for the full message, the silent one and no message; whether the fact is
    readable (p >= 0.5) and, where the silent message has text, whether the filler map could delete
    the abstention (it cannot when its loss exceeds the tolerance min(abs_tol, I/2), I >= abs_tol)."""
    from dataclasses import replace

    out: dict = {}
    for style in styles:
        rd = Reader(reader.backend, replace(reader.config, missing_style=style))
        rows = []
        for c in cases:
            it, probe = c["item"], c["probe"]
            floor = float(rd.log_floors([probe])[0])  # a yes/no question has its own floor
            msgs = [c["full"]] + ([c["silent"]] if c["silent"] is not None else [])
            lp = rd.probe_logprobs(it.request, msgs, [probe])[:, 0]
            lp0 = float(rd.prior_logprobs(it.request, [probe])[0])
            gain = max(lp[0], floor) - max(lp0, floor)
            row = {"kind": c["kind"], "p_full": float(np.exp(lp[0])), "p_empty": float(np.exp(lp0))}
            if c["silent"] is not None:
                silent_gain = max(lp[1], floor) - max(lp0, floor)
                row["p_silent"] = float(np.exp(lp[1]))
                if c["silent"]:  # an abstention next to other text: can the filler map delete it?
                    row["protected"] = bool(gain >= abs_tol and gain - silent_gain > min(abs_tol, gain / 2))
            rows.append(row)
        shown = [c for c in cases if c["kind"] == "abstaining answer"][:show]
        replies = [{"question": c["probe"].question,
                    **{name: rd.answer(c["item"].request, msg, rd.question(c["probe"]), max_new_tokens=12)
                       for name, msg in (("full", c["full"]), ("silent", c["silent"]), ("no message", ""))}}
                   for c in shown]
        by_kind = {"replies": replies} if replies else {}
        for kind in ("gold", "abstaining answer", "other answer"):
            k = [r for r in rows if r["kind"] == kind]
            if not k:
                continue
            med = lambda key: float(np.median([r[key] for r in k if key in r])) if any(key in r for r in k) \
                else float("nan")  # noqa: E731
            prot = [r["protected"] for r in k if "protected" in r]
            by_kind[kind] = {"n": len(k), "p_full_median": med("p_full"), "p_silent_median": med("p_silent"),
                             "p_empty_median": med("p_empty"),
                             "readable": float(np.mean([r["p_full"] >= 0.5 for r in k])),
                             "protected": float(np.mean(prot)) if prot else float("nan"), "n_protected": len(prot)}
        out[style] = by_kind
    return out


PASS = {"kept": 0.9, "gold_read": 0.8, "abstaining_read": 0.8, "other_read": 0.2, "p_no_message": 0.05}


def missing_check_summary(res: dict) -> dict:
    """Per style: the numbers that decide (abstentions kept by the map, gold and abstentions read,
    answers without an abstention read as one, p with no message) and whether all pass PASS."""
    out = {}
    for style, by_kind in res.items():
        g, a, o = (by_kind.get(k) or {} for k in ("gold", "abstaining answer", "other answer"))
        v = {"kept": a.get("protected", float("nan")), "gold_read": g.get("readable", float("nan")),
             "abstaining_read": a.get("readable", float("nan")), "other_read": o.get("readable", float("nan")),
             "p_no_message": g.get("p_empty_median", float("nan"))}
        ok = lambda key, high: v[key] != v[key] or (v[key] >= PASS[key] if high else v[key] <= PASS[key])  # noqa: E731
        v["pass"] = bool(ok("kept", True) and ok("gold_read", True) and ok("abstaining_read", True)
                         and ok("other_read", False) and ok("p_no_message", False))  # NaN: nothing to judge
        out[style] = v
    return out


def format_missing_check(res: dict) -> str:
    lines = ["style | cases | n | p(full) | p(silent) | p(no message) | read (p ≥ 0.5) | abstention kept by the map"]
    for style, by_kind in res.items():
        for kind, v in by_kind.items():
            if kind == "replies":
                continue
            prot = f"{v['protected']:.2f} (n={v['n_protected']})" if v["n_protected"] else "—"
            lines.append(f"{style} | {kind} | {v['n']} | {v['p_full_median']:.3f} | {v['p_silent_median']:.3f} | "
                         f"{v['p_empty_median']:.3f} | {v['readable']:.2f} | {prot}")
    for style, by_kind in res.items():
        for r in by_kind.get("replies", []):
            cut = lambda t: " ".join(str(t).split())[:40]  # noqa: E731
            lines.append(f"  {style} replies to «{cut(r['question'])}»: with the abstention «{cut(r['full'])}» | "
                         f"without it «{cut(r['silent'])}» | no message «{cut(r['no message'])}»")
    lines.append("")
    lines.append("style | abstentions kept | gold read | abstaining read | other read (false) | p, no message | pass")
    for style, v in missing_check_summary(res).items():
        lines.append(f"{style} | {v['kept']:.2f} | {v['gold_read']:.2f} | {v['abstaining_read']:.2f} | "
                     f"{v['other_read']:.2f} | {v['p_no_message']:.3f} | {'yes' if v['pass'] else 'no'}")
    return "\n".join(lines)
