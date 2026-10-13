"""The confirmatory tests of the protocol (docs/PROTOCOL.md): eight one-sided tests on the main
reader's filler map, Holm-adjusted.

H1  T1, R3      validity of the deletions on planted filler, on the responses with a filler map: a requested
                fact lost in fewer than 5% of them, and more than 95% of the deleted fragments planted filler or
                planted wrong statements (intersection-union test: p is the larger of the two p-values).
H2  T1, R3, HL  the filler map ranks responses by filler better than length: pairwise accuracy (pairs whose true
                removable share differs by ≥ 0.2; ties count half) of the filler map minus that of length > 0.
H3  T1, R3, HL  the same with Spearman's correlation with the true removable share.

T1: planted filler in synthetic reports; R3: planted filler in real texts (items without a missing fact); HL: the
human labels of the annotation sample (truth: the annotator with the most complete labels, or --primary). H2 and
H3 count every response the truth covers: one from which the reader reads none of the requested facts is all
removable (the score the benchmark gives it); a response whose gold answer the reader cannot read stays out.
p-values: B bootstrap resamples of items (default 2000, seed 0), p = (1 + #{resamples at or beyond the null}) /
(B + 1). Holm over the eight tests, family-wise α = 0.05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

ALPHA = 0.05
N_BOOT = 2000
MIN_GAP = 0.2
MAX_LOST = 0.05       # H1: share of responses with a lost fact
MIN_PRECISION = 0.95  # H1: share of deleted fragments that are planted filler or wrong
TESTS = [("H1", "T1"), ("H1", "R3"), ("H2", "T1"), ("H2", "R3"), ("H2", "HL"), ("H3", "T1"), ("H3", "R3"),
         ("H3", "HL")]
SETS = {"T1": "planted filler, synthetic reports", "R3": "planted filler, real texts",
        "HL": "human labels of the annotation sample"}


def resamples(groups: Sequence, n_boot: int = N_BOOT, seed: int = 0) -> Iterator[np.ndarray]:
    """Indices of each bootstrap resample of the groups (items): answers to one item are not independent."""
    blocks: dict = {}
    for i, g in enumerate(groups):
        blocks.setdefault(g, []).append(i)
    bl = [np.asarray(v) for v in blocks.values()]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([bl[j] for j in rng.integers(0, len(bl), len(bl))])


def p_value(boots: Sequence[float], null: float, greater: bool = True) -> float:
    """One-sided bootstrap p-value of 'the statistic is above null' (greater) or 'below null'."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return float("nan")
    beyond = (b <= null) if greater else (b >= null)
    return float((1 + beyond.sum()) / (len(b) + 1))


def ci(boots: Sequence[float]) -> tuple[float, float]:
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) else (np.nan, np.nan)


def holm(p: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (NaN stays NaN and does not count)."""
    p = np.asarray(p, float)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m, out, run = len(idx), np.full(len(p), np.nan), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * p[i]))
        out[i] = run
    return list(out)


# ---------------------------------------------------------------- H1

def validity(rows: Sequence[dict], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """H1 on the rows of a planted-filler check (rows.jsonl of filler_check or filler_check_real)."""
    ok = [r for r in rows if r.get("loo_status") == "ok"]
    out: dict = {"built": len(rows), "with_map": len(ok), "items": len({r["item_id"] for r in ok})}
    if not ok:
        return out
    lost = np.array([r["loo_safety"] < 1.0 for r in ok], float)
    removed = np.array([sum(not k for k in r["loo_kept"]) for r in ok], float)
    right = np.array([round(r["loo_filler_precision"] * n) for r, n in zip(ok, removed)], float)
    b_lost, b_prec = [], []
    for idx in resamples([r["item_id"] for r in ok], n_boot, seed):
        b_lost.append(lost[idx].mean())
        b_prec.append(right[idx].sum() / removed[idx].sum() if removed[idx].sum() else np.nan)
    out.update({"lost": float(lost.mean()), "lost_ci": ci(b_lost), "p_lost": p_value(b_lost, MAX_LOST, greater=False),
                "precision": float(right.sum() / removed.sum()) if removed.sum() else float("nan"),
                "precision_ci": ci(b_prec), "p_precision": p_value(b_prec, MIN_PRECISION),
                "deleted": int(removed.sum())})
    out["p"] = max(out["p_lost"], out["p_precision"])
    return out


# ---------------------------------------------------------------- H2, H3

def ranking(cases: Sequence[dict], key: str = "map:main", n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """H2 and H3 on cases (drybench.compare) carrying the filler map under `key`."""
    from drybench.compare import pair_accuracy, share, spearman

    cs = [c for c in cases if c["true_share"] == c["true_share"] and key in c.get("methods", {})]
    out: dict = {"responses": len(cs), "items": len({c["group"] for c in cs})}
    if len(cs) < 3:
        return out
    t = np.array([c["true_share"] for c in cs], float)
    m = np.array([share(c, c["methods"][key]["removable"]) for c in cs], float)
    L = np.array([c["length"] for c in cs], float)
    acc_m, pairs = pair_accuracy(t, m, MIN_GAP)
    acc_l, _ = pair_accuracy(t, L, MIN_GAP)
    rho_m, rho_l = spearman(m, t), spearman(L, t)
    b_acc, b_rho = [], []
    for idx in resamples([c["group"] for c in cs], n_boot, seed):
        b_acc.append(pair_accuracy(t[idx], m[idx], MIN_GAP)[0] - pair_accuracy(t[idx], L[idx], MIN_GAP)[0])
        b_rho.append(spearman(m[idx], t[idx]) - spearman(L[idx], t[idx]))
    out.update({"pairs": pairs, "acc_map": acc_m, "acc_length": acc_l, "acc_diff": acc_m - acc_l,
                "acc_ci": ci(b_acc), "p_acc": p_value(b_acc, 0.0), "rho_map": rho_m, "rho_length": rho_l,
                "rho_diff": rho_m - rho_l, "rho_ci": ci(b_rho), "p_rho": p_value(b_rho, 0.0)})
    return out


# ---------------------------------------------------------------- data

def planted(sub: Path, set_name: str) -> tuple[list[dict], list[dict]]:
    """Rows and cases (main reader; unscored responses all removable) of a planted check in a run folder;
    planted_real leaves out items with a missing fact, as the report's R3 and R5."""
    from drybench.compare import PLANTED_SETS, cases_for, fill_unscored, read_rows
    from drybench.rescore import has_missing
    from drybench.types import load_items

    sub = Path(sub)
    rows = read_rows(sub / PLANTED_SETS[set_name] / "rows.jsonl")
    items, missing = None, set()
    if set_name == "planted_real" and (sub / "items.jsonl").exists():
        loaded = load_items(sub / "items.jsonl")
        items = {i.item_id: i for i in loaded}
        missing = {i.item_id for i in loaded if has_missing(i)}
    rows = [r for r in rows if r["item_id"] not in missing]
    cases = [c for c in cases_for(set_name, [("main", sub)], sub, items) if c["group"] not in missing]
    return rows, fill_unscored(cases, ["map:main"])


def confirm(synthetic: str | Path | None = None, real: str | Path | None = None,
            annotation: str | Path | None = None, label_files: Sequence[str | Path] = (), primary: str | None = None,
            n_boot: int = N_BOOT, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """The eight tests: one row each (a missing truth set leaves its rows without p), and the parts."""
    parts: dict = {}
    if synthetic is not None and (Path(synthetic) / "filler_check" / "rows.jsonl").exists():
        rows, cases = planted(Path(synthetic), "planted_synthetic")
        parts["T1"] = {"H1": validity(rows, n_boot, seed), "H2": ranking(cases, n_boot=n_boot, seed=seed)}
    if real is not None and (Path(real) / "filler_check_real" / "rows.jsonl").exists():
        rows, cases = planted(Path(real), "planted_real")
        parts["R3"] = {"H1": validity(rows, n_boot, seed), "H2": ranking(cases, n_boot=n_boot, seed=seed)}
    if annotation is not None and label_files:
        from drybench.annotate import labelled_cases

        got = labelled_cases(annotation, label_files, primary)
        parts["HL"] = {"H2": ranking(got["lab_cases"], n_boot=n_boot, seed=seed), "primary": got["primary"]}
    out = {}
    for h, s in TESTS:
        part = parts.get(s, {})
        if h == "H1":
            v = part.get("H1") or {}
            est = (f"lost {_pct(v.get('lost'))} [{_pct(v.get('lost_ci', (np.nan,) * 2)[0])}, "
                   f"{_pct(v.get('lost_ci', (np.nan,) * 2)[1])}]; precision {_num(v.get('precision'), '.3f')} "
                   f"[{_num(v.get('precision_ci', (np.nan,) * 2)[0], '.3f')}, "
                   f"{_num(v.get('precision_ci', (np.nan,) * 2)[1], '.3f')}]") if v.get("with_map") else "–"
            n = f"{v.get('with_map', 0)} of {v.get('built', 0)} responses, {v.get('items', 0)} items" if v else "–"
            out[f"{h} {s}"] = {"truth set": SETS[s], "hypothesis": f"lost < {MAX_LOST:.0%} and precision > "
                               f"{MIN_PRECISION:.2f}", "data": n, "estimate [95% CI]": est,
                               "p": v.get("p", np.nan)}
        else:
            v = part.get("H2") or {}
            what, key = ("pairwise accuracy", "acc") if h == "H2" else ("Spearman", "rho")
            est = (f"{_num(v.get(f'{key}_map'), '.3f')} vs {_num(v.get(f'{key}_length'), '.3f')}; difference "
                   f"{_num(v.get(f'{key}_diff'), '+.3f')} [{_num(v[f'{key}_ci'][0], '+.3f')}, "
                   f"{_num(v[f'{key}_ci'][1], '+.3f')}]") if v.get(f"{key}_ci") else "–"
            n = (f"{v.get('responses', 0)} responses, {v.get('items', 0)} items"
                 + (f", {v['pairs']} pairs" if h == "H2" and v.get("pairs") else "")) if v else "–"
            out[f"{h} {s}"] = {"truth set": SETS[s], "hypothesis": f"{what}: filler map > length", "data": n,
                               "estimate [95% CI]": est, "p": v.get(f"p_{key}", np.nan)}
    t = pd.DataFrame(out).T
    t["p (Holm)"] = holm(t["p"].astype(float).tolist())
    t["result"] = [("–" if q != q else "confirmed" if q <= ALPHA else "not confirmed") for q in t["p (Holm)"]]
    t.index.name = "test"
    return t, parts


def _num(v, spec: str) -> str:
    return "n/a" if v is None or v != v else format(v, spec)


def _pct(v) -> str:
    return "n/a" if v is None or v != v else f"{100 * v:.1f}%"


def markdown(t: pd.DataFrame, parts: dict, n_boot: int = N_BOOT) -> str:
    from drybench.report import md

    done = int(np.isfinite(t["p"].astype(float)).sum())
    show = t.copy()
    for col in ("p", "p (Holm)"):
        show[col] = [("–" if v != v else "< 0.001" if v < 0.001 else f"{v:.3f}") for v in t[col].astype(float)]
    doc = ["# Confirmatory tests", "", f"Main reader; one-sided p-values from {n_boot} bootstrap resamples of items; "
           f"Holm over the {len(TESTS)} tests, family-wise α = {ALPHA}. Definitions: docs/PROTOCOL.md."]
    if done < len(TESTS):
        doc += ["", f"**Provisional:** {done} of {len(TESTS)} tests have data; the Holm adjustment is final only with "
                "all of them."]
    if "HL" in parts:
        doc += ["", f"Human labels: the truth is {parts['HL']['primary']}'s labels."]
    return "\n".join(doc + ["", md(show)]) + "\n"
