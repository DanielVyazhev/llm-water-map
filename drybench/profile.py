"""Model verbosity profile: alpha (autopilot), beta (entropy -> filler), gamma (complexity ->
substance), with item-bootstrap confidence intervals.

Input: one row per (model, item) with columns
    model, item_id, k, answerable, ambiguous, hops, H_ans, W, tokens, tokens_kept, kappa
H_ans is converted to a within-model percentile H_pct.

alpha        mean W on easy items the model is sure about: k=1, answerable, unambiguous, 1 hop,
             H_pct <= 0.5 (alpha_n = number of such items)
beta         OLS slope of W on H_pct, controlling for k
beta_ctrl    same, also controlling for unanswerable / ambiguous / 2-hop / 3+-hop / "explain" design knobs
             (explain since v1.2: the knob that drives filler most)
gamma        OLS slope of substance tokens (tokens_kept) on k, controlling for H_pct, on plain
             items (answerable, unambiguous, 1 hop), where k is exactly the number of facts asked
gamma_filler OLS slope of filler tokens on k, same items and controls (ideal: ~0)
W is the filler fraction (uninformative + duplicate spans; misleading spans are errors, not filler).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["model", "item_id", "k", "answerable", "ambiguous", "hops", "H_ans", "W",
                    "tokens", "tokens_kept", "kappa"]
STATS = ["alpha", "beta", "beta_ctrl", "gamma", "gamma_filler", "W_mean", "kappa_mean"]
POINT_ONLY = ["alpha_n"]


def ols_coef(y: np.ndarray, X: dict[str, np.ndarray], target: str) -> float:
    """OLS coefficient of `target`; constant regressors are dropped. NaN if not identified."""
    cols = {k: np.asarray(v, float) for k, v in X.items() if np.nanstd(v) > 0}
    if target not in cols or len(y) <= len(cols) + 1:
        return float("nan")
    names = list(cols)
    A = np.column_stack([np.ones(len(y))] + [cols[n] for n in names])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return float(coef[1 + names.index(target)])


def _stats(d: pd.DataFrame) -> dict[str, float]:
    ans = d["answerable"].astype(bool)
    amb = d["ambiguous"].astype(bool)
    easy = d[(d["k"] == 1) & ans & ~amb & (d["hops"] == 1) & (d["H_pct"] <= 0.5)]
    k = d["k"].to_numpy(float)
    h = d["H_pct"].to_numpy(float)
    W = d["W"].to_numpy(float)
    filler_tok = (d["filler_tokens"] if "filler_tokens" in d else d["tokens"] - d["tokens_kept"]).to_numpy(float)
    kept = d["tokens_kept"].to_numpy(float)
    # special designs (unanswerable, ambiguous, 2-hop) have k=1 but other answer shapes
    plain = (ans & ~amb & (d["hops"] == 1)).to_numpy()
    return {
        "alpha": float(easy["W"].mean()) if len(easy) else float("nan"),
        "beta": ols_coef(W, {"H_pct": h, "k": k}, "H_pct"),
        "beta_ctrl": ols_coef(W, {"H_pct": h, "k": k, "unans": (~ans).to_numpy(float),
                                  "amb": amb.to_numpy(float),
                                  "hop2": (d["hops"] == 2).to_numpy(float),
                                  "hop3p": (d["hops"] >= 3).to_numpy(float),
                                  "explain": (d["explain"].astype(bool).to_numpy(float) if "explain" in d
                                              else np.zeros(len(d)))}, "H_pct"),
        "gamma": ols_coef(kept[plain], {"k": k[plain], "H_pct": h[plain]}, "k"),
        "gamma_filler": ols_coef(filler_tok[plain], {"k": k[plain], "H_pct": h[plain]}, "k"),
        "alpha_n": float(len(easy)),
        "W_mean": float(np.nanmean(W)),
        "kappa_mean": float(np.nanmean(d["kappa"].to_numpy(float))),
    }


def model_profile(d: pd.DataFrame, n_boot: int = 500, seed: int = 0, ci: float = 0.95) -> dict[str, float]:
    d = d.dropna(subset=["W", "H_ans"]).reset_index(drop=True)
    if "H_pct" not in d:
        d = d.assign(H_pct=d["H_ans"].rank(pct=True, method="average"))
    point = _stats(d)
    rng = np.random.default_rng(seed)
    boots = {s: [] for s in STATS}
    for _ in range(n_boot):
        b = _stats(d.iloc[rng.integers(0, len(d), len(d))])
        for s in STATS:
            boots[s].append(b[s])
    out: dict[str, float] = {"n": float(len(d)), **{k: point[k] for k in POINT_ONLY}}
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    for s in STATS:
        out[s] = point[s]
        vals = np.array(boots[s], float)
        vals = vals[~np.isnan(vals)]
        out[f"{s}_lo"] = float(np.quantile(vals, lo_q)) if len(vals) else float("nan")
        out[f"{s}_hi"] = float(np.quantile(vals, hi_q)) if len(vals) else float("nan")
    return out


def profiles(df: pd.DataFrame, by: str = "model", n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    rows = {m: model_profile(g, n_boot=n_boot, seed=seed) for m, g in df.groupby(by)}
    return pd.DataFrame.from_dict(rows, orient="index")
