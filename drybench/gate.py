"""Entropy gate: decide before generating whether to answer normally or fall back.

A short-answer pass gives H_ans (prefill is shared with the main generation). Items
with H_ans > tau get a fallback: a concise uncertain answer / clarification, or routing
to a stronger model. Evaluated offline from logged runs of both policies.

Cost-of-Pass = total cost / number of successful responses (kappa >= success_kappa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_gate(df: pd.DataFrame, tau: float, h_col: str = "H_ans", tokens_col: str = "tokens",
                  kappa_col: str = "kappa", fb_tokens_col: str = "fb_tokens",
                  fb_kappa_col: str = "fb_kappa", price: float = 1.0, fb_price: float | None = None,
                  success_kappa: float = 0.8, overhead_tokens: float = 0.0) -> dict[str, float]:
    fb_price = price if fb_price is None else fb_price
    gated = (df[h_col] > tau).to_numpy()
    tokens = np.where(gated, df[fb_tokens_col], df[tokens_col]).astype(float)
    kappa = np.where(gated, df[fb_kappa_col], df[kappa_col]).astype(float)
    cost = np.where(gated, df[fb_tokens_col] * fb_price, df[tokens_col] * price) + overhead_tokens * price
    success = np.nan_to_num(kappa, nan=0.0) >= success_kappa
    n_success = int(success.sum())
    return {
        "tau": float(tau),
        "frac_gated": float(gated.mean()),
        "tokens_mean": float(tokens.mean()),
        "kappa_mean": float(np.nanmean(kappa)),
        "success_rate": float(success.mean()),
        "cost_total": float(cost.sum()),
        "cost_of_pass": float(cost.sum() / n_success) if n_success else float("inf"),
    }


def gate_curve(df: pd.DataFrame, taus: list[float] | None = None, h_col: str = "H_ans",
               **kwargs) -> pd.DataFrame:
    """Gate metrics over thresholds (default: deciles of H_ans plus 'never gate')."""
    if taus is None:
        taus = sorted(set(np.quantile(df[h_col], np.linspace(0, 0.9, 10)).tolist())) + [float("inf")]
    return pd.DataFrame([evaluate_gate(df, t, h_col=h_col, **kwargs) for t in taus])
