"""2x2 attribution of filler spans from two Shannon-entropy signals.

                      h_span low (template)       h_span high (searching)
  H_ans low (knows)   autopilot                   over_elaboration
  H_ans high (unsure) template_hedge              compensation

Both signals are within-model percentiles (0..1). The cell contents are a hypothesis
to be validated against human span labels; the split itself is purely mechanical.
"""

from __future__ import annotations

import pandas as pd

CELL_LABELS = {
    (False, False): "autopilot",
    (False, True): "over_elaboration",
    (True, False): "template_hedge",
    (True, True): "compensation",
}
LABEL_ORDER = ["autopilot", "over_elaboration", "template_hedge", "compensation"]


def attribute_spans(df: pd.DataFrame, h_ans_pct: str = "H_ans_pct", h_span_pct: str = "h_span_pct",
                    threshold: float = 0.5) -> pd.Series:
    """Label each (filler) span row by its 2x2 cell."""
    hi_ans = df[h_ans_pct] > threshold
    hi_span = df[h_span_pct] > threshold
    return pd.Series([CELL_LABELS[(bool(a), bool(s))] for a, s in zip(hi_ans, hi_span)],
                     index=df.index, name="filler_source")


def attribution_shares(df: pd.DataFrame, label_col: str = "filler_source", weight: str = "tokens",
                       by: str = "model") -> pd.DataFrame:
    """Token-weighted share of each filler source per group (rows sum to 1)."""
    tab = df.pivot_table(index=by, columns=label_col, values=weight, aggfunc="sum", fill_value=0)
    tab = tab.reindex(columns=[c for c in LABEL_ORDER if c in tab.columns])
    return tab.div(tab.sum(axis=1), axis=0)
