import numpy as np
import pandas as pd
import pytest

from drybench.baselines import length_chars, length_words, stopword_share, yap_excess_chars
from drybench.distill import select_targets, selection_summary
from drybench.gate import evaluate_gate, gate_curve
from drybench.profile import ols_coef, profiles


def _planted(n=400, beta=0.4, gamma=9.0, seed=0, model="m"):
    rng = np.random.default_rng(seed)
    k = rng.integers(1, 7, n)
    H = rng.random(n)
    ans = rng.random(n) > 0.1
    amb = rng.random(n) > 0.95
    hops = np.where(rng.random(n) > 0.9, 2, 1)
    W = np.clip(0.1 + beta * H + rng.normal(0, 0.03, n), 0, 0.95)
    kept = gamma * k + rng.normal(0, 1.0, n) + 5
    tokens = kept / (1 - W)
    return pd.DataFrame({"model": model, "item_id": [f"i{i}" for i in range(n)], "k": k,
                         "answerable": ans, "ambiguous": amb, "hops": hops, "H_ans": H,
                         "W": W, "tokens": tokens, "tokens_kept": kept, "kappa": 1.0})


def test_ols_recovers_coefficients_and_drops_constants():
    rng = np.random.default_rng(1)
    x, z = rng.random(200), rng.random(200)
    y = 2 + 3 * x - z + rng.normal(0, 0.01, 200)
    assert ols_coef(y, {"x": x, "z": z}, "x") == pytest.approx(3, abs=0.02)
    assert ols_coef(y, {"x": x, "c": np.ones(200)}, "x") == pytest.approx(3, abs=0.2)
    assert np.isnan(ols_coef(y, {"c": np.ones(200)}, "c"))


def test_profile_recovers_planted_beta_and_gamma():
    df = pd.concat([_planted(model="a", beta=0.4), _planted(model="b", beta=0.0, seed=1)])
    prof = profiles(df, n_boot=100)
    # H_pct is a monotone transform of uniform H, so slope on H_pct ~ planted beta
    assert prof.loc["a", "beta"] == pytest.approx(0.4, abs=0.05)
    assert abs(prof.loc["b", "beta"]) < 0.03
    assert prof.loc["a", "beta_lo"] <= prof.loc["a", "beta"] <= prof.loc["a", "beta_hi"]
    assert prof.loc["a", "beta_lo"] > 0.3 and prof.loc["a", "beta_hi"] < 0.5
    assert prof.loc["b", "beta_lo"] < 0 < prof.loc["b", "beta_hi"]
    assert prof.loc["a", "gamma"] == pytest.approx(9.0, abs=0.3)
    assert prof.loc["a", "alpha"] < prof.loc["a", "W_mean"]
    with pytest.raises(ValueError):
        profiles(df.drop(columns=["kappa"]))


def test_gate_tradeoff():
    df = pd.DataFrame({"H_ans": [0.1, 0.2, 2.0, 2.5], "tokens": [50, 60, 300, 400],
                       "kappa": [1.0, 1.0, 0.2, 0.0], "fb_tokens": [10, 10, 10, 10],
                       "fb_kappa": [0.0, 0.0, 1.0, 1.0]})
    never = evaluate_gate(df, float("inf"))
    gated = evaluate_gate(df, 1.0)
    assert never["frac_gated"] == 0 and gated["frac_gated"] == 0.5
    assert gated["tokens_mean"] < never["tokens_mean"]
    assert gated["cost_of_pass"] < never["cost_of_pass"]
    curve = gate_curve(df)
    assert curve["tau"].iloc[-1] == float("inf") and len(curve) >= 2


def test_select_targets_prefers_dry_complete_samples_on_easy_bins():
    s = pd.DataFrame({
        "prompt_id": ["p1"] * 3 + ["p2"] * 3,
        "H_bin": ["low"] * 3 + ["high"] * 3,
        "W": [0.6, 0.1, 0.0, 0.5, 0.1, 0.0],
        "kappa": [1.0, 0.95, 0.5, 1.0, 0.7, 0.6],
        "tokens": [100, 40, 20, 300, 90, 50],
    })
    sel = select_targets(s)
    assert sel.set_index("prompt_id").loc["p1", "tokens"] == 40   # driest with kappa >= 0.9
    assert sel.set_index("prompt_id").loc["p2", "tokens"] == 300  # hard: most complete
    summ = selection_summary(sel, s)
    assert set(summ.index) == {"selected", "first_sample", "all_samples"}


def test_baselines():
    assert length_chars(" abc ") == 3 and length_words("It is Port Vel-ro.") == 4
    assert yap_excess_chars("Paris, of course!", "Paris") == 12
    assert yap_excess_chars("Par", "Paris") == 0
    assert stopword_share("the of and") == 1.0 and stopword_share("Ostrell Freight") == 0.0


def test_judge_filler_expected_value():
    from drybench.backends.mock import MockBackend
    from drybench.baselines import judge_filler

    # every option equally likely -> expected value 0.5
    flat = MockBackend(p_present=1e-4, p_absent=1e-4)
    assert judge_filler(flat, "req", "Great question! Paris.") == pytest.approx(0.5)
    # an option echoed in the prompt ("30%" in the instruction) is favoured by the copy-mock
    assert judge_filler(MockBackend(), "req", "Paris.") == pytest.approx(0.3, abs=0.01)


def test_gamma_ignores_special_designs():
    """Special items have k=1 but long, reasoned answers; they must not bend the k-slope."""
    df = _planted(model="a", beta=0.0, gamma=7.0, seed=3)
    special = ~df["answerable"] | df["ambiguous"] | (df["hops"] == 2)
    df.loc[special, "k"] = 1
    df.loc[special, "tokens_kept"] += 60  # e.g. step-by-step reasoning on 2-hop questions
    df.loc[special, "tokens"] += 90
    prof = profiles(df, n_boot=50)
    assert prof.loc["a", "gamma"] == pytest.approx(7.0, abs=0.3)
