import numpy as np
import pandas as pd
import pytest

from pulse.stats import holm, paired_stats, paired_table


def test_known_differences_give_t_interval():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    s = paired_stats(d)
    assert s["n"] == 4 and s["mean_diff"] == pytest.approx(0.25)
    half = 3.182446305284263 * d.std(ddof=1) / 2          # t_{0.975,3} * sd / sqrt(4)
    assert s["ci_low"] == pytest.approx(0.25 - half)
    assert s["ci_high"] == pytest.approx(0.25 + half)
    assert s["win_share"] == 1.0 and s["d_z"] > 0
    assert s["boot_low"] <= s["mean_diff"] <= s["boot_high"]


def test_all_zero_differences_do_not_crash():
    s = paired_stats(np.zeros(10))
    assert s["mean_diff"] == 0 and s["p_wilcoxon"] == 1.0 and s["ci_low"] == s["ci_high"] == 0


def test_too_few_seeds_return_nan():
    s = paired_stats(np.array([0.3]))
    assert s["n"] == 1 and np.isnan(s["ci_low"]) and np.isnan(s["p_wilcoxon"])
    assert paired_stats(np.array([np.nan, np.nan]))["n"] == 0


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    # sorted 0.01,0.03,0.04 -> 3*0.01, max(.03, 2*.03), max(.06, 1*.04)
    np.testing.assert_allclose(adj[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adj[3])


def test_paired_table_pairs_on_seed_not_row_order():
    runs = pd.DataFrame({
        "config": ["A"] * 3 + ["B"] * 3,
        "seed": [1, 2, 3, 3, 1, 2],                       # B listed in a different order
        "need_fill": [0.25, 0.5, 0.75, 1.0, 0.5, 0.75],   # B - A = +0.25 on every seed
    })
    t = paired_table(runs, [("x", "B", "A")], ["need_fill"])
    row = t.iloc[0]
    assert row["mean_diff"] == 0.25 and row["sd_diff"] == 0
    assert row["primary"] and row["win_share"] == 1.0
