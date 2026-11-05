"""Paired statistics for comparing configurations on common random numbers.

Every configuration in a suite runs on the same seeds, and match outcomes and
abandonment use pre-drawn uniforms (README Section 13). The per-seed difference
between two configurations therefore isolates the effect of the policy, so all
comparisons here are PAIRED on seed.

See CHANGES.md (decisions D3-D7) for why each choice was made.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

# Holm correction is applied across these metrics within one comparison (D6).
# Other metrics are still reported, but as secondary (p_holm left blank).
PRIMARY_METRICS = ("need_fill", "wneed_fill_ref", "wneed_fill_w123", "fill_L3_honest",
                   "exp_fill_nonbulk")

N_BOOT = 5000


def paired_stats(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Summarise per-seed paired differences (config minus reference) on one metric."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    nan = float("nan")
    out = {"n": n, "mean_diff": nan, "sd_diff": nan, "ci_low": nan, "ci_high": nan,
           "boot_low": nan, "boot_high": nan, "p_wilcoxon": nan, "p_ttest": nan,
           "d_z": nan, "win_share": nan, "loss_share": nan}
    if n == 0:
        return out
    mean = float(diff.mean())
    out.update(mean_diff=mean,
               win_share=float((diff > 0).mean()),
               loss_share=float((diff < 0).mean()))
    if n < 2:
        return out

    sd = float(diff.std(ddof=1))
    out["sd_diff"] = sd
    if sd == 0.0:
        # every seed moved by exactly the same amount (usually 0: metric untouched)
        out.update(ci_low=mean, ci_high=mean, boot_low=mean, boot_high=mean,
                   p_wilcoxon=1.0 if mean == 0 else nan, p_ttest=1.0 if mean == 0 else nan,
                   d_z=0.0 if mean == 0 else float(np.sign(mean) * np.inf))
        return out

    half = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    out.update(ci_low=mean - half, ci_high=mean + half, d_z=mean / sd,
               p_ttest=float(sps.ttest_1samp(diff, 0.0).pvalue))

    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out["boot_low"], out["boot_high"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))

    if np.any(diff != 0):
        try:
            out["p_wilcoxon"] = float(sps.wilcoxon(diff).pvalue)
        except ValueError:          # too few non-zero differences
            pass
    return out


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values; NaNs stay NaN and are not counted."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m == 0:
        return adj
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def paired_table(runs: pd.DataFrame, comparisons, metrics) -> pd.DataFrame:
    """One row per (comparison, metric).

    comparisons: iterable of (kind, config, reference) where both labels are in runs["config"].
    Only seeds present for both configurations are used.
    """
    by_cfg = {c: g.set_index("seed") for c, g in runs.groupby("config", sort=False)}
    rows = []
    for kind, config, reference in comparisons:
        if config not in by_cfg or reference not in by_cfg:
            continue
        a, b = by_cfg[config], by_cfg[reference]
        seeds = a.index.intersection(b.index)
        block = []
        for m in metrics:
            if m not in a.columns:
                continue
            d = (a.loc[seeds, m] - b.loc[seeds, m]).to_numpy(dtype=float)
            block.append({"comparison": kind, "config": config, "reference": reference,
                          "metric": m, "primary": m in PRIMARY_METRICS,
                          "ref_mean": float(b.loc[seeds, m].mean()),
                          **paired_stats(d)})
        prim = [r for r in block if r["primary"]]
        for r, p in zip(prim, holm([r["p_wilcoxon"] for r in prim])):
            r["p_holm"] = p
        rows.extend(block)
    df = pd.DataFrame(rows)
    if "p_holm" not in df.columns:
        df["p_holm"] = np.nan
    return df
