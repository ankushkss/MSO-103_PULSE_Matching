"""Figures from experiment results.

    python plots.py                 # every suite found in results/
    python plots.py --results results --out results/figures

Reads results/<suite>_runs.csv and results/<suite>_rounds.csv written by
experiments.py, and results/single/rounds.csv written by run.py.
Error bars are one standard error of the mean across seeds.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# categorical slots in fixed order (blue, orange, aqua); text stays in ink colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _sem(s):
    return s.std(ddof=1) / np.sqrt(max(1, s.count())) if s.count() > 1 else 0.0


def _agg(df, by, metrics):
    g = df.groupby(by, sort=False)
    return g[metrics].mean(), g[metrics].agg(_sem)


def plot_ablation(runs: pd.DataFrame, out: str) -> str:
    metrics = [("wfill_ref", "urgency-weighted fill"), ("fill_nonbulk", "non-bulk fill"),
               ("fill_L3_unflagged", "level-3 fill (unflagged)")]
    mean, err = _agg(runs, "config", [m for m, _ in metrics])
    labels = list(mean.index)
    y = np.arange(len(labels))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(labels) + 1.2))
    for k, (m, name) in enumerate(metrics):
        ax.barh(y + (k - 1) * h, mean[m], height=h - 0.03, xerr=err[m], color=SERIES[k],
                label=name, error_kw=dict(ecolor=INK_2, lw=0.8, capsize=2))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("share of requested units")
    ax.set_xlim(0, 1)
    ax.set_title("Ablation: Shield-Net to PULSE, one feature at a time", loc="left", color=INK, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    path = os.path.join(out, "ablation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sweeps(runs: pd.DataFrame, out: str) -> str:
    groups = list(dict.fromkeys(runs["group"]))
    n = len(groups)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), squeeze=False)
    for ax, g in zip(axes.flat, groups):
        sub = runs[runs["group"] == g]
        mean, err = _agg(sub, "config", ["wfill_ref", "fill_nonbulk"])
        labels = [c.split("=", 1)[1] for c in mean.index]
        x = np.arange(len(labels))
        for k, (m, name) in enumerate([("wfill_ref", "urgency-weighted fill"),
                                       ("fill_nonbulk", "non-bulk fill")]):
            ax.errorbar(x, mean[m], yerr=err[m], color=SERIES[k], lw=2, marker="o", ms=5,
                        capsize=2, label=name)
        ax.set_xticks(x, labels)
        ax.set_title(g, loc="left", color=INK)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="lower left")
    fig.suptitle("Sensitivity sweeps (base: PULSE defaults)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    path = os.path.join(out, "sweeps.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning(runs: pd.DataFrame, rounds: pd.DataFrame, out: str) -> list[str]:
    paths = []
    runs = runs.copy()
    runs["learn"] = runs["config"].str.endswith("learn=on")
    mean, err = _agg(runs, ["group", "learn"], ["exp_fill_nonbulk"])
    groups = list(dict.fromkeys(runs["group"]))
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, learn in enumerate([False, True]):
        m = [mean.loc[(g, learn), "exp_fill_nonbulk"] for g in groups]
        e = [err.loc[(g, learn), "exp_fill_nonbulk"] for g in groups]
        ax.bar(x + (k - 0.5) * 0.38, m, width=0.36, yerr=e, color=SERIES[k],
               label="learning on" if learn else "learning off (prior only)",
               error_kw=dict(ecolor=INK_2, lw=0.8, capsize=2))
    ax.set_xticks(x, [g.replace("true=", "") for g in groups], rotation=30, ha="right")
    ax.set_xlabel("true (ψ same, ψ cross); prior is (0.40, 0.16)")
    ax.set_ylabel("expected fill")
    ax.set_title("Value of learning ψ", loc="left", color=INK)
    ax.legend(loc="upper left")
    p = os.path.join(out, "learning_value.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    on = rounds[rounds["config"].str.endswith("learn=on")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    for ax, col, name in ((axes[0], "psi_hat_same", r"$\hat\psi$ same state"),
                          (axes[1], "psi_hat_cross", r"$\hat\psi$ cross state")):
        for g in groups:
            sub = on[on["group"] == g].groupby("round")[col].mean()
            ax.plot(sub.index, sub.values, color=SERIES[0], alpha=0.35, lw=1.2)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("round")
    axes[0].set_ylabel("belief at start of round")
    fig.suptitle("How beliefs move from the prior (one line per true ψ)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    p = os.path.join(out, "learning_trajectories.png")
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def plot_single(rounds: pd.DataFrame, out: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    x = rounds["round"]
    ax = axes[0]
    for k, (col, name) in enumerate([("A_tot", "available supply"), ("D_tot", "outstanding demand")]):
        ax.plot(x, rounds[col] / 1000, color=SERIES[k], lw=2, marker="o", ms=5, label=name)
    ax.set_ylabel("thousand units")
    ax.set_xlabel("round")
    ax.set_title("Supply and demand", loc="left", color=INK)
    ax.legend(loc="upper left")
    ax = axes[1]
    for k, (col, name) in enumerate([("allocated", "allocated"), ("delivered", "delivered")]):
        ax.plot(x, rounds[col] / 1000, color=SERIES[k], lw=2, marker="o", ms=5, label=name)
    ax.set_xlabel("round")
    ax.set_title("Allocated vs delivered", loc="left", color=INK)
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = os.path.join(out, "single_run.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None, help="default: <results>/figures")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "figures")
    os.makedirs(out, exist_ok=True)
    made = []

    def load(name):
        p = os.path.join(a.results, f"{name}_runs.csv")
        q = os.path.join(a.results, f"{name}_rounds.csv")
        if os.path.exists(p):
            return pd.read_csv(p), (pd.read_csv(q) if os.path.exists(q) else None)
        return None, None

    runs, _ = load("ablation")
    if runs is not None:
        made.append(plot_ablation(runs, out))
    runs, _ = load("sweeps")
    if runs is not None:
        made.append(plot_sweeps(runs, out))
    runs, rounds = load("learning")
    if runs is not None and rounds is not None:
        made.extend(plot_learning(runs, rounds, out))
    single = os.path.join(a.results, "single", "rounds.csv")
    if os.path.exists(single):
        made.append(plot_single(pd.read_csv(single), out))
    if not made:
        print(f"no results found in {a.results}/ (run run.py or experiments.py first)")
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
