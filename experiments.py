"""Experiment suites from README Section 12.

    python experiments.py --suite ablation --seeds 5
    python experiments.py --suite sweeps   --seeds 5 --jobs 4
    python experiments.py --suite learning --seeds 5
    python experiments.py --suite benchmarks --seeds 30
    python experiments.py --suite all      --seeds 20

Every configuration in a suite is run on the same seeds, so differences come from
the policy, not from different random instances. Outputs go to results/:
    <suite>_runs.csv    one row per (configuration, seed): the run summary
    <suite>_rounds.csv  one row per (configuration, seed, round)
    <suite>_table.csv   mean and standard deviation per configuration
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pulse import Config, Simulation, load_districts, parse_overrides
from pulse.make_districts import make_placeholder
from pulse.benchmarks import hindsight
from pulse.stats import paired_table

KEY_METRICS = ["wneed_fill_ref", "wneed_fill_w123", "need_fill", "wfill_ref", "exp_fill_nonbulk", "exp_fill_overall", "fill_nonbulk", "fill_L1", "fill_L2", "fill_L3", "fill_L3_honest",
               "fill_bulk", "same_state_share", "mean_km", "failed_match_share",
               "units_abandoned", "stranded_requester_rounds", "solve_seconds"]

_DISTRICTS = {}


def _districts(cfg: Config):
    key = (cfg.data_path, cfg.district_seed, tuple(cfg.r_intra_range))
    if key not in _DISTRICTS:
        _DISTRICTS[key] = load_districts(cfg.data_path, cfg.district_seed, cfg.r_intra_range)
    return _DISTRICTS[key]


# ---------------------------------------------------------------- suites
def ablation_suite(base: Config, R_max: float) -> list[tuple[str, str, dict]]:
    """Shield-Net (README 7.4), then one model feature added per step (README 13).

    Every step runs in the SAME simulated world as full PULSE (carried stock,
    abandonment, random match outcomes, refusals), so the steps differ only in
    the platform's decisions.
    """
    shieldnet = dict(use_psi_objective=False, learn_psi=False, W=(1.0, 2.0, 3.0),
                     alpha=1e-6 * R_max, beta=0.0, mu=0.0,
                     core_excess=False, price_filter=False, cert_filter=False)
    steps = [
        ("0 Shield-Net", {}),
        ("1 +expected-delivery objective", dict(use_psi_objective=True)),
        ("2 +learning psi", dict(learn_psi=True)),
        ("3 +urgency weights 1/10/100", dict(W=(1.0, 10.0, 100.0))),
        ("4 +price & certificate", dict(price_filter=True, cert_filter=True, alpha=base.alpha,
                                        beta=base.beta, mu=base.mu)),
        ("5 +core/excess split = PULSE", dict(core_excess=True)),
    ]
    out, acc = [], dict(shieldnet)
    for label, delta in steps:
        acc.update(delta)
        out.append((label, "ablation", dict(acc)))
    # Fair baseline (referee point A2): Shield-Net that also drops pairs the requester
    # would refuse. Separates the gain from filtering out sure refusals from the gain
    # from PULSE's objective. Not cumulative, so it sits outside the ladder.
    out.insert(1, ("0b Shield-Net + price & cert filters", "ablation",
                   dict(shieldnet, price_filter=True, cert_filter=True)))
    return out


def sweeps_suite(base: Config) -> list[tuple[str, str, dict]]:
    grid = {
        "supply_scale": [0.6, 1.0, 1.5],
        "p_bulk": [0.0, 0.005, 0.02],
        "H": [7.0, 14.0, 28.0],
        "eta": [0.0, 0.5, 1.0],
        "omega": [0.0, 0.1, 0.2],
        "W": [(1.0, 10.0, 100.0), (1.0, 2.0, 3.0)],
    }
    out = []
    for param, values in grid.items():
        for v in values:
            label = f"{param}={v}"
            out.append((label, param, {param: v}))
    return out


def learning_suite(base: Config) -> list[tuple[str, str, dict]]:
    out = []
    for ps in (0.30, 0.40, 0.55):
        for pc in (0.08, 0.16, 0.25):
            for learn in (True, False):
                out.append((f"true=({ps},{pc}) learn={'on' if learn else 'off'}",
                            f"true=({ps},{pc})",
                            dict(psi_true_same=ps, psi_true_cross=pc, learn_psi=learn)))
    return out


HINDSIGHT = "H1 clairvoyant fluid bound (knows arrivals)"
HINDSIGHT_REALISED = "H2 perfect hindsight (knows arrivals + outcomes)"
PULSE_LABEL = "PULSE"


def benchmarks_suite(base: Config, R_max: float) -> list[tuple[str, str, dict]]:
    """PULSE against an upper bound and simple allocation rules (CHANGES.md, session 2)."""
    fair_sn = [c for c in ablation_suite(base, R_max) if c[0].startswith("0b")][0][2]
    return [
        (HINDSIGHT_REALISED, "bound", {"_hindsight": "realised"}),
        (HINDSIGHT, "bound", {"_hindsight": "expected"}),
        (PULSE_LABEL, "lp", {}),
        ("Shield-Net + filters", "lp", fair_sn),
        ("Proportional rationing", "heuristic", dict(policy="proportional")),
        ("FCFS (nearest supplier)", "heuristic", dict(policy="fcfs")),
        ("Random", "heuristic", dict(policy="random")),
    ]


# ---------------------------------------------------------------- comparisons
# Each suite gets the reference that answers its own question (CHANGES.md D4).
SHIELDNET, FAIR_SHIELDNET = "0 Shield-Net", "0b Shield-Net + price & cert filters"


def ablation_comparisons(configs) -> list[tuple[str, str, str]]:
    labels = [c[0] for c in configs]
    ladder = [l for l in labels if l != FAIR_SHIELDNET]
    out = [("vs_shieldnet", l, SHIELDNET) for l in labels if l != SHIELDNET]
    out += [("vs_fair_shieldnet", l, FAIR_SHIELDNET) for l in ladder if l != SHIELDNET]
    out += [("vs_previous_step", b, a) for a, b in zip(ladder, ladder[1:])]
    return out


def sweep_comparisons(configs, base: Config) -> list[tuple[str, str, str]]:
    out = []
    for group in dict.fromkeys(g for _, g, _ in configs):
        members = [(l, ov) for l, g, ov in configs if g == group]
        ref = [l for l, ov in members if ov[group] == getattr(base, group)]
        if not ref:
            continue                     # base was overridden to a value outside the grid
        out += [("vs_default", l, ref[0]) for l, _ in members if l != ref[0]]
    return out


def benchmark_comparisons(configs) -> list[tuple[str, str, str]]:
    return [("vs_pulse", l, PULSE_LABEL) for l, _, _ in configs if l != PULSE_LABEL]


def hindsight_gap(rdf: pd.DataFrame, metrics=("need_fill", "wneed_fill_ref", "wneed_fill_w123")):
    """Per config: mean metric over seeds / mean bound over seeds, for each bound.

    Ratio of means, not mean of per-seed ratios: the fluid bound H1 limits EXPECTED
    performance, so it is only meaningful against an average (CHANGES.md D12).
    """
    rows = []
    bounds = [b for b in (HINDSIGHT, HINDSIGHT_REALISED) if (rdf["config"] == b).any()]
    for config, g in rdf.groupby("config", sort=False):
        row = {"config": config, "n": int(g["seed"].nunique())}
        for b in bounds:
            hs = rdf[rdf["config"] == b]
            tag = "H1" if b == HINDSIGHT else "H2"
            for m in metrics:
                row[f"{m}_pct_of_{tag}"] = float(g[m].mean() / hs[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def learning_comparisons(configs) -> list[tuple[str, str, str]]:
    out = []
    for group in dict.fromkeys(g for _, g, _ in configs):
        on = [l for l, g, ov in configs if g == group and ov["learn_psi"]]
        off = [l for l, g, ov in configs if g == group and not ov["learn_psi"]]
        if on and off:
            out.append(("learn_on_vs_off", on[0], off[0]))
    return out


# ---------------------------------------------------------------- running
def _init_worker():
    # Worker processes start fresh on Windows (spawn), so set logging again.
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")


def _run_one(args):
    label, group, overrides, base, seed = args
    overrides = dict(overrides)
    is_bound = overrides.pop("_hindsight", False)
    cfg = base.with_(**overrides)
    t = time.perf_counter()
    if is_bound:
        rounds, summary = hindsight(cfg, _districts(cfg), seed, outcomes=is_bound)
    else:
        rounds, summary = Simulation(cfg, _districts(cfg), seed=seed).run()
    tag = {"config": label, "group": group, "seed": seed}
    summary = {**tag, **summary, "wall_seconds": round(time.perf_counter() - t, 2)}
    rounds = [{**tag, **r} for r in rounds]
    return summary, rounds


def run_suite(name: str, configs, base: Config, seeds: list[int], jobs: int, out_dir: str,
              comparisons=()):
    tasks = [(label, group, ov, base, s) for (label, group, ov) in configs for s in seeds]
    print(f"[{name}] {len(configs)} configurations x {len(seeds)} seeds = {len(tasks)} runs, {jobs} jobs")
    runs, rounds = [], []
    t0 = time.perf_counter()
    if jobs <= 1:
        for k, task in enumerate(tasks, 1):
            s, r = _run_one(task)
            runs.append(s)
            rounds.extend(r)
            print(f"  {k}/{len(tasks)}  {s['config']}  seed {s['seed']}  ({s['wall_seconds']}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
            futs = [ex.submit(_run_one, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                s, r = f.result()
                runs.append(s)
                rounds.extend(r)
                print(f"  {k}/{len(tasks)}  {s['config']}  seed {s['seed']}  ({s['wall_seconds']}s)", flush=True)

    order = {label: i for i, (label, _, _) in enumerate(configs)}
    rdf = pd.DataFrame(runs)
    rdf["_o"] = rdf["config"].map(order)
    rdf = rdf.sort_values(["_o", "seed"]).drop(columns="_o")
    rodf = pd.DataFrame(rounds, columns=None if rounds else ["config", "seed", "round"])
    rodf["_o"] = rodf["config"].map(order)
    rodf = rodf.sort_values(["_o", "seed", "round"]).drop(columns="_o")

    metrics = [m for m in KEY_METRICS if m in rdf.columns]
    table = rdf.groupby(["config", "group"], sort=False)[metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.reset_index()

    os.makedirs(out_dir, exist_ok=True)
    rdf.to_csv(os.path.join(out_dir, f"{name}_runs.csv"), index=False)
    rodf.to_csv(os.path.join(out_dir, f"{name}_rounds.csv"), index=False)
    table.to_csv(os.path.join(out_dir, f"{name}_table.csv"), index=False)
    if comparisons and len(seeds) >= 2:
        paired = paired_table(rdf, comparisons, metrics)
        paired.to_csv(os.path.join(out_dir, f"{name}_paired.csv"), index=False)
        _print_paired(paired)

    if rdf["config"].isin([HINDSIGHT, HINDSIGHT_REALISED]).any():
        gap = hindsight_gap(rdf)
        gap.to_csv(os.path.join(out_dir, f"{name}_gap.csv"), index=False)
        print("share of the hindsight upper bound achieved (mean over seeds):")
        print(gap.round(3).to_string(index=False))

    cols = [m for m in ("need_fill", "wneed_fill_ref", "fill_nonbulk", "fill_L3",
                        "same_state_share", "solve_seconds") if f"{m}_mean" in table.columns]
    show = table[["config"] + [f"{m}_mean" for m in cols]]
    show.columns = ["config"] + cols
    print(show.round(3).to_string(index=False))
    print(f"[{name}] done in {time.perf_counter() - t0:.0f}s -> {out_dir}/{name}_*.csv\n")


def _print_paired(paired: pd.DataFrame) -> None:
    show = paired[paired["primary"] & paired["metric"].isin(["need_fill", "wneed_fill_ref"])]
    if show.empty:
        return
    show = show[["comparison", "config", "metric", "mean_diff", "ci_low", "ci_high",
                 "win_share", "p_holm"]]
    print("paired differences (config - reference), 95% CI, Holm-adjusted Wilcoxon p:")
    print(show.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["ablation", "sweeps", "learning", "benchmarks", "all"], default="ablation")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (1..N)")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="parallel worker processes")
    ap.add_argument("--data", default="data/districts.xlsx")
    ap.add_argument("--out", default="results")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override the base configuration (applies to every run)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")

    base = parse_overrides(Config(data_path=a.data), a.set)
    if not os.path.exists(base.data_path):
        if os.path.normpath(base.data_path) != os.path.normpath("data/districts.xlsx"):
            sys.exit(f"data file not found: {base.data_path}")
        print("generating the 800-district placeholder...")
        make_placeholder(base.data_path)
    d = _districts(base)          # parse once (and write the cache) before workers start
    seeds = list(range(1, a.seeds + 1))

    suites = ["ablation", "sweeps", "learning", "benchmarks"] if a.suite == "all" else [a.suite]
    for name in suites:
        if name == "ablation":
            configs = ablation_suite(base, d.R_max)
            comps = ablation_comparisons(configs)
        elif name == "sweeps":
            configs = sweeps_suite(base)
            comps = sweep_comparisons(configs, base)
        elif name == "benchmarks":
            configs = benchmarks_suite(base, d.R_max)
            comps = benchmark_comparisons(configs)
        else:
            configs = learning_suite(base)
            comps = learning_comparisons(configs)
        run_suite(name, configs, base, seeds, a.jobs, a.out, comps)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
