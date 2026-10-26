"""Experiment suites from README Section 12.

    python experiments.py --suite ablation --seeds 5
    python experiments.py --suite sweeps   --seeds 5 --jobs 4
    python experiments.py --suite learning --seeds 5
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

KEY_METRICS = ["wfill_ref", "exp_fill_overall", "fill_nonbulk", "fill_L1", "fill_L2", "fill_L3", "fill_L3_unflagged",
               "fill_bulk", "same_state_share", "mean_km", "failed_match_share",
               "units_abandoned", "stranded_requester_rounds", "solve_seconds", "max_gap"]

_DISTRICTS = {}


def _districts(cfg: Config):
    key = (cfg.data_path, cfg.district_seed, tuple(cfg.r_intra_range))
    if key not in _DISTRICTS:
        _DISTRICTS[key] = load_districts(cfg.data_path, cfg.district_seed, cfg.r_intra_range)
    return _DISTRICTS[key]


# ---------------------------------------------------------------- suites
def ablation_suite(base: Config, R_max: float) -> list[tuple[str, str, dict]]:
    """Shield-Net (README 7.6), then one feature added per step (README 12)."""
    shieldnet = dict(use_psi_objective=False, learn_psi=False, W=(1.0, 2.0, 3.0),
                     alpha=1e-6 * R_max, beta=0.0, mu=0.0, eta=0.0, omega=0.0,
                     hoarding_control=False, price_filter=False, cert_filter=False,
                     enforce_min_shipment=False)
    steps = [
        ("0 Shield-Net", {}),
        ("1 +psi objective (fixed prior)", dict(use_psi_objective=True)),
        ("2 +psi learning", dict(learn_psi=True)),
        ("3 +urgency weights 1/10/100", dict(W=(1.0, 10.0, 100.0))),
        ("4 +price & certificate", dict(price_filter=True, cert_filter=True, alpha=base.alpha,
                                        beta=base.beta, mu=base.mu)),
        ("5 +minimum orders (Mode B)", dict(enforce_min_shipment=True)),
        ("6 +inventory", dict(eta=base.eta)),
        ("7 +hoarding control", dict(hoarding_control=True)),
        ("8 +abandonment = PULSE", dict(omega=base.omega)),
    ]
    out, acc = [], dict(shieldnet)
    for label, delta in steps:
        acc.update(delta)
        out.append((label, "ablation", dict(acc)))
    return out


def sweeps_suite(base: Config) -> list[tuple[str, str, dict]]:
    grid = {
        "supply_scale": [0.6, 1.0, 1.5],
        "p_bulk": [0.0, 0.005, 0.02],
        "c": [0.02, 0.05, 0.10],
        "nu": [0.1, 0.25, 0.5],
        "eta": [0.0, 0.5, 1.0],
        "omega": [0.0, 0.1, 0.2],
        "W": [(1.0, 10.0, 100.0), (1.0, 2.0, 3.0)],
        "enforce_min_shipment": [False, True],
    }
    out = []
    for param, values in grid.items():
        for v in values:
            label = f"{param}={'B' if v is True else 'A' if v is False else v}" \
                if param == "enforce_min_shipment" else f"{param}={v}"
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


# ---------------------------------------------------------------- running
def _init_worker():
    # Worker processes start fresh on Windows (spawn), so set logging again.
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")


def _run_one(args):
    label, group, overrides, base, seed = args
    cfg = base.with_(**overrides)
    t = time.perf_counter()
    rounds, summary = Simulation(cfg, _districts(cfg), seed=seed).run()
    tag = {"config": label, "group": group, "seed": seed}
    summary = {**tag, **summary, "wall_seconds": round(time.perf_counter() - t, 2)}
    rounds = [{**tag, **r} for r in rounds]
    return summary, rounds


def run_suite(name: str, configs, base: Config, seeds: list[int], jobs: int, out_dir: str):
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
    rodf = pd.DataFrame(rounds)
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

    show = table[["config"] + [f"{m}_mean" for m in ("wfill_ref", "fill_nonbulk", "fill_L3",
                                                     "same_state_share", "solve_seconds")]]
    show.columns = ["config", "wfill_ref", "fill_nonbulk", "fill_L3", "same_state", "solve_s"]
    print(show.round(3).to_string(index=False))
    print(f"[{name}] done in {time.perf_counter() - t0:.0f}s -> {out_dir}/{name}_*.csv\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["ablation", "sweeps", "learning", "all"], default="ablation")
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

    suites = ["ablation", "sweeps", "learning"] if a.suite == "all" else [a.suite]
    for name in suites:
        if name == "ablation":
            configs = ablation_suite(base, d.R_max)
        elif name == "sweeps":
            configs = sweeps_suite(base)
        else:
            configs = learning_suite(base)
        run_suite(name, configs, base, seeds, a.jobs, a.out)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
