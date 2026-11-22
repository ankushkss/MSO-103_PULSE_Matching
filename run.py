"""Run one PULSE simulation and print what happened round by round.

Examples:
    python run.py                         # seed 1, placeholder data
    python run.py --seed 3
    python run.py --data path/to/real.xlsx --set c=0.1 --set "W=(1,2,3)"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from pulse import Config, Simulation, load_districts, parse_overrides
from pulse.make_districts import make_placeholder

DEFAULT_DATA = "data/districts.xlsx"

ROUND_COLS = ["round", "day", "suppliers_active", "requesters_active", "A_tot", "D_tot",
              "supply_demand_ratio", "edges", "allocated", "delivered", "failed_matches",
              "core_demand", "excess_demand", "core_allocated", "excess_allocated", "stranded",
              "psi_hat_same", "psi_hat_cross",
              "solve_seconds"]


def ensure_data(path: str) -> None:
    if os.path.exists(path):
        return
    if os.path.normpath(path) != os.path.normpath(DEFAULT_DATA):
        sys.exit(f"data file not found: {path}")
    print(f"{path} not found; generating the 800-district placeholder (takes ~10 s)...")
    make_placeholder(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, help="district workbook (README 2.1)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any Config field, e.g. --set c=0.1 (repeatable)")
    ap.add_argument("--out", default="results/single", help="output folder")
    ap.add_argument("--matches", action="store_true", help="also write every match to matches.csv")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    cfg = parse_overrides(Config(data_path=a.data), a.set)
    ensure_data(cfg.data_path)
    districts = load_districts(cfg.data_path, cfg.district_seed, cfg.r_intra_range)

    sim = Simulation(cfg, districts, seed=a.seed)
    rounds, summary = sim.run()

    os.makedirs(a.out, exist_ok=True)
    rdf = pd.DataFrame(rounds)
    rdf.to_csv(os.path.join(a.out, "rounds.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    if a.matches:
        sim.matches.to_frame().to_csv(os.path.join(a.out, "matches.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\nPULSE  seed {a.seed}  districts {districts.K}  "
          f"suppliers {sim.inst.n_suppliers}  requesters {sim.inst.n_requesters}\n")
    print(rdf[ROUND_COLS].round(3).to_string(index=False))
    print("\nSummary")
    for k, v in summary.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"\nwrote {a.out}/rounds.csv and {a.out}/summary.json")


if __name__ == "__main__":
    main()
