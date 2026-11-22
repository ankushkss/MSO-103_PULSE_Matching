# One PULSE simulation: python run.py [--seed N] [--set KEY=VALUE] [--out DIR]
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

from experiments import ablation_suite
from pulse import Config, Simulation, load_districts, parse_overrides

DEFAULT_DATA = "data/districts.xlsx"

COLS = [  # (header, width, value)
    ("Rnd", 3, lambda r: f"{r['round']}"),
    ("Supply", 7, lambda r: k(r["A_tot"])),
    ("Demand", 7, lambda r: k(r["D_tot"])),
    ("Planned", 7, lambda r: k(r["allocated"])),
    ("Delivered", 9, lambda r: k(r["delivered"])),
    ("Hit", 4, lambda r: f"{r['delivered'] / max(r['allocated'], 1):.0%}"),
    ("ToCore", 7, lambda r: k(r["core_allocated"])),
    ("ToExcess", 8, lambda r: k(r["excess_allocated"])),
    ("Refused", 7, lambda r: f"{r['refused_matches']}"),
]


def k(n) -> str:
    n = float(n)
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}"


def print_rounds(rdf: pd.DataFrame) -> None:
    line = "  ".join(h.rjust(w) for h, w, _ in COLS)
    print(line)
    print("-" * len(line))
    for _, r in rdf.iterrows():
        print("  ".join(f(r).rjust(w) for _, w, f in COLS))
    print("-" * len(line))
    tot = {"Planned": k(rdf["allocated"].sum()), "Delivered": k(rdf["delivered"].sum()),
           "Hit": f"{rdf['delivered'].sum() / max(rdf['allocated'].sum(), 1):.0%}",
           "ToCore": k(rdf["core_allocated"].sum()), "ToExcess": k(rdf["excess_allocated"].sum()),
           "Refused": f"{rdf['refused_matches'].sum()}"}
    print("  ".join((tot.get(h, "all" if h == "Rnd" else "")).rjust(w) for h, w, _ in COLS))


def print_matches(sim, districts, n: int) -> None:
    m = sim.matches.to_frame()
    m = m[m["tau"] == 1].sort_values("x", ascending=False).head(n)
    inst = sim.inst

    def place(k_):
        return f"{districts.names[k_]} ({districts.state_names[districts.state[k_]]})"

    print(f"\nRound 1: the {len(m)} largest matches the LP chose")
    print(f"{'Supplier district':26s} {'Requester district':26s} {'Lvl':>3} {'Units':>6} {'km':>5} {'Result':>9}")
    print("-" * 81)
    for _, r in m.iterrows():
        sp = place(inst.s_dist[int(r["sup"])])[:26]
        rq = place(inst.r_dist[int(r["req"])])[:26]
        res = "delivered" if r["delivered"] > 0 else "failed"
        print(f"{sp:26s} {rq:26s} {int(r['level']):>3} {int(r['x']):>6} {r['km']:>5.0f} {res:>9}")


def ensure_data(path: str) -> None:
    if os.path.exists(path):
        return
    sys.exit(f"data file not found: {path}\n"
             "The real district workbook ships as data/districts.xlsx. For a synthetic one, run:\n"
             "    python -m pulse.make_districts data/placeholder.xlsx")


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, help="district workbook (.xlsx)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", choices=["pulse", "shieldnet"], default="pulse",
                    help="shieldnet = the paper's original LP, on the same simulated world")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any Config field, e.g. --set c=0.1 (repeatable)")
    ap.add_argument("--out", default="results/single", help="output folder")
    ap.add_argument("--matches", action="store_true", help="also write every match to matches.csv")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print the N largest round-1 matches")
    ap.add_argument("--summary", action="store_true", help="print every summary metric")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    cfg = parse_overrides(Config(data_path=a.data), a.set)
    ensure_data(cfg.data_path)
    districts = load_districts(cfg.data_path, cfg.district_seed, cfg.r_intra_range)
    if a.model == "shieldnet":
        cfg = cfg.with_(**ablation_suite(cfg, districts.R_max)[0][2])
        if a.out == "results/single":
            a.out = "results/shieldnet"

    sim = Simulation(cfg, districts, seed=a.seed)
    rounds, summary = sim.run()

    os.makedirs(a.out, exist_ok=True)
    rdf = pd.DataFrame(rounds)
    rdf.to_csv(os.path.join(a.out, "rounds.csv"), index=False)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    if a.matches:
        sim.matches.to_frame().to_csv(os.path.join(a.out, "matches.csv"), index=False)

    name = "Shield-Net" if a.model == "shieldnet" else "PULSE"
    print(f"\n{name}  |  seed {a.seed}  |  {districts.K} districts  |  "
          f"{sim.inst.n_suppliers} suppliers  |  {sim.inst.n_requesters} requesters\n")
    print_rounds(rdf)
    print(f"\nGenuine need met       {summary['need_fill']:.0%}")
    print(f"Urgency-weighted need  {summary['wneed_fill_ref']:.0%}")
    print(f"Delivered              {summary['delivered']:,} of {summary['requested']:,} requested")
    print(f"LP: up to {int(rdf['edges'].max()):,} pairs per round, {rdf['solve_seconds'].sum():.1f} s solving in total")
    if a.show:
        print_matches(sim, districts, a.show)
    if a.summary:
        print("\nSummary")
        for key, v in summary.items():
            print(f"  {key:28s} {v:.4f}" if isinstance(v, float) else f"  {key:28s} {v}")
    print(f"\n(saved to {a.out}/)")


if __name__ == "__main__":
    main()
