# Shield-Net vs PULSE on the same seed: python compare.py [--seed N]
from __future__ import annotations

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import ablation_suite
from plots import INK, INK_2, SERIES
from pulse import Config, Simulation, load_districts, parse_overrides

GREY = "#9a9893"


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data/districts.xlsx")
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override the base configuration for both models")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    base = parse_overrides(Config(data_path=a.data), a.set)
    d = load_districts(base.data_path, base.district_seed, base.r_intra_range)
    steps = ablation_suite(base, d.R_max)
    models = {"Shield-Net": steps[0][2], "PULSE": steps[-1][2]}

    rounds, summ = {}, {}
    for name, ov in models.items():
        print(f"running {name} (seed {a.seed})...")
        r, s = Simulation(base.with_(**ov), d, seed=a.seed).run()
        rounds[name], summ[name] = pd.DataFrame(r), s

    sn, pu = rounds["Shield-Net"], rounds["PULSE"]
    cols = {"Shield-Net": GREY, "PULSE": SERIES[0]}
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # cumulative deliveries
    a1 = ax[0, 0]
    for name, df in rounds.items():
        a1.plot(df["round"], df["delivered"].cumsum() / 1000, marker="o", lw=2, ms=5, color=cols[name], label=name)
    a1.set_title("Units actually delivered (cumulative)", loc="left", color=INK)
    a1.set_xlabel("round"); a1.set_ylabel("thousand units"); a1.legend(loc="upper left")

    # allocated vs delivered: where the shipments go
    a2 = ax[0, 1]
    names = list(rounds)
    alloc = [rounds[n]["allocated"].sum() / 1000 for n in names]
    deliv = [rounds[n]["delivered"].sum() / 1000 for n in names]
    y = np.arange(len(names))
    a2.barh(y - 0.2, alloc, height=0.38, color="#d9d7d1")
    a2.barh(y + 0.2, deliv, height=0.38, color=[cols[n] for n in names])
    for k in range(len(names)):
        a2.text(alloc[k] + 20, y[k] - 0.2, f"planned {alloc[k]:,.0f}k", va="center", color=INK_2, fontsize=8)
        a2.text(deliv[k] + 20, y[k] + 0.2, f"delivered {deliv[k]:,.0f}k ({deliv[k] / alloc[k]:.0%} of plan)",
                va="center", color=INK, fontsize=8)
    a2.set_yticks(y, names); a2.invert_yaxis()
    a2.set_xlim(0, max(alloc) * 1.3)
    a2.set_xlabel("thousand units, all rounds")
    a2.set_title("Plans vs outcomes", loc="left", color=INK)

    # headline metrics
    a3 = ax[1, 0]
    mets = [("wneed_fill_ref", "urgency-weighted\nneed met"), ("need_fill", "genuine\nneed met"),
            ("fill_L3", "level-3\nfill"), ("exp_fill_nonbulk", "expected fill\n(non-bulk)")]
    x = np.arange(len(mets)); w = 0.38
    for k, n in enumerate(names):
        vals = [summ[n][m] for m, _ in mets]
        b = a3.bar(x + (k - 0.5) * w, vals, width=w - 0.03, color=cols[n], label=n)
        a3.bar_label(b, fmt="%.2f", fontsize=7.5, color=INK_2, padding=2)
    a3.set_xticks(x, [lab for _, lab in mets]); a3.set_ylim(0, 1.05)
    a3.set_title("How much real need gets met", loc="left", color=INK)
    a3.legend(loc="upper right")

    # waste
    a4 = ax[1, 1]
    labels = ["units shipped in\nmatches that failed", "units to over-askers\nbeyond their need"]
    x = np.arange(len(labels))
    for k, n in enumerate(names):
        v = [rounds[n]["failed_units"].sum(), summ[n]["excess_received"]]
        b = a4.bar(x + (k - 0.5) * w, [t / 1000 for t in v], width=w - 0.03, color=cols[n], label=n)
        a4.bar_label(b, labels=[f"{t / 1000:,.0f}k" for t in v], fontsize=7.5, color=INK_2, padding=2)
    a4.set_xticks(x, labels); a4.set_ylabel("thousand units")
    ref = {n: int(rounds[n]["refused_matches"].sum()) for n in names}
    a4.set_title(f"Waste (refused matches: Shield-Net {ref['Shield-Net']:,}, PULSE {ref['PULSE']:,})",
                 loc="left", color=INK)
    a4.legend(loc="upper right")

    fig.suptitle(f"Shield-Net vs PULSE, same world (seed {a.seed})", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    png = os.path.join(a.out, f"compare_seed{a.seed}.png")
    fig.savefig(png); plt.close(fig)

    keys = ["delivered", "wneed_fill_ref", "need_fill", "fill_L3", "failed_match_share", "excess_received",
            "units_abandoned", "same_state_share", "mean_km"]
    table = pd.DataFrame({n: [summ[n][k] for k in keys] for n in names}, index=keys)
    table.loc["allocated"] = [rounds[n]["allocated"].sum() for n in names]
    table.loc["refused_matches"] = [rounds[n]["refused_matches"].sum() for n in names]
    csv = os.path.join(a.out, f"compare_seed{a.seed}.csv")
    table.to_csv(csv)
    rows = [
        ("Units delivered", lambda n: f"{summ[n]['delivered']:,}"),
        ("Share of plan that arrived", lambda n: f"{rounds[n]['delivered'].sum() / rounds[n]['allocated'].sum():.0%}"),
        ("Genuine need met", lambda n: f"{summ[n]['need_fill']:.0%}"),
        ("Urgency-weighted need met", lambda n: f"{summ[n]['wneed_fill_ref']:.0%}"),
        ("Level-3 (hospital) fill", lambda n: f"{summ[n]['fill_L3']:.0%}"),
        ("Refused matches", lambda n: f"{int(rounds[n]['refused_matches'].sum()):,}"),
        ("Units to over-askers beyond need", lambda n: f"{summ[n]['excess_received']:,}"),
    ]
    print(f"\n{'':34s}{'Shield-Net':>12s}{'PULSE':>12s}")
    print("-" * 58)
    for label, f_ in rows:
        print(f"{label:34s}{f_('Shield-Net'):>12s}{f_('PULSE'):>12s}")
    print()
    print("wrote", png, "and", csv)


if __name__ == "__main__":
    main()
