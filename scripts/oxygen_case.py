"""Oxygen 2021 case study: how should scarce medical oxygen be shared among states?

Real data (data/oxygen/README.md):
  - state demand and allocation (MT/day), order of 21 April 2021, 19 states
    (Rajya Sabha Unstarred Q.1056, 12 Dec 2023, Annexure)
  - active COVID-19 cases per state on 21 April 2021 (covid19india)

On 21 April total allocation (6,722 MT) equalled total demand (6,721 MT), so every
rule that never exceeds a request gives the same answer at full supply. Deliveries
in practice fell short of allocations, so the question is how each rule behaves when
supply is SHORT. Supply is scaled to s x total demand, s in {0.6, ..., 1.0}.

"Genuine need" (the core in PULSE terms) is caseload-based, as the government's own
rule was: need_j = m * k * active_j, where k = national allocation per active case on
21 April (total allocation / total active cases of the 19 states). m scales how much
oxygen a case really needs relative to that average. PULSE always plans with m = 1
(the published norm); the metrics are evaluated at m in {0.75, 1, 1.25}, so the
m != 1 rows test PULSE under a WRONG need norm.

Rules compared (all water-filling, i.e. proportional shares with caps):
  pro-rata of requests      every state gets the same share of what it asked for
  caseload, capped          shares proportional to active cases, never above request
  PULSE core/excess         core = min(request, need at m=1) served first, pro-rata;
                            leftover goes to excess (request - core), pro-rata
  actual order (s = 1 only) what the government actually allocated

With a single pooled supply and no per-state urgency, PULSE's LP has many optimal
solutions; the pro-rata split inside core and inside excess is the tie-break (D17).

    python scripts/oxygen_case.py            # writes results/oxygen/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DATE = "2021_04_21"


def capped_proportional(supply: float, weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Split `supply` in proportion to `weights`, never above `caps` (water-filling)."""
    x = np.zeros(len(weights))
    free = (weights > 0) & (caps > 0)
    left = float(supply)
    while left > 1e-9 and free.any():
        share = left * weights[free] / weights[free].sum()
        room = caps[free] - x[free]
        if (share <= room + 1e-12).all():
            x[free] += share
            break
        hit = np.zeros_like(free)
        hit[np.flatnonzero(free)[share > room]] = True
        left -= (caps[hit] - x[hit]).sum()
        x[hit] = caps[hit]
        free &= ~hit
    return x


def pulse_core_excess(supply: float, demand: np.ndarray, need: np.ndarray) -> np.ndarray:
    core = np.minimum(demand, need)
    x = capped_proportional(min(supply, core.sum()), core, core)
    rest = supply - x.sum()
    excess = demand - core
    return x + capped_proportional(rest, excess, excess)


def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def evaluate(alloc: np.ndarray, need: np.ndarray) -> dict:
    met = np.minimum(alloc, need)
    fill = met / need
    return {
        "need_met_share": met.sum() / need.sum(),
        "states_below_80pct_need": int((fill < 0.8).sum()),
        "worst_state_fill": float(fill.min()),
        "allocated_beyond_need_mt": float(np.maximum(0, alloc - need).sum()),
        "gini_of_need_fill": gini(fill),
    }


def load(data_dir: str) -> pd.DataFrame:
    o = pd.read_csv(os.path.join(data_dir, "oxygen_allocation.csv"))
    a = pd.read_csv(os.path.join(data_dir, "active_cases_2021.csv"))
    d = o.merge(a[["State", f"active_{DATE}"]], left_on="state", right_on="State", how="left")
    if d[f"active_{DATE}"].isna().any():
        raise ValueError(f"no active cases for: {list(d.loc[d[f'active_{DATE}'].isna(), 'state'])}")
    d = d[d[f"demand_mt_{DATE}"].notna()].reset_index(drop=True)
    return d.rename(columns={f"demand_mt_{DATE}": "demand", f"allocation_mt_{DATE}": "actual",
                             f"active_{DATE}": "active"})[["state", "active", "demand", "actual"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/oxygen")
    ap.add_argument("--out", default="results/oxygen")
    a = ap.parse_args()

    d = load(a.data)
    D = d["demand"].to_numpy(float)
    act = d["active"].to_numpy(float)
    k = d["actual"].sum() / act.sum()                       # MT per active case, 21 April
    need_plan = k * act                                     # PULSE plans with m = 1

    rows, per_state = [], []
    for m in (0.75, 1.0, 1.25):
        need = m * k * act
        for s in (0.6, 0.7, 0.8, 0.9, 1.0):
            S = s * D.sum()
            rules = {
                "pro-rata of requests": capped_proportional(S, D, D),
                "caseload, capped at request": capped_proportional(S, act, D),
                "PULSE core/excess": pulse_core_excess(S, D, need_plan),
            }
            if s == 1.0:
                rules["actual order (21 Apr)"] = d["actual"].to_numpy(float)
            for name, x in rules.items():
                rows.append({"need_multiplier_m": m, "supply_share_s": s, "rule": name,
                             "supply_mt": float(x.sum()), **evaluate(x, need)})
                if m == 1.0 and s in (0.7, 1.0):
                    per_state.append(pd.DataFrame({"supply_share_s": s, "rule": name,
                                                   "state": d["state"], "allocation_mt": x.round(1),
                                                   "need_mt": need.round(1), "demand_mt": D}))

    res = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "oxygen_rules.csv"), index=False)
    pd.concat(per_state).to_csv(os.path.join(a.out, "oxygen_per_state.csv"), index=False)

    ratio = (D / need_plan)
    print(f"19 states, 21 April 2021. Supply = {d['actual'].sum():.0f} MT, demand = {D.sum():.0f} MT.")
    print(f"Caseload norm k = {k * 1000:.2f} MT per 1,000 active cases.")
    print("Request / caseload need (m = 1): "
          + ", ".join(f"{st} {r:.2f}" for st, r in sorted(zip(d['state'], ratio), key=lambda t: -t[1])))
    pd.set_option("display.width", 200)
    show = res[res["need_multiplier_m"] == 1.0].drop(columns="need_multiplier_m")
    print("\nNeed norm m = 1 (PULSE's planning norm is correct):")
    print(show.round(3).to_string(index=False))
    for m in (0.75, 1.25):
        sub = res[(res["need_multiplier_m"] == m) & (res["supply_share_s"] == 0.7)]
        print(f"\nRobustness: true need norm m = {m} (PULSE still plans with m = 1), supply s = 0.7:")
        print(sub.drop(columns=["need_multiplier_m"]).round(3).to_string(index=False))
    print(f"\nwrote {a.out}/oxygen_rules.csv and oxygen_per_state.csv")


if __name__ == "__main__":
    main()
