"""Per-round records and run summaries (README Section 9).

Fill rates in the run summary are measured on REALISED deliveries.
`wfill_ref` weights requesters by fixed reference weights (1, 10, 100) for
levels 1/2/3, whatever W the run used, so runs with different W stay comparable.
Bulk requesters are reported separately because one bulk request can be the
size of a whole round's supply and would swamp every other fill number.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .simulation import Simulation

REF_W = np.array([1.0, 10.0, 100.0])


class MatchLog:
    """Every match made in a run (one row per x_ij > 0)."""
    FIELDS = ("tau", "sup", "req", "x", "delivered", "km", "same_state", "same_district",
              "price", "cert", "kappa", "level", "psi_true")

    def __init__(self):
        self._parts = {f: [] for f in self.FIELDS}

    def add(self, tau: int, **cols):
        n = len(cols["x"])
        self._parts["tau"].append(np.full(n, tau, dtype=np.int64))
        for f in self.FIELDS[1:]:
            self._parts[f].append(np.asarray(cols[f]))

    def get(self, f: str) -> np.ndarray:
        parts = self._parts[f]
        return np.concatenate(parts) if parts else np.zeros(0)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame({f: self.get(f) for f in self.FIELDS})


def _ratio(a, b):
    return float(a) / float(b) if b else float("nan")


def round_record(*, sim: "Simulation", tau, A_tot, D, Dt, req, n_sup, cap, flagged, capped,
                 x, deliv, matched, B, psi, c2_ok, sol, n_edges, stranded, n_stranded,
                 leaked, lost_storage, n_new, n_served, n_abandoned, units_abandoned,
                 n_verified, same, Rk, cls) -> dict:
    alloc = int(x.sum())
    D_tot = int(D.sum())
    rec = {
        "seed": sim.seed,
        "mode": sim.cfg.mode,
        "round": tau,
        "day": sim.cfg.days[tau - 1],
        "suppliers_active": n_sup,
        "requesters_active": len(req),
        "requesters_new": n_new,
        "A_tot": A_tot,
        "D_tot": D_tot,
        "supply_demand_ratio": _ratio(A_tot, D_tot),
        "cap": cap,
        "flagged": int(flagged.sum()),
        "capped": int(capped.sum()),
        "units_deferred_by_cap": int((D - Dt)[capped].sum()),
        "edges": n_edges,
        "matches": int(matched.sum()),
        "allocated": alloc,
        "expected_delivered": float((psi[cls] * x).sum()) if len(x) else 0.0,
        "delivered": int(deliv.sum()),
        "utilisation": _ratio(alloc, A_tot),
        "round_fill": _ratio(int(deliv.sum()), D_tot),
        "failed_matches": int((matched & ~B).sum()),
        "failed_units": int(x[matched & ~B].sum()),
        "same_state_share": _ratio(int(x[same].sum()), alloc),
        "mean_km": _ratio(float((Rk * x).sum()), alloc),
        "median_km": float(np.median(Rk[matched])) if matched.any() else float("nan"),
        "psi_hat_same": float(psi[0]),
        "psi_hat_cross": float(psi[1]),
        "c2_ok": bool(c2_ok),
        "served": n_served,
        "abandoned": n_abandoned,
        "units_abandoned": units_abandoned,
        "verified": n_verified,
        "leaked_stock": leaked,
        "stock_lost_to_storage_cap": lost_storage,
        "stranded": n_stranded,
        "solver_status": sol.status,
        "sol_status": sol.sol_status,
        "objective": sol.objective,
        "z_lp_bound": sol.z_lp,
        "gap": sol.gap,
        "solve_seconds": round(sol.seconds, 3),
        "binaries": sol.n_binaries,
    }
    for cause, n in stranded.items():
        rec[f"stranded_{cause}"] = n
    return rec


def run_summary(sim: "Simulation") -> dict:
    inst, cfg = sim.inst, sim.cfg
    arrived = sim.status != 0
    bulk = inst.r_bulk & arrived
    nonbulk = ~inst.r_bulk & arrived
    D0, got, lvl = inst.r_D0, sim.delivered, inst.r_level

    out = {
        "seed": sim.seed,
        "mode": cfg.mode,
        "suppliers": inst.n_suppliers,
        "requesters": int(arrived.sum()),
        "bulk_requesters": int(bulk.sum()),
        "requested": int(D0[arrived].sum()),
        "delivered": int(got.sum()),
        "fill_overall": _ratio(got[arrived].sum(), D0[arrived].sum()),
        "fill_nonbulk": _ratio(got[nonbulk].sum(), D0[nonbulk].sum()),
        "fill_bulk": _ratio(got[bulk].sum(), D0[bulk].sum()),
    }
    for l in (1, 2, 3):
        sel = nonbulk & (lvl == l)
        out[f"fill_L{l}"] = _ratio(got[sel].sum(), D0[sel].sum())
    sel = nonbulk & (lvl == 3) & ~sim.ever_flagged
    out["fill_L3_unflagged"] = _ratio(got[sel].sum(), D0[sel].sum())
    wts = REF_W[lvl - 1]
    out["wfill_ref"] = _ratio((wts * got)[nonbulk].sum(), (wts * D0)[nonbulk].sum())
    out["served_share_nonbulk"] = _ratio(((sim.status == 2) & nonbulk).sum(), nonbulk.sum())
    out["flagged_ever"] = int(sim.ever_flagged.sum())
    out["units_abandoned"] = int(sim.abandoned_units.sum())
    out["outstanding_end"] = int(sim.carry[sim.status == 1].sum())

    m = sim.matches
    x, dv, km = m.get("x"), m.get("delivered"), m.get("km")
    ss, sdis = m.get("same_state").astype(bool), m.get("same_district").astype(bool)
    price, cert, kap = m.get("price"), m.get("cert").astype(bool), m.get("kappa")
    # Expected deliveries under the TRUE success probabilities: the same policy value
    # without the luck of the Bernoulli draws, so it varies much less across seeds.
    exp = (m.get("psi_true") * x).sum()
    out["expected_delivered"] = float(exp)
    out["exp_fill_overall"] = _ratio(exp, D0[arrived].sum())
    out["matches"] = int(len(x))
    out["failed_match_share"] = _ratio((dv == 0).sum(), len(x))
    out["same_state_share"] = _ratio(x[ss].sum(), x.sum())
    out["same_district_share"] = _ratio(x[sdis].sum(), x.sum())
    out["mean_km"] = _ratio((km * x).sum(), x.sum())
    out["median_km"] = float(np.median(km)) if len(km) else float("nan")
    out["mean_price_paid"] = _ratio((price * dv).sum(), dv.sum())
    out["certified_share"] = _ratio(dv[cert].sum(), dv.sum())
    k1 = kap == 1
    out["certified_share_prefer"] = _ratio(dv[k1 & cert].sum(), dv[k1].sum())

    psi = sim.psi_hat()
    out["psi_hat_same_final"] = float(psi[0])
    out["psi_hat_cross_final"] = float(psi[1])
    rs = sim.rounds
    out["c2_failures"] = int(sum(not r["c2_ok"] for r in rs))
    out["stranded_requester_rounds"] = int(sum(r["stranded"] for r in rs))
    out["solve_seconds"] = round(float(sum(r["solve_seconds"] for r in rs)), 2)
    gaps = [r["gap"] for r in rs if r["gap"] is not None]
    out["max_gap"] = float(max(gaps)) if gaps else float("nan")
    out["rounds_not_proven_optimal"] = int(sum(r["sol_status"] != 1 for r in rs))
    return out
