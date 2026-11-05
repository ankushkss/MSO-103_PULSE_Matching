"""Benchmark policies for judging how good PULSE is (CHANGES.md, session 2).

Two kinds:

1. ONLINE HEURISTICS (`heuristic_round`). They replace only the round LP inside
   `Simulation`. They see exactly the same edge set, stock A_i and core/excess rows
   C_j / X_j as PULSE, so any difference comes from the allocation rule alone.
       fcfs          requesters in arrival order, each served by its nearest suppliers
       proportional  every requester gets the same share of its demand
                     (min(1, supply/demand)), then leftovers are handed out greedily;
                     this is how "allocate by caseload" rules work in practice
       random        random requester order, random supplier order (a floor)
   Every heuristic serves all core before any excess and ends with a greedy top-up,
   so its allocation is maximal: the README 7.1 / 7.4 invariant checks still hold.

2. CLAIRVOYANT BOUNDS (`hindsight`). One LP over all rounds that knows every arrival,
   enrolment and abandonment draw in advance. Two variants:
     outcomes="realised"  also knows every match's coin flip, so it ships only on
                          matches that will succeed. A strict per-seed upper bound,
                          but very loose (close to 100%): it removes failure entirely.
     outcomes="expected"  does NOT know the coin flips: a shipment of x units delivers
                          psi_true * x in expectation (fluid relaxation). It isolates
                          the value of knowing the future arrivals. It bounds a policy's
                          EXPECTED performance, so compare it with the policy's mean
                          over seeds, not seed by seed.
   Both relax integrality and the floor in the carry-over rule, and each is solved
   once per target metric, because a bound on one metric is not a bound on another.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .config import Config
from .districts import Districts
from .metrics import ALT_W, REF_W
from .model import RoundProblem, Solution

HEURISTICS = ("fcfs", "proportional", "random")
HINDSIGHT_TARGETS = {"need_fill": None, "wneed_fill_ref": REF_W, "wneed_fill_w123": ALT_W}


# ====================================================================== online heuristics
def _greedy(order: np.ndarray, P: RoundProblem, A_rem: np.ndarray, dem: np.ndarray,
            flow: np.ndarray) -> None:
    """Walk edges in `order`; ship min(stock left, demand left). Updates in place."""
    es, er = P.e_sup, P.e_req
    for k in order:
        i, j = es[k], er[k]
        a = A_rem[i]
        if a <= 0:
            continue
        d = dem[j]
        if d <= 0:
            continue
        q = a if a < d else d
        flow[k] += q
        A_rem[i] = a - q
        dem[j] = d - q


def _proportional(order: np.ndarray, P: RoundProblem, A_rem: np.ndarray, dem: np.ndarray,
                  flow: np.ndarray) -> None:
    """Target the same fill share for every requester, then top up greedily."""
    total = dem.sum()
    if total <= 0:
        return
    share = min(1.0, A_rem.sum() / total)
    target = np.floor(dem * share).astype(np.int64)
    before = flow.copy()
    _greedy(order, P, A_rem, target, flow)
    got = np.bincount(P.e_req, weights=flow - before, minlength=len(dem)).astype(np.int64)
    dem -= got
    _greedy(order, P, A_rem, dem, flow)


def heuristic_round(policy: str, P: RoundProblem, km: np.ndarray, req_key: np.ndarray,
                    rng: np.random.Generator) -> Solution:
    """Allocate one round with a simple rule instead of the LP.

    km:      road distance per edge (nearest supplier first within a requester)
    req_key: per local requester, the order key (arrival round for FCFS)
    """
    t0 = time.perf_counter()
    nE = P.n_edges
    xc = np.zeros(nE, dtype=np.int64)
    xe = np.zeros(nE, dtype=np.int64)
    if nE == 0:
        return Solution(xc=xc, xe=xe, status="NoEdges", objective=0.0, seconds=0.0)

    if policy == "random":
        req_rank = rng.permutation(len(P.C))
        order = np.lexsort((rng.random(nE), req_rank[P.e_req]))
    else:
        # requester key first (arrival round, then index = enrolment order), nearest supplier next
        order = np.lexsort((km, P.e_req, req_key[P.e_req]))

    A_rem = P.A.astype(np.int64).copy()
    core = P.C.astype(np.int64).copy()
    exc = P.X.astype(np.int64).copy()
    fill = _proportional if policy == "proportional" else _greedy
    fill(order, P, A_rem, core, xc)          # all core first ...
    fill(order, P, A_rem, exc, xe)           # ... then excess from what is left
    return Solution(xc=xc, xe=xe, status="Heuristic",
                    objective=float(P.w @ xc + P.eps * (P.w @ xe)),
                    seconds=time.perf_counter() - t0)


# ====================================================================== hindsight optimum
def _world_edges(sim, t: int, sup: np.ndarray, req: np.ndarray, outcomes: str):
    """Edges a shipment could succeed on in round t (world rules, not policy filters),
    and the delivered fraction per shipped unit on each edge."""
    inst, d, cfg = sim.inst, sim.d, sim.cfg
    sd, rd = inst.s_dist[sup], inst.r_dist[req]
    ok = (inst.s_var[sup][:, None] & inst.r_var[req][None, :]) != 0
    ok &= inst.s_price[sup][:, None] <= inst.r_pmax[req][None, :]        # else refused
    ok &= (inst.r_kappa[req] != 2)[None, :] | inst.s_cert[sup][:, None]  # else refused
    ok &= np.isfinite(d.R[np.ix_(sd, rd)])
    same = d.state[sd][:, None] == d.state[rd][None, :]
    p = np.where(same, cfg.psi_true[0], cfg.psi_true[1]) if cfg.simulate_failures \
        else np.ones(ok.shape)
    if outcomes == "realised":
        if cfg.simulate_failures:
            ok &= sim.U_out[t - 1][np.ix_(sup, req)] < p                  # the coin flip
        p = np.ones(ok.shape)
    ei, ej = np.nonzero(ok)
    return sup[ei], req[ej], p[ei, ej]


def hindsight(cfg: Config, districts: Districts, seed: int,
              outcomes: str = "realised") -> tuple[list[dict], dict]:
    """Clairvoyant bound on need_fill, wneed_fill_ref and wneed_fill_w123 (module docstring)."""
    if outcomes not in ("realised", "expected"):
        raise ValueError("outcomes must be 'realised' or 'expected'")
    from .simulation import Simulation        # local import: simulation imports this module

    t_start = time.perf_counter()
    sim = Simulation(cfg, districts, seed=seed)   # same instance and pre-drawn uniforms
    inst, T = sim.inst, cfg.n_rounds
    nS, nR = inst.n_suppliers, inst.n_requesters
    gaps = np.asarray(cfg.gaps, dtype=float)

    # requester j can receive in rounds arrival_j .. last_j, where last_j is the first
    # round its abandonment draw fires (it is served in that round before leaving)
    arr = inst.r_arrival
    last = np.full(nR, T)
    for t in range(T, 0, -1):
        fires = (sim.U_ab[t - 1] < cfg.omega) & (arr <= t)
        last = np.where(fires, t, last)

    # ---- variables: x (shipments that will succeed), r (carried stock), y (credited need)
    xs, xr, xt, xp = [], [], [], []
    for t in range(1, T + 1):
        sup = np.nonzero(inst.s_enrol <= t)[0]
        req = np.nonzero((arr <= t) & (last >= t))[0]
        if len(sup) and len(req):
            s_, r_, p_ = _world_edges(sim, t, sup, req, outcomes)
            xs.append(s_); xr.append(r_); xt.append(np.full(len(s_), t)); xp.append(p_)
    xs = np.concatenate(xs) if xs else np.zeros(0, int)
    xr = np.concatenate(xr) if xr else np.zeros(0, int)
    xt = np.concatenate(xt) if xt else np.zeros(0, int)
    xp = np.concatenate(xp) if xp else np.zeros(0, float)    # delivered per shipped unit
    nX = len(xs)

    # r[i, t] = stock carried out of round t into t+1, for t = 1..T-1 once enrolled
    r_idx = -np.ones((nS, T + 1), dtype=np.int64)
    k = nX
    for t in range(1, T):
        for i in np.nonzero(inst.s_enrol <= t)[0]:
            r_idx[i, t] = k
            k += 1
    nRv = k - nX
    y0 = k
    nV = k + nR

    rows, cols, vals, rhs = [], [], [], []
    row = 0

    def add(r, c, v):
        rows.extend(r); cols.extend(c); vals.extend(v)

    # supply in round t:  sum_j x_tij - r_{i,t-1} <= s_i * gap_t
    # carry (failed shipments return to the supplier, as in the simulation):
    #                     r_it + eta * sum_j p_tij x_tij - eta * r_{i,t-1} <= eta * s_i * gap_t
    key = xt * nS + xs
    order = np.argsort(key, kind="stable")
    bounds_k = np.searchsorted(key[order], np.arange(T + 2) * nS)
    for t in range(1, T + 1):
        lo, hi = bounds_k[t], bounds_k[t + 1]
        edges_t = order[lo:hi]
        by_sup = {}
        for e in edges_t:
            by_sup.setdefault(int(xs[e]), []).append(int(e))
        for i in np.nonzero(inst.s_enrol <= t)[0]:
            es = by_sup.get(int(i), [])
            prev = r_idx[i, t - 1] if t > 1 else -1
            fresh = float(inst.s_rate[i] * gaps[t - 1])
            # supply
            add([row] * len(es), es, [1.0] * len(es))
            if prev >= 0:
                add([row], [prev], [-1.0])
            rhs.append(fresh); row += 1
            # carry-over (only rounds 1..T-1 carry)
            if t < T:
                add([row], [r_idx[i, t]], [1.0])
                add([row] * len(es), es, list(cfg.eta * xp[es]))
                if prev >= 0:
                    add([row], [prev], [-cfg.eta])
                rhs.append(cfg.eta * fresh); row += 1

    # demand, round by round, as in the simulation: what is shipped in round t may not
    # exceed what is still outstanding, i.e. the request minus (expected) deliveries in
    # earlier rounds. Without this the fluid LP could "overbook": ship ~1/psi times the
    # request and rely on most shipments failing, which no PULSE-class policy may do.
    #     sum_i x_tij + sum_{t'<t} sum_i p_t'ij x_t'ij <= D0_j     for t in j's window
    eidx = np.arange(nX)
    for t in range(1, T + 1):
        live = np.nonzero((arr <= t) & (last >= t))[0]
        if not len(live):
            continue
        rid = np.full(nR, -1)
        rid[live] = row + np.arange(len(live))
        now = eidx[(xt == t) & (rid[xr] >= 0)]
        before = eidx[(xt < t) & (rid[xr] >= 0)]
        add(rid[xr[now]], now, np.ones(len(now)))
        add(rid[xr[before]], before, xp[before])
        rhs.extend(inst.r_D0[live].astype(float)); row += len(live)
    # credited need: y_j - sum p_tij x_tij <= 0
    add(row + np.arange(nR), y0 + np.arange(nR), np.ones(nR))
    add(row + xr, np.arange(nX), -xp)
    rhs.extend([0.0] * nR); row += nR

    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(row, nV))
    b_ub = np.asarray(rhs)

    genuine = np.minimum(inst.r_D0, inst.r_usage * inst.r_size * cfg.need_days).astype(float)
    ub = np.full(nV, np.inf)
    for i in range(nS):
        for t in range(1, T):
            if r_idx[i, t] >= 0:
                ub[r_idx[i, t]] = cfg.h * inst.s_rate[i]
    ub[y0:] = genuine
    bnds = np.column_stack([np.zeros(nV), ub])

    arrived = arr <= T
    lvl = inst.r_level
    summary = {"seed": seed, "suppliers": nS, "requesters": int(arrived.sum()), "outcomes": outcomes,
               "hindsight_vars": nV, "hindsight_rows": row}
    for metric, wts in HINDSIGHT_TARGETS.items():
        w = np.ones(nR) if wts is None else wts[lvl - 1]
        c = np.zeros(nV)
        c[y0:] = -(w * arrived)
        # tiny bonus on deliveries so leftover stock still goes somewhere sensible
        c[:nX] = -1e-9
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"hindsight LP failed for {metric}: {res.message}")
        y = res.x[y0:]
        summary[metric] = float((w * y)[arrived].sum() / (w * genuine)[arrived].sum())
        if metric == "need_fill":
            got = np.bincount(xr, weights=xp * res.x[:nX], minlength=nR)
            summary["delivered"] = float(got[arrived].sum())
    summary["solve_seconds"] = round(time.perf_counter() - t_start, 2)
    return [], summary
