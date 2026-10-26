"""One round's optimisation model in PuLP (README Section 6).

Mode A: LP, constraints (1)-(2). Integral by total unimodularity.
Mode B: MILP, adds (3) strict minimum order and (4)-(5) no tiny remainder.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pulp

log = logging.getLogger("pulse")


@dataclass
class RoundProblem:
    e_sup: np.ndarray      # local supplier index per edge
    e_req: np.ndarray      # local requester index per edge
    w: np.ndarray          # objective coefficient per edge (p_j * psi_hat - c_ij)
    xbar: np.ndarray       # upper bound per edge, min(A_i, D~_j)
    moq: np.ndarray        # m_i per edge
    A: np.ndarray          # available stock per local supplier
    Dt: np.ndarray         # matchable demand per local requester (after cap)
    D: np.ndarray          # outstanding demand per local requester (uncapped)
    m_min: int

    @property
    def n_edges(self) -> int:
        return len(self.w)


@dataclass
class Solution:
    x: np.ndarray          # integer allocation per edge
    status: str
    sol_status: int        # PuLP: 1 optimal, 2 feasible (e.g. time limit), 0 none, ...
    objective: float
    seconds: float
    z_lp: Optional[float]  # Mode B: LP-relaxation bound
    n_vars: int
    n_binaries: int

    @property
    def gap(self) -> Optional[float]:
        if self.z_lp is None or self.z_lp <= 0:
            return None
        return max(0.0, (self.z_lp - self.objective) / self.z_lp)


def _solver(name: str, time_limit: Optional[float], gap_rel: Optional[float] = None,
            warm: bool = False):
    name = name.lower()
    kw = {"msg": False}
    if warm and name == "cbc":
        kw["warmStart"] = True
    if time_limit is not None:
        kw["timeLimit"] = time_limit
    if gap_rel is not None:
        kw["gapRel"] = gap_rel
    if name == "cbc":
        return pulp.PULP_CBC_CMD(**kw)
    if name == "highs":
        return pulp.HiGHS(**kw)
    raise ValueError(f"unknown solver {name!r} (use 'cbc' or 'highs')")


def _groups(idx: np.ndarray) -> dict:
    g = defaultdict(list)
    for k, v in enumerate(idx.tolist()):
        g[v].append(k)
    return g


def greedy_mode_b(P: RoundProblem) -> np.ndarray:
    """A feasible Mode B allocation, built edge by edge in decreasing value.

    Each shipment is either skipped or sized as large as possible while keeping
    (1)-(5): at least m_i, within stock and matchable demand, and never leaving
    the requester with a remainder in (0, m_min). Used as the MILP warm start, so
    the solver always has a valid incumbent even if it hits the time limit.
    """
    E = P.n_edges
    x = np.zeros(E, dtype=np.int64)
    res_s = P.A.astype(np.int64).copy()
    alloc = np.zeros(len(P.D), dtype=np.int64)
    order = np.argsort(-P.w, kind="stable")
    for k in order:
        i, j = P.e_sup[k], P.e_req[k]
        room = min(res_s[i], P.Dt[j] - alloc[j])
        if room < P.moq[k]:
            continue
        left = P.D[j] - alloc[j] - room            # remainder if we ship `room`
        if 0 < left < P.m_min:
            room = P.D[j] - alloc[j] - P.m_min     # leave exactly m_min instead
            if room < P.moq[k]:
                continue
        x[k] = room
        res_s[i] -= room
        alloc[j] += room
    return x


def _build(P: RoundProblem, mode_b: bool, relax: bool):
    lp = pulp.LpProblem("pulse_round", pulp.LpMaximize)
    E = P.n_edges
    integer = mode_b and not relax
    cat_x = "Integer" if integer else "Continuous"
    xbar = P.xbar.tolist()
    x = [pulp.LpVariable(f"x{k}", lowBound=0, upBound=xbar[k], cat=cat_x) for k in range(E)]
    lp.setObjective(pulp.LpAffineExpression(list(zip(x, P.w.tolist()))))

    by_sup = _groups(P.e_sup)
    by_req = _groups(P.e_req)
    for i, ks in by_sup.items():                                   # (1)
        lp.addConstraint(pulp.LpAffineExpression([(x[k], 1) for k in ks]) <= int(P.A[i]), f"sup{i}")
    for j, ks in by_req.items():                                   # (2)
        lp.addConstraint(pulp.LpAffineExpression([(x[k], 1) for k in ks]) <= int(P.Dt[j]), f"dem{j}")

    n_bin = 0
    if mode_b:
        cat_b = "Binary" if integer else "Continuous"
        moq = P.moq.tolist()
        y = [pulp.LpVariable(f"y{k}", lowBound=0, upBound=1, cat=cat_b) for k in range(E)]
        for k in range(E):                                         # (3)
            lp.addConstraint(pulp.LpAffineExpression([(x[k], 1), (y[k], -moq[k])]) >= 0, f"lo{k}")
            lp.addConstraint(pulp.LpAffineExpression([(x[k], 1), (y[k], -xbar[k])]) <= 0, f"hi{k}")
        n_bin += E
        for j, ks in by_req.items():
            Dj = int(P.D[j])
            capped = P.Dt[j] < P.D[j]
            wj = pulp.LpVariable(f"w{j}", lowBound=0, upBound=0 if capped else 1, cat=cat_b)
            n_bin += 1
            terms = [(x[k], 1) for k in ks]
            lp.addConstraint(pulp.LpAffineExpression(terms + [(wj, -Dj)]) >= 0, f"close{j}")          # (4)
            lp.addConstraint(pulp.LpAffineExpression(terms + [(wj, -P.m_min)]) <= Dj - P.m_min,
                             f"rem{j}")                                                               # (5)
    return lp, x, n_bin


def is_feasible_mode_b(P: RoundProblem, x: np.ndarray) -> bool:
    """Check (1)-(5) directly on an integer allocation."""
    if (x < 0).any() or (x > P.xbar).any():
        return False
    pos = x > 0
    if (x[pos] < P.moq[pos]).any():
        return False
    if (np.bincount(P.e_sup, weights=x, minlength=len(P.A)) > P.A).any():
        return False
    alloc = np.bincount(P.e_req, weights=x, minlength=len(P.D))
    if (alloc > P.Dt).any():
        return False
    rem = P.D - alloc
    return bool(((rem == 0) | (rem >= P.m_min)).all())


def _subset(P: RoundProblem, sel: np.ndarray) -> RoundProblem:
    return RoundProblem(e_sup=P.e_sup[sel], e_req=P.e_req[sel], w=P.w[sel], xbar=P.xbar[sel],
                        moq=P.moq[sel], A=P.A, Dt=P.Dt, D=P.D, m_min=P.m_min)


def _top_k_per_requester(P: RoundProblem, k: int) -> np.ndarray:
    order = np.lexsort((-P.w, P.e_req))              # by requester, then value descending
    req_sorted = P.e_req[order]
    first = np.r_[True, req_sorted[1:] != req_sorted[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order)) - start
    return rank < k


def _values(xvars) -> np.ndarray:
    return np.array([v.varValue if v.varValue is not None else np.nan for v in xvars], dtype=float)


def solve_round(P: RoundProblem, mode_b: bool, solver: str = "cbc",
                time_limit: Optional[float] = 30.0, compute_bound: bool = True,
                gap_rel: Optional[float] = 0.005, candidate_k: int = 10) -> Solution:
    """Solve one round.

    Mode A: one LP, integral by total unimodularity (README 7.3).

    Mode B (README 7.5 and 14):
      1. Solve the LP relaxation of the full Mode B model. Its value z_lp is a valid
         upper bound on the true Mode B optimum.
      2. Keep a candidate edge set: the LP solution's support, the greedy solution's
         support, and each requester's top-`candidate_k` edges by value
         (candidate_k = 0 keeps every edge, i.e. the exact model).
      3. Solve the MILP on the candidates, warm-started from the greedy solution
         (CBC), stopping at `gap_rel` or `time_limit`.
      4. If the solver returns nothing valid, use the greedy solution.
    The reported gap (z_lp - objective) / z_lp is measured against the FULL problem's
    bound, so it certifies the quality of the answer even though step 2 is a heuristic.
    """
    t0 = time.perf_counter()
    if P.n_edges == 0:
        return Solution(x=np.zeros(0, dtype=np.int64), status="NoEdges", sol_status=1,
                        objective=0.0, seconds=0.0, z_lp=0.0 if mode_b else None,
                        n_vars=0, n_binaries=0)

    if not mode_b:
        lp, x, _ = _build(P, mode_b=False, relax=False)
        lp.solve(_solver(solver, None))
        status = pulp.LpStatus[lp.status]
        vals = _values(x)
        if status != "Optimal" or np.isnan(vals).any():
            raise RuntimeError(f"Mode A LP failed: {status}")
        frac = np.abs(vals - np.round(vals)).max()
        if frac > 1e-6:                                            # README 7.3
            raise AssertionError(f"Mode A solution not integral (max fractional part {frac:.2e})")
        xi = np.clip(np.round(vals), 0, P.xbar).astype(np.int64)
        return Solution(x=xi, status=status, sol_status=1, objective=float(P.w @ xi),
                        seconds=time.perf_counter() - t0, z_lp=None, n_vars=len(x), n_binaries=0)

    # ---- Mode B ----
    x_greedy = greedy_mode_b(P)
    z_lp, lp_support = None, np.zeros(P.n_edges, dtype=bool)
    if compute_bound or candidate_k > 0:
        lr, xr, _ = _build(P, mode_b=True, relax=True)
        lr.solve(_solver(solver, None))
        if pulp.LpStatus[lr.status] == "Optimal":
            z_lp = float(pulp.value(lr.objective) or 0.0)
            lp_support = np.nan_to_num(_values(xr)) > 1e-6

    if candidate_k > 0:
        keep = lp_support | (x_greedy > 0) | _top_k_per_requester(P, candidate_k)
        sel = np.nonzero(keep)[0]
    else:
        sel = np.arange(P.n_edges)
    Q = _subset(P, sel)
    xg_q = x_greedy[sel]

    lp, x, n_bin = _build(Q, mode_b=True, relax=False)
    warm = solver.lower() == "cbc"
    if warm:
        byname = {v.name: v for v in lp.variables()}
        for k in range(Q.n_edges):
            x[k].setInitialValue(int(xg_q[k]))
            byname[f"y{k}"].setInitialValue(int(xg_q[k] > 0))
        alloc = np.bincount(Q.e_req, weights=xg_q, minlength=len(Q.D))
        for j in set(Q.e_req.tolist()):
            if f"w{j}" in byname:
                byname[f"w{j}"].setInitialValue(int(alloc[j] == Q.D[j]))
    lp.solve(_solver(solver, time_limit, gap_rel, warm=warm))
    status = pulp.LpStatus[lp.status]
    sol_status = int(getattr(lp, "sol_status", 0))

    xi = np.zeros(P.n_edges, dtype=np.int64)
    vals = _values(x)
    ok = sol_status in (1, 2) and not np.isnan(vals).any()
    if ok:
        xi[sel] = np.clip(np.round(vals), 0, Q.xbar).astype(np.int64)
        ok = is_feasible_mode_b(P, xi) and float(P.w @ xi) >= float(P.w @ x_greedy) - 1e-9
    if not ok:
        log.info("Mode B solver gave no better valid solution (%s); using the greedy allocation", status)
        xi = x_greedy
        status, sol_status = "Greedy", 2
    return Solution(x=xi, status=status, sol_status=sol_status, objective=float(P.w @ xi),
                    seconds=time.perf_counter() - t0, z_lp=z_lp if compute_bound else None,
                    n_vars=len(x) + n_bin, n_binaries=n_bin)
