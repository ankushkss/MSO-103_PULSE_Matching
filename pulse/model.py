"""One round's optimisation model in PuLP (README Section 6).

    max   sum_(i,j) in E  w_ij x_ij              w_ij = p_j * psi_hat_ij - c_ij
    s.t.  sum_j x_ij <= A_i       for every supplier i      (1)
          sum_i x_ij <= D~_j      for every requester j     (2)
          0 <= x_ij <= min(A_i, D~_j)

A transportation-type LP. Its constraint matrix is the incidence matrix of a
bipartite graph, so it is totally unimodular and every vertex is integral when
A_i and D~_j are integers (README 7.3).
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pulp


@dataclass
class RoundProblem:
    e_sup: np.ndarray      # local supplier index per edge
    e_req: np.ndarray      # local requester index per edge
    w: np.ndarray          # objective coefficient per edge (p_j * psi_hat - c_ij)
    xbar: np.ndarray       # upper bound per edge, min(A_i, D~_j)
    A: np.ndarray          # available stock per local supplier
    Dt: np.ndarray         # matchable demand per local requester (after the cap)

    @property
    def n_edges(self) -> int:
        return len(self.w)


@dataclass
class Solution:
    x: np.ndarray          # integer allocation per edge
    status: str
    objective: float
    seconds: float


def _solver(name: str):
    name = name.lower()
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=False)
    if name == "highs":
        return pulp.HiGHS(msg=False)
    raise ValueError(f"unknown solver {name!r} (use 'cbc' or 'highs')")


def _groups(idx: np.ndarray) -> dict:
    g = defaultdict(list)
    for k, v in enumerate(idx.tolist()):
        g[v].append(k)
    return g


def build_lp(P: RoundProblem):
    lp = pulp.LpProblem("pulse_round", pulp.LpMaximize)
    xbar = P.xbar.tolist()
    x = [pulp.LpVariable(f"x{k}", lowBound=0, upBound=xbar[k], cat="Continuous")
         for k in range(P.n_edges)]
    lp.setObjective(pulp.LpAffineExpression(list(zip(x, P.w.tolist()))))
    for i, ks in _groups(P.e_sup).items():                                  # (1)
        lp.addConstraint(pulp.LpAffineExpression([(x[k], 1) for k in ks]) <= int(P.A[i]), f"sup{i}")
    for j, ks in _groups(P.e_req).items():                                  # (2)
        lp.addConstraint(pulp.LpAffineExpression([(x[k], 1) for k in ks]) <= int(P.Dt[j]), f"dem{j}")
    return lp, x


def solve_round(P: RoundProblem, solver: str = "cbc") -> Solution:
    t0 = time.perf_counter()
    if P.n_edges == 0:
        return Solution(x=np.zeros(0, dtype=np.int64), status="NoEdges", objective=0.0, seconds=0.0)
    lp, x = build_lp(P)
    lp.solve(_solver(solver))
    status = pulp.LpStatus[lp.status]
    vals = np.array([v.varValue if v.varValue is not None else np.nan for v in x], dtype=float)
    if status != "Optimal" or np.isnan(vals).any():
        raise RuntimeError(f"round LP failed: {status}")
    frac = np.abs(vals - np.round(vals)).max()
    if frac > 1e-6:                                                         # README 7.3
        raise AssertionError(f"LP solution not integral (max fractional part {frac:.2e})")
    xi = np.clip(np.round(vals), 0, P.xbar).astype(np.int64)
    return Solution(x=xi, status=status, objective=float(P.w @ xi),
                    seconds=time.perf_counter() - t0)
