# The per-round LP, built and solved with PuLP
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
    w: np.ndarray          # value of one core unit on the edge (p_j * psi_hat - c_ij)
    A: np.ndarray          # available stock per local supplier
    C: np.ndarray          # core demand per local requester
    X: np.ndarray          # excess demand per local requester
    eps: float = 1e-4      # value of an excess unit relative to a core unit

    @property
    def n_edges(self) -> int:
        return len(self.w)


@dataclass
class Solution:
    xc: np.ndarray         # integer core flow per edge
    xe: np.ndarray         # integer excess flow per edge
    status: str
    objective: float
    seconds: float

    @property
    def x(self) -> np.ndarray:
        return self.xc + self.xe


def _solver(name: str):
    name = name.lower()
    if name == "cbc":
        return pulp.PULP_CBC_CMD(msg=False)
    if name == "highs":
        return pulp.HiGHS(msg=False)
    raise ValueError(f"unknown solver {name!r} (use 'cbc' or 'highs')")


def build_lp(P: RoundProblem):
    lp = pulp.LpProblem("pulse_round", pulp.LpMaximize)
    A = P.A.astype(np.int64)
    Cj = P.C[P.e_req].astype(np.int64)
    Xj = P.X[P.e_req].astype(np.int64)
    Ai = A[P.e_sup]
    xc, xe, terms = [], [], []
    sup_rows, core_rows, exc_rows = defaultdict(list), defaultdict(list), defaultdict(list)
    for k in range(P.n_edges):
        i, j = int(P.e_sup[k]), int(P.e_req[k])
        vc = ve = None
        if Cj[k] > 0:
            vc = pulp.LpVariable(f"c{k}", lowBound=0, upBound=int(min(Ai[k], Cj[k])))
            terms.append((vc, float(P.w[k])))
            sup_rows[i].append(vc)
            core_rows[j].append(vc)
        if Xj[k] > 0:
            ve = pulp.LpVariable(f"e{k}", lowBound=0, upBound=int(min(Ai[k], Xj[k])))
            terms.append((ve, float(P.eps * P.w[k])))
            sup_rows[i].append(ve)
            exc_rows[j].append(ve)
        xc.append(vc)
        xe.append(ve)
    lp.setObjective(pulp.LpAffineExpression(terms))
    for i, vs in sup_rows.items():
        lp.addConstraint(pulp.LpAffineExpression([(v, 1) for v in vs]) <= int(A[i]), f"sup{i}")
    for j, vs in core_rows.items():
        lp.addConstraint(pulp.LpAffineExpression([(v, 1) for v in vs]) <= int(P.C[j]), f"core{j}")
    for j, vs in exc_rows.items():
        lp.addConstraint(pulp.LpAffineExpression([(v, 1) for v in vs]) <= int(P.X[j]), f"exc{j}")
    return lp, xc, xe


def _vals(vs) -> np.ndarray:
    return np.array([0.0 if v is None else (np.nan if v.varValue is None else v.varValue)
                     for v in vs], dtype=float)


def solve_round(P: RoundProblem, solver: str = "cbc") -> Solution:
    t0 = time.perf_counter()
    if P.n_edges == 0:
        z = np.zeros(0, dtype=np.int64)
        return Solution(xc=z, xe=z.copy(), status="NoEdges", objective=0.0, seconds=0.0)
    lp, vc, ve = build_lp(P)
    if not lp.variables():
        z = np.zeros(P.n_edges, dtype=np.int64)
        return Solution(xc=z, xe=z.copy(), status="NoDemand", objective=0.0, seconds=0.0)
    lp.solve(_solver(solver))
    status = pulp.LpStatus[lp.status]
    c, e = _vals(vc), _vals(ve)
    if status != "Optimal" or np.isnan(c).any() or np.isnan(e).any():
        raise RuntimeError(f"round LP failed: {status}")
    frac = max(np.abs(c - np.round(c)).max(), np.abs(e - np.round(e)).max())
    if frac > 1e-6:
        raise AssertionError(f"LP solution not integral (max fractional part {frac:.2e})")
    xc, xe = np.round(c).astype(np.int64), np.round(e).astype(np.int64)
    return Solution(xc=xc, xe=xe, status=status,
                    objective=float(P.w @ xc + P.eps * (P.w @ xe)),
                    seconds=time.perf_counter() - t0)
