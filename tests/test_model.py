"""Properties from README Section 7. The simulation also asserts them every round
(check_invariants=True), so a clean run is itself a test; these add direct checks."""
import numpy as np
import pytest
from scipy.optimize import linprog

from pulse import Simulation
from pulse.model import RoundProblem, solve_round


@pytest.mark.parametrize("seed", [1, 2])
def test_runs_with_all_invariants(small_cfg, districts, seed):
    sim = Simulation(small_cfg, districts, seed=seed)
    rounds, s = sim.run()
    assert len(rounds) == 6
    assert all(r["solver_status"] in ("Optimal", "NoEdges") for r in rounds)
    assert 0 <= s["fill_overall"] <= 1


def _toy():
    # 2 suppliers, 2 requesters, all 4 edges; requester 1 has 50 units of excess
    return RoundProblem(e_sup=np.array([0, 0, 1, 1]), e_req=np.array([0, 1, 0, 1]),
                        w=np.array([3.0, 1.0, 2.0, 1.5]), A=np.array([150, 200]),
                        C=np.array([120, 70]), X=np.array([0, 50]))


def test_integral_and_no_idle_edge():
    P = _toy()
    sol = solve_round(P)
    assert sol.x.dtype.kind == "i"
    sup = np.bincount(P.e_sup, weights=sol.x, minlength=2)
    req = np.bincount(P.e_req, weights=sol.x, minlength=2)
    D = P.C + P.X
    assert (sup <= P.A).all() and (req <= D).all()
    idle = (P.A[P.e_sup] - sup[P.e_sup] > 0) & (D[P.e_req] - req[P.e_req] > 0)
    assert not idle.any()                                          # README 7.1


def test_optimal_against_scipy():
    rng = np.random.default_rng(0)
    for _ in range(10):
        nS, nR, E = 6, 8, 30
        e_sup, e_req = rng.integers(0, nS, E), rng.integers(0, nR, E)
        A, C, X = rng.integers(0, 500, nS), rng.integers(0, 600, nR), rng.integers(0, 300, nR)
        P = RoundProblem(e_sup=e_sup, e_req=e_req, w=rng.random(E) * 100, A=A, C=C, X=X)
        sol = solve_round(P)
        # independent formulation: core and excess flow per edge, solved with scipy
        M = np.zeros((nS + 2 * nR, 2 * E))
        M[e_sup, np.arange(E)] = 1
        M[e_sup, E + np.arange(E)] = 1
        M[nS + e_req, np.arange(E)] = 1
        M[nS + nR + e_req, E + np.arange(E)] = 1
        ref = linprog(-np.r_[P.w, P.eps * P.w], A_ub=M, b_ub=np.r_[A, C, X],
                      bounds=[(0, None)] * (2 * E), method="highs")
        assert abs(-ref.fun - sol.objective) <= 1e-6 * max(1, abs(ref.fun))   # README 7.3


def test_urgency_dominates_locality():
    # one supplier with 100 units; a level-1 same-state requester and a
    # level-2 cross-state requester. With W = (1, 10, 100) level 2 must win.
    w_l1_same = 1 * 0.40 - 0.01
    w_l2_cross = 10 * 0.16 - 0.01
    P = RoundProblem(e_sup=np.array([0, 0]), e_req=np.array([0, 1]),
                     w=np.array([w_l1_same, w_l2_cross]), A=np.array([100]),
                     C=np.array([100, 100]), X=np.array([0, 0]))
    sol = solve_round(P)
    assert sol.x.tolist() == [0, 100]


def test_core_before_excess():
    # one supplier with 100 units. Requester 0: level 1, 60 core. Requester 1: level 3,
    # 0 core and 100 excess (an over-asker). Core must be served first, even though
    # requester 1 is more urgent: 60 core to requester 0, the remaining 40 as excess.
    P = RoundProblem(e_sup=np.array([0, 0]), e_req=np.array([0, 1]),
                     w=np.array([1 * 0.40 - 0.01, 100 * 0.40 - 0.01]),
                     A=np.array([100]), C=np.array([60, 0]), X=np.array([0, 100]))
    sol = solve_round(P)
    assert sol.xc.tolist() == [60, 0]
    assert sol.xe.tolist() == [0, 40]


def test_excess_by_value_when_core_done():
    # all core met; leftover goes to excess by value (urgency): level 3 first
    P = RoundProblem(e_sup=np.array([0, 0]), e_req=np.array([0, 1]),
                     w=np.array([0.39, 39.99]), A=np.array([100]),
                     C=np.array([10, 10]), X=np.array([100, 100]))
    sol = solve_round(P)
    assert sol.xc.tolist() == [10, 10]
    assert sol.xe.tolist() == [0, 80]
