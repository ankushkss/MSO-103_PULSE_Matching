"""Properties from README Section 7. The simulation also asserts them every round
(check_invariants=True), so a clean run is itself a test; these add direct checks."""
import numpy as np
import pytest

from pulse import Simulation
from pulse.model import RoundProblem, greedy_mode_b, is_feasible_mode_b, solve_round


def run(cfg, districts, seed=1):
    sim = Simulation(cfg, districts, seed=seed)
    rounds, summary = sim.run()
    return sim, rounds, summary


@pytest.mark.parametrize("seed", [1, 2])
def test_mode_a_runs_with_all_invariants(small_cfg, districts, seed):
    sim, rounds, s = run(small_cfg, districts, seed)
    assert len(rounds) == 6
    assert all(r["solver_status"] in ("Optimal", "NoEdges") for r in rounds)
    assert 0 <= s["fill_overall"] <= 1


@pytest.mark.parametrize("seed", [1, 2])
def test_mode_b_runs_with_all_invariants(small_cfg, districts, seed):
    cfg = small_cfg.with_(enforce_min_shipment=True)
    sim, rounds, s = run(cfg, districts, seed)
    x = sim.matches.get("x")
    sup = sim.matches.get("sup").astype(int)
    assert (x >= sim.inst.s_moq[sup]).all()                        # strict minimum order
    active = sim.status == 1
    assert (sim.carry[active] >= cfg.m_min).all()                  # README 7.4 invariant
    for r in rounds:                                               # README 7.5 bound
        if r["z_lp_bound"] is not None:
            assert r["objective"] <= r["z_lp_bound"] + 1e-6


def _toy(mode_b_moq=100):
    # 2 suppliers, 2 requesters, all 4 edges
    return RoundProblem(e_sup=np.array([0, 0, 1, 1]), e_req=np.array([0, 1, 0, 1]),
                        w=np.array([3.0, 1.0, 2.0, 1.5]), xbar=np.array([150, 120, 120, 120]),
                        moq=np.full(4, mode_b_moq), A=np.array([150, 200]),
                        Dt=np.array([120, 120]), D=np.array([120, 120]), m_min=50)


def test_mode_a_is_integral_and_maximal():
    P = _toy()
    sol = solve_round(P, mode_b=False)
    assert sol.x.dtype.kind == "i"
    sup = np.bincount(P.e_sup, weights=sol.x, minlength=2)
    req = np.bincount(P.e_req, weights=sol.x, minlength=2)
    idle = (P.A[P.e_sup] - sup[P.e_sup] > 0) & (P.Dt[P.e_req] - req[P.e_req] > 0)
    assert not idle.any()                                          # README 7.1


def test_mode_b_respects_moq_and_remainder():
    P = _toy()
    sol = solve_round(P, mode_b=True, candidate_k=0)
    assert is_feasible_mode_b(P, sol.x)
    req = np.bincount(P.e_req, weights=sol.x, minlength=2)
    rem = P.D - req
    assert ((rem == 0) | (rem >= 50)).all()
    assert sol.objective <= sol.z_lp + 1e-9


def test_remainder_rule_example():
    # one requester needing 120, one supplier with minimum order 100 and plenty of stock:
    # shipping 100 would leave 20 (< 50), so the only valid choices are 0 or all 120.
    P = RoundProblem(e_sup=np.array([0]), e_req=np.array([0]), w=np.array([1.0]),
                     xbar=np.array([120]), moq=np.array([100]), A=np.array([500]),
                     Dt=np.array([120]), D=np.array([120]), m_min=50)
    sol = solve_round(P, mode_b=True, candidate_k=0)
    assert sol.x[0] == 120
    assert greedy_mode_b(P)[0] == 120


def test_capped_requester_cannot_close():
    # capped at 60 of 200: may receive at most 60, and remainder 140 >= 50 is fine
    P = RoundProblem(e_sup=np.array([0]), e_req=np.array([0]), w=np.array([1.0]),
                     xbar=np.array([60]), moq=np.array([50]), A=np.array([500]),
                     Dt=np.array([60]), D=np.array([200]), m_min=50)
    sol = solve_round(P, mode_b=True, candidate_k=0)
    assert sol.x[0] == 60


def test_greedy_is_always_feasible(small_cfg, districts):
    rng = np.random.default_rng(5)
    for _ in range(20):
        E = 40
        P = RoundProblem(e_sup=rng.integers(0, 6, E), e_req=rng.integers(0, 8, E),
                         w=rng.random(E), xbar=np.zeros(E, int), moq=rng.choice([50, 100, 200], E),
                         A=rng.integers(0, 600, 6), Dt=np.zeros(8, int), D=rng.integers(50, 700, 8),
                         m_min=50)
        P.Dt[:] = P.D
        P.xbar[:] = np.minimum(P.A[P.e_sup], P.Dt[P.e_req])
        assert is_feasible_mode_b(P, greedy_mode_b(P))
