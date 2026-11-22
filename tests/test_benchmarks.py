import numpy as np
import pytest

from pulse import Simulation
from pulse.benchmarks import heuristic_round, hindsight
from pulse.model import RoundProblem


def _toy():
    # supplier 0 reaches both requesters, supplier 1 only requester 1
    return RoundProblem(e_sup=np.array([0, 0, 1]), e_req=np.array([0, 1, 1]),
                        w=np.array([1.0, 1.0, 1.0]), A=np.array([100, 30]),
                        C=np.array([80, 60]), X=np.array([0, 40]))


@pytest.mark.parametrize("policy", ["fcfs", "proportional", "random"])
def test_heuristic_respects_limits_and_serves_core_first(policy):
    P = _toy()
    sol = heuristic_round(policy, P, km=np.array([10.0, 20.0, 5.0]),
                          req_key=np.array([1, 2]), rng=np.random.default_rng(0))
    sup = np.bincount(P.e_sup, weights=sol.x, minlength=2)
    core = np.bincount(P.e_req, weights=sol.xc, minlength=2)
    exc = np.bincount(P.e_req, weights=sol.xe, minlength=2)
    assert (sup <= P.A).all() and (core <= P.C).all() and (exc <= P.X).all()
    assert sup.sum() == 130                       # total core 140 > stock 130: all stock used on core
    assert exc.sum() == 0


def test_fcfs_serves_earliest_requester_first():
    P = _toy()
    sol = heuristic_round("fcfs", P, km=np.array([10.0, 20.0, 5.0]),
                          req_key=np.array([1, 2]), rng=np.random.default_rng(0))
    assert sol.xc[0] == 80                        # requester 0 arrived first: fully served


def test_proportional_equalises_fill():
    P = RoundProblem(e_sup=np.array([0, 0]), e_req=np.array([0, 1]), w=np.ones(2),
                     A=np.array([50]), C=np.array([60, 40]), X=np.array([0, 0]))
    sol = heuristic_round("proportional", P, km=np.ones(2), req_key=np.zeros(2, int),
                          rng=np.random.default_rng(0))
    assert tuple(sol.xc) == (30, 20)              # both get 50% of their core


@pytest.mark.parametrize("policy", ["fcfs", "proportional", "random"])
def test_heuristic_policies_run_with_invariants(small_cfg, districts, policy):
    rounds, s = Simulation(small_cfg.with_(policy=policy), districts, seed=1).run()
    assert len(rounds) == 6 and 0 <= s["need_fill"] <= 1


def test_hindsight_bounds_every_policy(small_cfg, districts):
    _, hs = hindsight(small_cfg, districts, seed=2, outcomes="realised")
    for policy in ("lp", "fcfs", "proportional", "random"):
        _, s = Simulation(small_cfg.with_(policy=policy), districts, seed=2).run()
        for m in ("need_fill", "wneed_fill_ref", "wneed_fill_w123"):
            assert s[m] <= hs[m] + 1e-6, (policy, m, s[m], hs[m])


def test_hindsight_is_reproducible(small_cfg, districts):
    assert hindsight(small_cfg, districts, 3)[1]["need_fill"] == \
        hindsight(small_cfg, districts, 3)[1]["need_fill"]


def test_fluid_bound_is_a_valid_fill(small_cfg, districts):
    # NB: H1 (fluid) and H2 (perfect hindsight) bound different things, so neither has
    # to dominate the other on one seed: the fluid LP may spread fractional shipments
    # over every edge, H2 only over edges whose coin flip succeeded (CHANGES.md D11).
    _, h1 = hindsight(small_cfg, districts, 1, outcomes="expected")
    for m in ("need_fill", "wneed_fill_ref", "wneed_fill_w123"):
        assert 0 < h1[m] <= 1 + 1e-9


def test_fluid_bound_above_mean_policy(small_cfg, districts):
    # in expectation no policy beats the fluid bound; check on a seed average
    seeds = range(1, 6)
    pol = np.mean([Simulation(small_cfg, districts, s).run()[1]["need_fill"] for s in seeds])
    h1 = np.mean([hindsight(small_cfg, districts, s, outcomes="expected")[1]["need_fill"]
                  for s in seeds])
    assert pol <= h1 + 1e-6
