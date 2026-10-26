import numpy as np
import pytest

from pulse import Config, Simulation
from pulse.generator import generate_instance


def test_conservation(small_cfg, districts):
    cfg = small_cfg.with_(omega=0.0)
    sim = Simulation(cfg, districts, seed=4)
    sim.run()
    inst = sim.inst
    arrived = sim.status != 0
    # every requester's demand is either delivered or still outstanding
    np.testing.assert_array_equal(inst.r_D0[arrived], (sim.delivered + sim.carry)[arrived])
    assert sim.delivered.sum() == sim.matches.get("delivered").sum()


def test_cap_binds_flagged(small_cfg, districts):
    cfg = small_cfg.with_(p_bulk=0.2, nu=0.0)          # many bulk (flagged) requesters
    sim = Simulation(cfg, districts, seed=2)
    rounds, _ = sim.run()
    assert sum(r["capped"] for r in rounds) > 0
    for r in rounds:
        assert r["units_deferred_by_cap"] >= 0


def test_instance_independent_of_policy(small_cfg, districts):
    a = generate_instance(small_cfg, districts, seed=9)
    b = generate_instance(small_cfg.with_(c=0.2, W=(1, 2, 3), enforce_min_shipment=True,
                                          omega=0.0), districts, seed=9)
    np.testing.assert_array_equal(a.r_D0, b.r_D0)
    np.testing.assert_array_equal(a.s_rate, b.s_rate)


def test_learning_moves_toward_truth(small_cfg, districts):
    cfg = small_cfg.with_(psi_true_same=0.9, psi_true_cross=0.05, n0=5.0)
    sim = Simulation(cfg, districts, seed=1)
    sim.run()
    psi = sim.psi_hat()
    assert psi[0] > cfg.psi0_same and psi[1] < cfg.psi0_cross


def test_learning_off_keeps_prior(small_cfg, districts):
    sim = Simulation(small_cfg.with_(learn_psi=False), districts, seed=1)
    sim.run()
    np.testing.assert_allclose(sim.psi_hat(), [0.40, 0.16])


def test_c1_violation_rejected(districts):
    with pytest.raises(ValueError):
        Simulation(Config(mu=0.5), districts, seed=1)


def test_reproducible(small_cfg, districts):
    _, s1 = Simulation(small_cfg, districts, seed=7).run()
    _, s2 = Simulation(small_cfg, districts, seed=7).run()
    for k in s1:
        if k in ("solve_seconds",):
            continue                                       # wall-clock time differs
        same = s1[k] == s2[k] or (s1[k] != s1[k] and s2[k] != s2[k])   # NaN == NaN
        assert same, k
