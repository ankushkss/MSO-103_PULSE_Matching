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


def test_core_excess_split(small_cfg, districts):
    """Over-askers get excess; core = remaining genuine need; inflating gains no core."""
    cfg = small_cfg.with_(p_bulk=0.2)                  # many bulk requesters, all over-askers
    sim = Simulation(cfg, districts, seed=2)
    rounds, s = sim.run()
    assert sum(r["excess_demand"] for r in rounds) > 0
    assert sum(r["core_allocated"] for r in rounds) > 0
    # every round's split adds up: core + excess = outstanding demand
    for r in rounds:
        assert r["core_demand"] + r["excess_demand"] == r["D_tot"]
    assert s["over_askers"] > 0


def test_core_off_treats_everything_as_core(small_cfg, districts):
    rounds, _ = Simulation(small_cfg.with_(core_excess=False, p_bulk=0.2), districts, seed=2).run()
    assert all(r["excess_demand"] == 0 for r in rounds)


def test_inflating_a_request_gains_no_core(small_cfg, districts):
    """Doubling one requester's request leaves its core demand unchanged."""
    from pulse.generator import generate_instance
    inst = generate_instance(small_cfg, districts, seed=5)
    j = int(np.argmax(inst.r_arrival == 1))
    base = Simulation(small_cfg, districts, seed=5, instance=inst)
    need_j = base.need[j]
    inst2 = generate_instance(small_cfg, districts, seed=5)
    inst2.r_D0 = inst2.r_D0.copy()
    inst2.r_D0[j] *= 2
    big = Simulation(small_cfg, districts, seed=5, instance=inst2)
    core1 = min(inst.r_D0[j], np.floor(need_j))
    core2 = min(inst2.r_D0[j], np.floor(big.need[j]))
    assert core1 == core2 or inst.r_D0[j] < need_j <= inst2.r_D0[j]
    assert big.need[j] == need_j


def test_instance_independent_of_policy(small_cfg, districts):
    a = generate_instance(small_cfg, districts, seed=9)
    b = generate_instance(small_cfg.with_(H=7.0, W=(1, 2, 3),
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


def test_common_random_numbers_across_policies(small_cfg, districts):
    """A match made under two different policies must turn out the same in both."""
    runs = []
    for cfg in (small_cfg, small_cfg.with_(H=7.0, W=(1.0, 2.0, 3.0), eta=0.0)):
        sim = Simulation(cfg, districts, seed=3)
        sim.run()
        m = sim.matches
        runs.append({(int(t), int(i), int(j)): bool(dv > 0)
                     for t, i, j, dv in zip(m.get("tau"), m.get("sup"), m.get("req"), m.get("delivered"))})
    shared = set(runs[0]) & set(runs[1])
    assert len(shared) > 20
    assert all(runs[0][k] == runs[1][k] for k in shared)


def test_refused_matches_never_deliver(small_cfg, districts):
    """Without the model's filters, over-budget or uncertified-when-required matches fail."""
    cfg = small_cfg.with_(price_filter=False, cert_filter=False)
    sim = Simulation(cfg, districts, seed=1)
    rounds, _ = sim.run()
    assert sum(r["refused_matches"] for r in rounds) > 0
    inst, m = sim.inst, sim.matches
    sup, req, dv = m.get("sup").astype(int), m.get("req").astype(int), m.get("delivered")
    bad = (inst.s_price[sup] > inst.r_pmax[req]) | ((inst.r_kappa[req] == 2) & ~inst.s_cert[sup])
    assert bad.any() and (dv[bad] == 0).all()
    # with the filters on, nothing is ever refused
    rounds_on, _ = Simulation(small_cfg, districts, seed=1).run()
    assert sum(r["refused_matches"] for r in rounds_on) == 0
