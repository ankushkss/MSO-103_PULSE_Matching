"""Idea A (misreporting, fairness) and idea C (supplier reliability): CHANGES.md session 4."""
import numpy as np
import pytest

from pulse import Simulation, generate_instance


def test_cohort_inflates_request_and_staff_only_for_cohort(small_cfg, districts):
    honest = generate_instance(small_cfg, districts, 4)
    liar = generate_instance(small_cfg.with_(misreport_factor=3.0), districts, 4)
    c = liar.r_cohort
    assert c.any() and (c == honest.r_cohort).all()          # same cohort in both worlds
    np.testing.assert_array_equal(liar.r_D0_true, honest.r_D0)
    np.testing.assert_array_equal(liar.r_size[~c], honest.r_size[~c])
    assert (liar.r_size[c] >= 3 * honest.r_size[c]).all()
    np.testing.assert_array_equal(liar.r_D0[c], np.rint(3 * honest.r_D0[c]))


def test_misreporting_never_moves_the_yardstick(small_cfg, districts):
    s1 = Simulation(small_cfg, districts, 2)
    s3 = Simulation(small_cfg.with_(misreport_factor=3.0), districts, 2)
    np.testing.assert_array_equal(s1.need_true, s3.need_true)
    assert (s3.need[s3.inst.r_cohort] > s1.need[s1.inst.r_cohort]).all()   # platform is fooled


def test_true_psi_without_reliability_is_class_truth(small_cfg, districts):
    sim = Simulation(small_cfg, districts, 1)
    cls = np.array([0, 1, 0])
    np.testing.assert_allclose(sim.true_psi(np.array([0, 1, 2]), cls), [0.40, 0.16, 0.40])


def test_reliability_shifts_suppliers_but_keeps_draws(small_cfg, districts):
    a = generate_instance(small_cfg.with_(reliability_sigma=0.5), districts, 3)
    b = generate_instance(small_cfg.with_(reliability_sigma=1.0), districts, 3)
    np.testing.assert_allclose(b.s_theta, 2 * a.s_theta)     # same draws, rescaled
    np.testing.assert_array_equal(a.r_D0, b.r_D0)


@pytest.mark.parametrize("mode", ["supplier", "supplier_eb", "thompson", "oracle"])
def test_learning_modes_run_with_invariants(small_cfg, districts, mode):
    cfg = small_cfg.with_(reliability_sigma=1.0, learn_mode=mode)
    rounds, s = Simulation(cfg, districts, 1).run()
    assert len(rounds) == 6 and 0 <= s["need_fill"] <= 1


def test_fairness_boost_runs_with_invariants(small_cfg, districts):
    _, s = Simulation(small_cfg.with_(fair_lambda=2.0), districts, 1).run()
    assert 0 <= s["gini_need_fill"] <= 1


@pytest.mark.parametrize("rule,expect", [("truth", 1.0), ("fraction=0.5", 0.5), ("zero", 0.0)])
def test_audit_rules_set_core_basis_of_caught_liars(small_cfg, districts, rule, expect):
    sim = Simulation(small_cfg.with_(misreport_factor=3.0, audit_prob=1.0, audit_rule=rule),
                     districts, 1)
    c = sim.caught
    assert c.any() and (c == sim.inst.r_cohort).all()        # audit everyone: every liar caught
    np.testing.assert_allclose(sim.need[c], expect * sim.need_true[c])
    np.testing.assert_array_equal(sim.need[~c], sim.need_true[~c])   # honest untouched


def test_audit_rule_rejects_unknown():
    from pulse.simulation import apply_audit
    with pytest.raises(ValueError):
        apply_audit(np.ones(2), np.ones(2), "fraction=2")


def test_eb_prior_strength_large_when_suppliers_alike(small_cfg, districts):
    sim = Simulation(small_cfg, districts, 1)
    rng = np.random.default_rng(0)
    sim.E_i = np.full(sim.inst.n_suppliers, 20.0)
    sim.S_i = rng.poisson(20.0, sim.inst.n_suppliers).astype(float)     # all alike: pure noise
    assert sim.eb_prior_strength() > 50
    sim.S_i = sim.E_i * rng.choice([0.3, 1.7], sim.inst.n_suppliers)    # clearly different
    assert sim.eb_prior_strength() < 5


# ---------------- session 7: overbooking ----------------
def test_overbooking_zero_is_identical(small_cfg, districts):
    _, a = Simulation(small_cfg, districts, 3).run()
    _, b = Simulation(small_cfg.with_(overbook=0.0), districts, 3).run()
    assert a["need_fill"] == b["need_fill"] and b["units_returned"] == 0


@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_overbooking_never_delivers_beyond_request(small_cfg, districts, beta):
    sim = Simulation(small_cfg.with_(overbook=beta), districts, 2)
    _, s = sim.run()                                   # invariants checked every round
    arrived = sim.status != 0
    assert (sim.delivered[arrived] <= sim.inst.r_D0[arrived]).all()
    assert (sim.carry >= 0).all() and s["units_returned"] >= 0
    m = sim.matches
    assert (m.get("delivered") <= m.get("x")).all()
