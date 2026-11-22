"""The six-round PULSE simulation (README Sections 4, 5, 8 and 10)."""
from __future__ import annotations

import logging

import numpy as np

from .config import Config
from .districts import Districts
from .generator import Instance, generate_instance, make_streams
from .metrics import MatchLog, round_record, run_summary
from .benchmarks import heuristic_round
from .model import RoundProblem, solve_round

log = logging.getLogger("pulse")

PENDING, ACTIVE, SERVED, ABANDONED = 0, 1, 2, 3
SAME, CROSS = 0, 1
STRAND_CAUSES = ("variant", "price", "certificate", "distance", "no_stock")


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_audit(need_declared: np.ndarray, need_true: np.ndarray, rule: str) -> np.ndarray:
    """Core basis (units of genuine need over H days) for requesters an audit CAUGHT
    over-declaring their staff count. Inputs are the caught requesters only; return
    one value per caught requester. Called once, before round 1 (idea A, CHANGES.md D21).
    """
    # Rules (CHANGES.md D27):
    #   "truth"        core basis reset to the true need: fair, but lying costs nothing if caught
    #   "fraction=f"   only f x true need is honoured (0 <= f <= 1): a proportional penalty
    #   "zero"         no core at all: the whole request waits behind everyone's core
    need_true = np.asarray(need_true, dtype=float)
    if rule == "truth":
        return need_true.copy()
    if rule == "zero":
        return np.zeros_like(need_true)
    if rule.startswith("fraction="):
        f = float(rule.split("=", 1)[1])
        if not 0 <= f <= 1:
            raise ValueError(f"audit fraction must be in [0, 1], got {f}")
        return f * need_true
    raise ValueError(f"unknown audit_rule {rule!r} (use 'truth', 'fraction=<f>' or 'zero')")


class Simulation:
    def __init__(self, cfg: Config, districts: Districts, seed: int,
                 instance: Instance | None = None):
        cfg.validate()
        self.cfg = cfg
        self.d = districts
        self.seed = seed
        streams = make_streams(seed)
        self.inst = instance if instance is not None else generate_instance(cfg, districts, seed, streams)
        # Common random numbers: one pre-drawn uniform per (round, supplier, requester)
        # for match outcomes, and per (round, requester) for abandonment.
        # A given match or exit then turns out the same under every
        # policy, so paired comparisons differ only because the decisions differ.
        T = cfg.n_rounds
        nS, nR = self.inst.n_suppliers, self.inst.n_requesters
        self.U_out = streams["outcomes"].random((T, nS, nR), dtype=np.float32)
        self.U_ab = streams["abandonment"].random((T, nR))

        inst = self.inst
        self.r = np.zeros(inst.n_suppliers, dtype=np.int64)          # carried stock
        self.carry = np.zeros(inst.n_requesters, dtype=np.int64)     # outstanding demand
        self.status = np.full(inst.n_requesters, PENDING, dtype=np.int8)
        # The platform's core basis: need over H days from the DECLARED staff count.
        self.need = inst.r_usage * inst.r_size * cfg.H
        # The truth (world and metrics only): need from the TRUE staff count.
        self.need_true = inst.r_usage * inst.r_size_true * cfg.H
        self.over_asker = inst.r_D0_true > self.need_true            # honest request above need
        # Idea A: audits, drawn from pre-drawn uniforms so other draws are unaffected.
        audited = inst.r_U_audit < cfg.audit_prob
        self.caught = audited & (inst.r_size > inst.r_size_true)
        if self.caught.any():
            self.need = self.need.astype(float)
            self.need[self.caught] = apply_audit(self.need[self.caught],
                                                 self.need_true[self.caught], cfg.audit_rule)
        self.delivered = np.zeros(inst.n_requesters, dtype=np.int64)
        self.abandoned_units = np.zeros(inst.n_requesters, dtype=np.int64)
        self.S = np.zeros(2, dtype=np.int64)                          # successes by class
        self.F = np.zeros(2, dtype=np.int64)                          # failures by class
        # Idea C: per-supplier evidence. E_i = successes expected under the class
        # estimate at the time of each match, so S_i / E_i estimates relative reliability.
        self.S_i = np.zeros(inst.n_suppliers)
        self.E_i = np.zeros(inst.n_suppliers)
        self.W = np.asarray(cfg.W, dtype=float)
        self.matches = MatchLog()
        self.rounds: list[dict] = []
        self.units_returned = 0          # successful shipments refused as surplus (overbooking)

    # ------------------------------------------------------------------
    def psi_hat(self) -> np.ndarray:
        cfg = self.cfg
        if not cfg.use_psi_objective:
            return np.array([1.0, 1.0])
        prior = np.array([cfg.psi0_same, cfg.psi0_cross])
        if not (cfg.learn_psi and cfg.simulate_failures):
            return prior
        post = (cfg.n0 * prior + self.S) / (cfg.n0 + self.S + self.F)
        return np.maximum(cfg.psi_floor, post)

    def true_psi(self, sup: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """TRUE success probability per pair: class value shifted by the supplier effect."""
        p = np.asarray(self.cfg.psi_true, dtype=float)[cls]
        theta = self.inst.s_theta[sup]
        if not np.any(theta):
            return p
        return _expit(_logit(p) + theta)

    def edge_psi(self, sup: np.ndarray, cls: np.ndarray, tau: int) -> np.ndarray:
        """The platform's success estimate per pair (idea C: per-supplier modes)."""
        cfg = self.cfg
        base = self.psi_hat()[cls]
        if not cfg.use_psi_objective or cfg.learn_mode == "class":
            return base
        if cfg.learn_mode == "oracle":
            est = self.true_psi(sup, cls)
        else:
            if not (cfg.learn_psi and cfg.simulate_failures):
                return base
            n0 = self.eb_prior_strength() if cfg.learn_mode == "supplier_eb" else cfg.n_supplier_prior
            a = n0 + self.S_i
            b = n0 + self.E_i
            if cfg.learn_mode in ("supplier", "supplier_eb"):
                ratio = a / b                                    # posterior mean reliability
            else:                                                # "thompson": one draw per supplier
                rng = np.random.default_rng([self.seed, tau, 104729])
                ratio = rng.gamma(a, 1.0 / b)
            est = base * ratio[sup]
        return np.clip(est, cfg.psi_floor, cfg.psi_cap)

    def eb_prior_strength(self, min_suppliers: int = 10, min_exposure: float = 0.5) -> float:
        """Empirical-Bayes prior strength n0 for supplier reliability (CHANGES.md D28).

        Reliability ratio r_i = S_i / E_i has, across suppliers, variance
            Var(r_i) ~ tau^2 + mean(r) / E_i         (true spread + Poisson noise),
        so tau^2 = Var(r_i) - mean(mean(r) / E_i). A Gamma(n0, n0) prior has variance
        1 / n0, hence n0 = 1 / tau^2: suppliers that look alike (tau^2 ~ 0) give a huge
        n0 (trust the class), clearly different suppliers give a small n0 (trust each
        supplier's record). Falls back to cfg.n_supplier_prior until there is data.
        """
        seen = self.E_i >= min_exposure
        if seen.sum() < min_suppliers:
            return self.cfg.n_supplier_prior
        r = self.S_i[seen] / self.E_i[seen]
        rbar = r.mean()
        tau2 = r.var(ddof=1) - np.mean(rbar / self.E_i[seen])
        return float(np.clip(1.0 / max(tau2, 1e-3), 1.0, 1000.0))

    def c2_holds(self, psi: np.ndarray) -> bool:
        """Urgency dominance across every pair of adjacent levels (README 7.2)."""
        cmax = self.cfg.tie_break_max
        return all(self.W[l + 1] * psi[CROSS] - cmax > self.W[l] * psi[SAME] for l in range(2))

    # ------------------------------------------------------------------
    def run(self) -> tuple[list[dict], dict]:
        for tau in range(1, self.cfg.n_rounds + 1):
            self.rounds.append(self._round(tau))
        summary = run_summary(self)
        return self.rounds, summary

    def _round(self, tau: int) -> dict:
        cfg, inst, d = self.cfg, self.inst, self.d
        gap = cfg.gaps[tau - 1]

        # 1-2. enrolment and supply
        sup = np.nonzero(inst.s_enrol <= tau)[0]
        A = inst.s_rate[sup] * gap + self.r[sup]
        A_tot = int(A.sum())

        # arrivals
        new = np.nonzero(inst.r_arrival == tau)[0]
        self.status[new] = ACTIVE
        self.carry[new] = inst.r_D0[new]

        # 3. outstanding demand
        req = np.nonzero(self.status == ACTIVE)[0]
        D = self.carry[req].copy()

        # 4. split into core (genuine need not yet received) and excess (the rest)
        if cfg.core_excess:
            room = np.maximum(0, np.floor(self.need[req] - self.delivered[req] + 1e-9)).astype(np.int64)
            C = np.minimum(D, room)
        else:
            C = D.copy()
        X = D - C

        # beliefs
        psi = self.psi_hat()
        c2_ok = self.c2_holds(psi)
        if not c2_ok:
            # Expected whenever W is not lexicographic-like, e.g. W = (1, 2, 3).
            # Recorded per round as `c2_ok` and per run as `c2_failures`.
            log.debug("round %d: (C2) fails with psi_hat = %s", tau, np.round(psi, 3))

        # 6. eligibility (vectorised over suppliers x requesters)
        sd, rd = inst.s_dist[sup], inst.r_dist[req]
        compat = (inst.s_var[sup][:, None] & inst.r_var[req][None, :]) != 0
        ones = np.ones((len(sup), len(req)), dtype=bool)
        price_ok = (inst.s_price[sup][:, None] <= inst.r_pmax[req][None, :]) if cfg.price_filter else ones
        cert_ok = ((inst.r_kappa[req] != 2)[None, :] | inst.s_cert[sup][:, None]) if cfg.cert_filter else ones
        Rsub = d.R[np.ix_(sd, rd)]
        dist_ok = np.isfinite(Rsub) & (Rsub <= cfg.delta_max)
        stock_ok = np.broadcast_to((A > 0)[:, None], ones.shape)
        elig = compat & price_ok & cert_ok & dist_ok & stock_ok

        # stranded requesters, by first failing filter
        stranded = {c: 0 for c in STRAND_CAUSES}
        if len(req):
            chain = [compat, price_ok, cert_ok, dist_ok, stock_ok]
            cum = np.ones(len(req), dtype=bool) & (len(sup) > 0)
            m = ones
            has_edge = elig.any(axis=0) if len(sup) else np.zeros(len(req), bool)
            for cause, f in zip(STRAND_CAUSES, chain):
                m = m & f
                ok_now = m.any(axis=0) if len(sup) else np.zeros(len(req), bool)
                stranded[cause] += int((cum & ~ok_now & ~has_edge).sum())
                cum = cum & ok_now
        n_stranded = sum(stranded.values())

        ei, ej = np.nonzero(elig)

        # 7. coefficients and bounds
        Rk = Rsub[ei, ej]
        chat = Rk / d.R_max
        rho = inst.s_price[sup][ei] / inst.r_pmax[req][ej]
        q = inst.s_cert[sup][ei]
        kap = inst.r_kappa[req][ej]
        cvec = cfg.alpha * chat + cfg.beta * rho + cfg.mu * ((kap == 1) & ~q)
        same = d.state[sd[ei]] == d.state[rd[ej]]
        cls = np.where(same, SAME, CROSS)
        level = inst.r_level[req][ej]
        psi_e = self.edge_psi(sup[ei], cls, tau)
        wvec = self.W[level - 1] * psi_e - cvec                      # value of an excess unit
        w_core = wvec
        if cfg.fair_lambda > 0:
            # Idea A: boost the core value of requesters that have received little of their
            # need so far (platform's view). Boost >= 1, so (C1)-(C3) still hold.
            basis = self.need[req]
            got = np.where(basis > 0, np.minimum(1.0, self.delivered[req] / np.maximum(basis, 1e-9)), 1.0)
            boost = 1.0 + cfg.fair_lambda * (1.0 - got)
            w_core = self.W[level - 1] * psi_e * boost[ej] - cvec
        if cfg.check_invariants and len(wvec):
            assert (wvec > 0).all(), "(C1) violated: an edge has non-positive value"
        # Overbooking: the platform may ship up to (1 + beta) x the outstanding core and
        # excess, expecting most shipments to fail. Rows stay integer, so the LP stays TU.
        if cfg.overbook > 0:
            Cm = np.floor((1 + cfg.overbook) * C + 1e-9).astype(np.int64)
            Xm = np.floor((1 + cfg.overbook) * X + 1e-9).astype(np.int64)
        else:
            Cm, Xm = C, X
        P = RoundProblem(e_sup=ei, e_req=ej, w=w_core, A=A, C=Cm, X=Xm, eps=cfg.excess_weight,
                         w_exc=None if w_core is wvec else wvec)
        if cfg.policy == "lp":
            sol = solve_round(P, solver=cfg.solver)
        else:
            # benchmark rule on the same edges, stock and core/excess rows (pulse/benchmarks.py);
            # its own RNG stream so the instance and outcome draws are untouched
            rng = np.random.default_rng([self.seed, tau, 7919])
            sol = heuristic_round(cfg.policy, P, Rk, inst.r_arrival[req], rng)
        x = sol.x

        # 8.1 realise matches (outcomes from the TRUE psi, never psi_hat)
        matched = x > 0
        B = np.ones(len(x), dtype=bool)
        if cfg.simulate_failures and matched.any():
            draws = self.U_out[tau - 1, sup[ei[matched]], req[ej[matched]]]
            B[matched] = draws < self.true_psi(sup[ei[matched]], cls[matched])
        # A requester refuses stock above its budget, or uncertified stock when it
        # requires certification. This is part of the world, so it applies even when
        # the model's own filters are switched off (e.g. the Shield-Net baseline).
        refused = matched & ((inst.s_price[sup][ei] > inst.r_pmax[req][ej])
                             | ((inst.r_kappa[req][ej] == 2) & ~inst.s_cert[sup][ei]))
        B[refused] = False
        deliv = np.where(B, x, 0)
        if cfg.overbook > 0 and len(deliv):
            # A requester accepts at most its outstanding demand D_j. Surplus successes are
            # refused (nearest supplier accepted first) and stay with the supplier as stock.
            order = np.lexsort((Rk, ej))
            d_o, j_o = deliv[order], ej[order]
            cum = np.cumsum(d_o)
            start = np.zeros(len(D) + 1, dtype=np.int64)
            np.add.at(start, j_o + 1, d_o)
            before = cum - d_o - np.cumsum(start)[j_o]        # accepted earlier, same requester
            acc = np.clip(D[j_o] - before, 0, d_o)
            accepted = np.empty_like(deliv)
            accepted[order] = acc
            self.units_returned += int((deliv - accepted).sum())
            deliv = accepted

        # 8.2 learn (counts matches, not units)
        if cfg.learn_psi and cfg.simulate_failures:
            # per-supplier evidence uses the class estimate in force when the match was made
            gsup = sup[ei[matched]]
            np.add.at(self.S_i, gsup, B[matched].astype(float))
            np.add.at(self.E_i, gsup, psi[cls[matched]])
            for g in (SAME, CROSS):
                sel = matched & (cls == g)
                self.S[g] += int((sel & B).sum())
                self.F[g] += int((sel & ~B).sum())

        nS, nR = len(sup), len(req)
        sup_alloc = np.bincount(ei, weights=x, minlength=nS).astype(np.int64)
        req_alloc = np.bincount(ej, weights=x, minlength=nR).astype(np.int64)
        sup_deliv = np.bincount(ei, weights=deliv, minlength=nS).astype(np.int64)
        req_deliv = np.bincount(ej, weights=deliv, minlength=nR).astype(np.int64)

        core_alloc = np.bincount(ej, weights=sol.xc, minlength=nR).astype(np.int64)
        exc_alloc = np.bincount(ej, weights=sol.xe, minlength=nR).astype(np.int64)
        if cfg.check_invariants:
            self._check(tau, A, Cm, Xm, ei, ej, sol.xe, sup_alloc, core_alloc, exc_alloc)

        # 8.3 suppliers: leakage and storage cap
        unsold = A - sup_deliv
        kept = np.floor(cfg.eta * unsold + 1e-9).astype(np.int64)
        r_new = np.minimum(kept, np.floor(cfg.h * inst.s_rate[sup]).astype(np.int64))
        self.r[sup] = r_new
        leaked = int((unsold - kept).sum())
        lost_storage = int((kept - r_new).sum())

        # 8.4 requesters
        self.carry[req] = D - req_deliv
        self.delivered[req] += req_deliv
        served = self.carry[req] == 0
        self.status[req[served]] = SERVED

        # 8.5 abandonment
        act = req[~served]
        leave = self.U_ab[tau - 1, act] < cfg.omega
        gone = act[leave]
        self.abandoned_units[gone] = self.carry[gone]
        self.status[gone] = ABANDONED

        # logging
        self.matches.add(tau=tau, sup=sup[ei[matched]], req=req[ej[matched]],
                         x=x[matched], delivered=deliv[matched], km=Rk[matched],
                         same_state=same[matched],
                         same_district=(sd[ei] == rd[ej])[matched],
                         price=inst.s_price[sup][ei][matched], cert=q[matched],
                         kappa=kap[matched], level=level[matched],
                         # expected-delivery weight: 0 for a refused match
                         psi_true=self.true_psi(sup[ei[matched]], cls[matched]) * ~refused[matched])
        return round_record(
            sim=self, tau=tau, A_tot=A_tot, D=D, C=C, X=X, req=req, n_sup=nS,
            core_alloc=core_alloc, exc_alloc=exc_alloc, x=x, deliv=deliv, matched=matched, B=B,
            psi=psi, c2_ok=c2_ok, sol=sol, n_edges=len(x), stranded=stranded, refused=refused,
            n_stranded=n_stranded, leaked=leaked, lost_storage=lost_storage,
            n_new=len(new), n_served=int(served.sum()), n_abandoned=int(leave.sum()),
            units_abandoned=int(self.abandoned_units[gone].sum()),
            same=same, Rk=Rk, cls=cls)

    # ------------------------------------------------------------------
    def _check(self, tau, A, C, X, ei, ej, xe, sup_alloc, core_alloc, exc_alloc):
        """README Section 7 properties, asserted every round."""
        assert (sup_alloc <= A).all(), f"round {tau}: supplier allocated beyond stock"
        assert (core_alloc <= C).all(), f"round {tau}: core allocated beyond core demand"
        assert (exc_alloc <= X).all(), f"round {tau}: excess allocated beyond excess demand"
        if len(ei):
            core_open = (C - core_alloc)[ej] > 0
            open_ = core_open | ((X - exc_alloc)[ej] > 0)
            # 7.1: no edge left idle with stock on one side and demand on the other
            bad = ((A - sup_alloc)[ei] > 0) & open_
            assert not bad.any(), f"round {tau}: an edge with stock and demand left idle"
            # 7.4: a supplier ships excess only if none of its requesters has core left
            ships_excess = np.bincount(ei, weights=xe, minlength=len(A)) > 0
            bad = ships_excess[ei] & core_open
            assert not bad.any(), f"round {tau}: excess served while reachable core was unmet"


def run_simulation(cfg: Config, districts: Districts, seed: int) -> tuple[list[dict], dict]:
    return Simulation(cfg, districts, seed).run()
