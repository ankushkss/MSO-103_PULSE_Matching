"""The six-round PULSE simulation (README Sections 4, 5, 8 and 10)."""
from __future__ import annotations

import logging
import math

import numpy as np

from .config import Config
from .districts import Districts
from .generator import Instance, generate_instance, make_streams
from .metrics import MatchLog, round_record, run_summary
from .model import RoundProblem, solve_round

log = logging.getLogger("pulse")

PENDING, ACTIVE, SERVED, ABANDONED = 0, 1, 2, 3
SAME, CROSS = 0, 1
STRAND_CAUSES = ("variant", "price", "certificate", "distance", "no_stock", "min_order")


class Simulation:
    def __init__(self, cfg: Config, districts: Districts, seed: int,
                 instance: Instance | None = None):
        cfg.validate()
        self.cfg = cfg
        self.d = districts
        self.seed = seed
        streams = make_streams(seed)
        self.inst = instance if instance is not None else generate_instance(cfg, districts, seed, streams)
        self.rng_out = streams["outcomes"]
        self.rng_ab = streams["abandonment"]
        self.rng_ver = streams["verification"]

        inst = self.inst
        self.r = np.zeros(inst.n_suppliers, dtype=np.int64)          # carried stock
        self.carry = np.zeros(inst.n_requesters, dtype=np.int64)     # outstanding demand
        self.status = np.full(inst.n_requesters, PENDING, dtype=np.int8)
        self.z = np.zeros(inst.n_requesters, dtype=bool)             # plausibility flag
        self.ever_flagged = np.zeros(inst.n_requesters, dtype=bool)
        self.delivered = np.zeros(inst.n_requesters, dtype=np.int64)
        self.abandoned_units = np.zeros(inst.n_requesters, dtype=np.int64)
        self.S = np.zeros(2, dtype=np.int64)                          # successes by class
        self.F = np.zeros(2, dtype=np.int64)                          # failures by class
        self.W = np.asarray(cfg.W, dtype=float)
        self.matches = MatchLog()
        self.rounds: list[dict] = []

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
        mode_b = cfg.enforce_min_shipment

        # 1-2. enrolment and supply
        sup = np.nonzero(inst.s_enrol <= tau)[0]
        A = inst.s_rate[sup] * gap + self.r[sup]
        A_tot = int(A.sum())

        # arrivals and plausibility flags
        new = np.nonzero(inst.r_arrival == tau)[0]
        self.status[new] = ACTIVE
        self.carry[new] = inst.r_D0[new]
        if cfg.hoarding_control:
            self.z[new] = inst.r_D0[new] > inst.r_P[new]
            self.ever_flagged[new] |= self.z[new]

        # 3. outstanding demand
        req = np.nonzero(self.status == ACTIVE)[0]
        D = self.carry[req].copy()

        # 5. share cap
        cap = max(int(math.floor(cfg.c * A_tot)), cfg.m_min)
        flagged = self.z[req]
        Dt = np.where(flagged, np.minimum(D, cap), D)
        capped = Dt < D

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
        if mode_b:
            moq_ok = np.minimum(A[:, None], Dt[None, :]) >= inst.s_moq[sup][:, None]
            elig_final = elig & moq_ok
        else:
            elig_final = elig

        # stranded requesters, by first failing filter
        stranded = {c: 0 for c in STRAND_CAUSES}
        if len(req):
            chain = [compat, price_ok, cert_ok, dist_ok, stock_ok]
            cum = np.ones(len(req), dtype=bool) & (len(sup) > 0)
            m = ones
            has_edge = elig_final.any(axis=0) if len(sup) else np.zeros(len(req), bool)
            for cause, f in zip(STRAND_CAUSES[:5], chain):
                m = m & f
                ok_now = m.any(axis=0) if len(sup) else np.zeros(len(req), bool)
                stranded[cause] += int((cum & ~ok_now & ~has_edge).sum())
                cum = cum & ok_now
            if mode_b:
                stranded["min_order"] += int((cum & ~has_edge).sum())
        n_stranded = sum(stranded.values())

        ei, ej = np.nonzero(elig_final)

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
        wvec = self.W[level - 1] * psi[cls] - cvec
        if cfg.check_invariants and len(wvec):
            assert (wvec > 0).all(), "(C1) violated: an edge has non-positive value"
        xbar = np.minimum(A[ei], Dt[ej])
        moq = inst.s_moq[sup][ei]

        P = RoundProblem(e_sup=ei, e_req=ej, w=wvec, xbar=xbar, moq=moq,
                         A=A, Dt=Dt, D=D, m_min=cfg.m_min)
        sol = solve_round(P, mode_b, solver=cfg.solver, time_limit=cfg.time_limit,
                          compute_bound=cfg.compute_bound, gap_rel=cfg.gap_rel,
                          candidate_k=cfg.candidate_k)
        x = sol.x

        # 8.1 realise matches (outcomes from the TRUE psi, never psi_hat)
        matched = x > 0
        B = np.ones(len(x), dtype=bool)
        if cfg.simulate_failures and matched.any():
            psi_true = np.asarray(cfg.psi_true)
            draws = self.rng_out.random(int(matched.sum()))
            B[matched] = draws < psi_true[cls[matched]]
        deliv = np.where(B, x, 0)

        # 8.2 learn (counts matches, not units)
        if cfg.learn_psi and cfg.simulate_failures:
            for g in (SAME, CROSS):
                sel = matched & (cls == g)
                self.S[g] += int((sel & B).sum())
                self.F[g] += int((sel & ~B).sum())

        nS, nR = len(sup), len(req)
        sup_alloc = np.bincount(ei, weights=x, minlength=nS).astype(np.int64)
        req_alloc = np.bincount(ej, weights=x, minlength=nR).astype(np.int64)
        sup_deliv = np.bincount(ei, weights=deliv, minlength=nS).astype(np.int64)
        req_deliv = np.bincount(ej, weights=deliv, minlength=nR).astype(np.int64)

        if cfg.check_invariants:
            self._check(tau, mode_b, x, moq, matched, A, Dt, D, ei, ej, sup_alloc, req_alloc,
                        flagged, cap)

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
        if cfg.check_invariants and mode_b:
            still = self.carry[req[~served]]
            assert (still >= cfg.m_min).all(), "Mode B carry invariant (README 7.4) broken"

        # 8.5 abandonment
        act = req[~served]
        leave = self.rng_ab.random(len(act)) < cfg.omega
        gone = act[leave]
        self.abandoned_units[gone] = self.carry[gone]
        self.status[gone] = ABANDONED

        # 8.6 verification
        remain = act[~leave]
        fl = remain[self.z[remain]]
        ver = self.rng_ver.random(len(fl)) < cfg.nu
        self.z[fl[ver]] = False

        # logging
        self.matches.add(tau=tau, sup=sup[ei[matched]], req=req[ej[matched]],
                         x=x[matched], delivered=deliv[matched], km=Rk[matched],
                         same_state=same[matched],
                         same_district=(sd[ei] == rd[ej])[matched],
                         price=inst.s_price[sup][ei][matched], cert=q[matched],
                         kappa=kap[matched], level=level[matched],
                         psi_true=np.asarray(cfg.psi_true)[cls[matched]])
        return round_record(
            sim=self, tau=tau, A_tot=A_tot, D=D, Dt=Dt, req=req, n_sup=nS, cap=cap,
            flagged=flagged, capped=capped, x=x, deliv=deliv, matched=matched, B=B,
            psi=psi, c2_ok=c2_ok, sol=sol, n_edges=len(x), stranded=stranded,
            n_stranded=n_stranded, leaked=leaked, lost_storage=lost_storage,
            n_new=len(new), n_served=int(served.sum()), n_abandoned=int(leave.sum()),
            units_abandoned=int(self.abandoned_units[gone].sum()), n_verified=int(ver.sum()),
            same=same, Rk=Rk, cls=cls)

    # ------------------------------------------------------------------
    def _check(self, tau, mode_b, x, moq, matched, A, Dt, D, ei, ej, sup_alloc, req_alloc,
               flagged, cap):
        assert (sup_alloc <= A).all(), f"round {tau}: supplier allocated beyond stock"
        assert (req_alloc <= Dt).all(), f"round {tau}: requester allocated beyond D~"
        assert (req_alloc[flagged] <= cap).all(), f"round {tau}: share cap broken"
        if mode_b:
            assert (x[matched] >= moq[matched]).all(), f"round {tau}: minimum order broken"
            rem = D - req_alloc
            assert ((rem == 0) | (rem >= self.cfg.m_min)).all(), f"round {tau}: tiny remainder"
        else:
            res_s = A - sup_alloc
            res_d = Dt - req_alloc
            if len(ei):
                bad = (res_s[ei] > 0) & (res_d[ej] > 0)
                assert not bad.any(), f"round {tau}: Mode A left a usable edge idle (README 7.1)"


def run_simulation(cfg: Config, districts: Districts, seed: int) -> tuple[list[dict], dict]:
    return Simulation(cfg, districts, seed).run()
