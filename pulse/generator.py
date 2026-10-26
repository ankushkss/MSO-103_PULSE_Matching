"""Synthetic suppliers and requesters (README Section 11).

The instance depends only on (config generator fields, districts, seed), never on
policy parameters, so every policy is compared on the same instance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .districts import Districts

# variant bitmasks
ASSEMBLED, KIT = 1, 2
BOTH = ASSEMBLED | KIT


def make_streams(seed: int) -> dict:
    """Independent RNG streams so that a policy change does not shift other draws."""
    names = ("suppliers", "requesters", "outcomes", "abandonment", "verification")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(c) for n, c in zip(names, children)}


@dataclass
class Instance:
    # suppliers (length N_S)
    s_dist: np.ndarray
    s_enrol: np.ndarray
    s_class: np.ndarray
    s_rate: np.ndarray        # units per day
    s_moq: np.ndarray
    s_price: np.ndarray
    s_cert: np.ndarray        # bool
    s_var: np.ndarray         # variant bitmask
    # requesters (length N_R)
    r_dist: np.ndarray
    r_arrival: np.ndarray
    r_level: np.ndarray       # 1/2/3
    r_D0: np.ndarray
    r_size: np.ndarray
    r_P: np.ndarray           # plausible need
    r_pmax: np.ndarray
    r_kappa: np.ndarray       # 0/1/2
    r_var: np.ndarray
    r_bulk: np.ndarray        # bool
    A_bar1: float             # expected round-1 supply

    @property
    def n_suppliers(self) -> int:
        return len(self.s_rate)

    @property
    def n_requesters(self) -> int:
        return len(self.r_D0)


def expected_daily_rate(cfg: Config) -> float:
    return float(sum(p * (lo + hi) / 2 for p, (lo, hi) in zip(cfg.class_probs, cfg.s_ranges)))


def generate_instance(cfg: Config, districts: Districts, seed: int,
                      streams: dict | None = None) -> Instance:
    streams = streams or make_streams(seed)
    K = districts.K

    # ---------------- suppliers ----------------
    rs = streams["suppliers"]
    N = max(1, int(round(cfg.n_suppliers * cfg.supply_scale)))
    n_rounds = cfg.n_rounds
    enrol = np.where(rs.random(N) < cfg.frac_enrol_round1, 1, rs.integers(2, n_rounds + 1, size=N))
    cls = rs.choice(3, size=N, p=np.asarray(cfg.class_probs) / sum(cfg.class_probs))
    lo = np.array([r[0] for r in cfg.s_ranges])[cls]
    hi = np.array([r[1] for r in cfg.s_ranges])[cls]
    rate = rs.integers(lo, hi + 1)
    moq = np.asarray(cfg.m_by_class, dtype=np.int64)[cls]
    price = rs.uniform(*cfg.price_range, size=N) * np.asarray(cfg.price_mult)[cls]
    cert = rs.random(N) < np.asarray(cfg.cert_probs)[cls]
    svar = rs.choice([ASSEMBLED, KIT, BOTH], size=N,
                     p=np.asarray(cfg.sup_variant_probs) / sum(cfg.sup_variant_probs))
    sdist = rs.integers(0, K, size=N)

    A_bar1 = cfg.frac_enrol_round1 * N * cfg.gaps[0] * expected_daily_rate(cfg)

    # ---------------- requesters ----------------
    rr = streams["requesters"]
    counts = [rr.poisson(cfg.arrivals_1 * cfg.arrivals_decay ** t) for t in range(n_rounds)]
    M = int(sum(counts))
    arrival = np.repeat(np.arange(1, n_rounds + 1), counts)
    if districts.population is not None and districts.population.sum() > 0:
        w = districts.population / districts.population.sum()
        rdist = rr.choice(K, size=M, p=w)
    else:
        rdist = rr.integers(0, K, size=M)
    level = rr.choice([1, 2, 3], size=M, p=np.asarray(cfg.level_probs) / sum(cfg.level_probs))
    need = np.rint(rr.lognormal(np.log(cfg.need_median), cfg.need_sigma, size=M))
    need = np.clip(need, max(cfg.need_clip[0], cfg.m_min), cfg.need_clip[1]).astype(np.int64)
    g = rr.lognormal(np.log(cfg.g_median), cfg.g_sigma, size=M)
    size = np.maximum(1, np.ceil(need * g / (cfg.u * cfg.T_cov))).astype(np.int64)
    P = cfg.u * size * cfg.T_cov
    bulk = rr.random(M) < cfg.p_bulk
    bulk_size = np.rint(rr.uniform(*cfg.bulk_range, size=M) * A_bar1).astype(np.int64)
    D0 = np.where(bulk, np.maximum(bulk_size, cfg.m_min), need).astype(np.int64)
    pmax = rr.uniform(*cfg.pmax_range, size=M) + cfg.pmax_per_level * (level - 1)
    kappa = np.empty(M, dtype=np.int64)
    for lv in (1, 2, 3):
        idx = np.nonzero(level == lv)[0]
        pk = np.asarray(cfg.kappa_probs[lv - 1]) / sum(cfg.kappa_probs[lv - 1])
        kappa[idx] = rr.choice(3, size=len(idx), p=pk)
    rvar = rr.choice([ASSEMBLED, KIT, BOTH], size=M,
                     p=np.asarray(cfg.req_variant_probs) / sum(cfg.req_variant_probs))

    return Instance(
        s_dist=sdist, s_enrol=enrol.astype(np.int64), s_class=cls, s_rate=rate.astype(np.int64),
        s_moq=moq, s_price=price, s_cert=cert, s_var=svar.astype(np.int64),
        r_dist=rdist, r_arrival=arrival.astype(np.int64), r_level=level.astype(np.int64),
        r_D0=D0, r_size=size, r_P=P, r_pmax=pmax, r_kappa=kappa,
        r_var=rvar.astype(np.int64), r_bulk=bulk, A_bar1=float(A_bar1),
    )
