"""All PULSE parameters in one place (README Section 3.3 and Section 11).

Every field has the README default. Override with `Config(...)`, `cfg.with_(...)`,
or `--set key=value` on the command line.
"""
from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, fields, replace
from typing import Optional

log = logging.getLogger("pulse")


@dataclass(frozen=True)
class Config:
    # ---------------- data ----------------
    data_path: str = "data/districts.xlsx"
    district_seed: int = 12345          # seeds the U[10,20] km diagonal fill
    r_intra_range: tuple = (10.0, 20.0)

    # ---------------- horizon ----------------
    days: tuple = (1, 3, 5, 8, 10, 12)  # round days; d_0 = -1, so gaps are (2,2,2,3,2,2)

    # ---------------- policy ----------------
    W: tuple = (1.0, 10.0, 100.0)       # urgency weights for levels 1/2/3
    m_min: int = 50                     # platform minimum request size
    H: float = 14.0                     # days of plausible need counted as "core"
    excess_weight: float = 1e-4         # value of an excess unit relative to the same unit as core
    alpha: float = 0.002                # distance tie-break weight (alpha-tilde)
    beta: float = 0.002                 # price tie-break weight
    mu: float = 0.02                    # "prefer certified" penalty per unit
    delta_max: float = math.inf         # optional distance cutoff (km)
    psi0_same: float = 0.40             # prior success probability, same state
    psi0_cross: float = 0.16            # prior success probability, cross state
    n0: float = 20.0                    # prior strength (pseudo-matches)
    psi_floor: float = 0.05             # lower clip on learned psi-hat

    # ---------------- scenario (truth, unknown to the model) ----------------
    psi_true_same: Optional[float] = None   # None = equal to the prior
    psi_true_cross: Optional[float] = None

    # ---------------- behavioural ----------------
    usage_rates: tuple = (0.25, 0.5, 1.0)   # plausible units per staff per day, levels 1/2/3
    eta: float = 0.5                    # share of unsold stock kept
    h: float = 5.0                      # storage cap in days of production
    omega: float = 0.10                 # abandonment probability per round

    # ---------------- switches ----------------
    simulate_failures: bool = True
    learn_psi: bool = True
    use_psi_objective: bool = True      # False = Shield-Net style (psi-hat fixed at 1)
    core_excess: bool = True            # False = the whole request is treated as core
    price_filter: bool = True
    cert_filter: bool = True

    # ---------------- solver ----------------
    solver: str = "cbc"                 # "cbc" (bundled with PuLP 3.x) or "highs" (needs highspy)
    check_invariants: bool = True       # assert the README Section 7 properties every round

    # ---------------- generator: suppliers ----------------
    n_suppliers: int = 450
    supply_scale: float = 1.0           # multiplies the number of suppliers
    frac_enrol_round1: float = 0.7
    class_probs: tuple = (0.5, 0.3, 0.2)                     # S / M / L
    s_ranges: tuple = ((25, 100), (60, 250), (200, 800))     # daily rate, integer uniform
    price_range: tuple = (20.0, 150.0)
    price_mult: tuple = (1.0, 0.9, 0.8)
    cert_probs: tuple = (0.3, 0.5, 0.8)
    sup_variant_probs: tuple = (0.57, 0.10, 0.33)            # assembled only / kit only / both

    # ---------------- generator: requesters ----------------
    arrivals_1: float = 120.0
    arrivals_decay: float = 0.7
    level_probs: tuple = (0.3, 0.4, 0.3)
    need_median: float = 400.0
    need_sigma: float = 1.4
    need_clip: tuple = (50, 100000)
    need_days: float = 14.0             # days of need the generator sizes facilities by (world, not policy)
    g_median: float = 2.0
    g_sigma: float = 0.5
    pmax_range: tuple = (30.0, 120.0)
    pmax_per_level: float = 20.0
    kappa_probs: tuple = ((0.5, 0.3, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5))
    req_variant_probs: tuple = (0.40, 0.33, 0.27)            # assembled / kit / either
    p_bulk: float = 0.005
    bulk_range: tuple = (0.5, 1.5)

    # ------------------------------------------------------------------
    @property
    def n_rounds(self) -> int:
        return len(self.days)

    @property
    def gaps(self) -> tuple:
        prev = -1
        out = []
        for d in self.days:
            out.append(d - prev)
            prev = d
        return tuple(out)

    @property
    def psi_true(self) -> tuple:
        same = self.psi0_same if self.psi_true_same is None else self.psi_true_same
        cross = self.psi0_cross if self.psi_true_cross is None else self.psi_true_cross
        return (same, cross)

    @property
    def tie_break_max(self) -> float:
        return self.alpha + self.beta + self.mu

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def validate(self) -> None:
        """Start-up checks. Raises on (C1) or impossible generator settings."""
        if len(self.W) != 3 or any(w <= 0 for w in self.W):
            raise ValueError("W must be three positive weights")
        psi_min = self.psi_floor if self.use_psi_objective else 1.0
        if self.use_psi_objective and not (self.learn_psi and self.simulate_failures):
            psi_min = min(self.psi0_same, self.psi0_cross)
        if not self.tie_break_max < self.W[0] * psi_min:
            raise ValueError(
                f"(C1) violated: alpha+beta+mu = {self.tie_break_max:.4f} must be < "
                f"W[1]*psi_min = {self.W[0] * psi_min:.4f}")
        # (C3): any core unit must beat any excess unit (psi_hat <= 1)
        core_min = self.W[0] * psi_min - self.tie_break_max
        if self.core_excess and not self.excess_weight * self.W[2] * 1.0 < core_min:
            raise ValueError(
                f"(C3) violated: excess_weight*W[3] = {self.excess_weight * self.W[2]:.2e} must be < "
                f"W[1]*psi_min - (alpha+beta+mu) = {core_min:.4f}")
        if len(self.usage_rates) != 3 or any(r <= 0 for r in self.usage_rates):
            raise ValueError("usage_rates must be three positive rates")
        if self.need_clip[0] < self.m_min:
            raise ValueError("need_clip lower bound must be >= m_min")
        if self.learn_psi and not self.simulate_failures:
            log.info("learn_psi has no effect without simulate_failures")


def parse_overrides(cfg: Config, pairs: list[str]) -> Config:
    """Apply `key=value` strings (from --set) to a Config, with type checking."""
    if not pairs:
        return cfg
    valid = {f.name for f in fields(Config)}
    kw = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise ValueError(f"unknown parameter {key!r}")
        raw = raw.strip()
        if raw.lower() in ("inf", "infinity"):
            val = math.inf
        elif raw.lower() in ("true", "false"):
            val = raw.lower() == "true"
        elif raw.lower() == "none":
            val = None
        else:
            try:
                val = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                val = raw  # plain string, e.g. a path
        if isinstance(val, list):
            val = tuple(val)
        kw[key] = val
    return replace(cfg, **kw)
