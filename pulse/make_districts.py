"""Placeholder district workbook (README Section 2.1).

Writes an Excel file in exactly the format the real data must follow:
  sheet `districts`: district_id, name, state, population
  sheet `distances`: |K| x |K| road distances in km, IDs as row and column headers

Usage:
    python -m pulse.make_districts data/districts.xlsx [--n 800] [--states 36] [--seed 7]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd


def make_placeholder(path: str, n: int = 800, n_states: int = 36, seed: int = 7,
                     side_km: float = 3000.0, sigma_km: float = 150.0,
                     road_factor: float = 1.3) -> str:
    if n < n_states:
        raise ValueError("need at least one district per state")
    rng = np.random.default_rng(seed)

    # District counts per state: Dirichlet(1) split, at least 1 each.
    shares = rng.dirichlet(np.ones(n_states))
    counts = 1 + rng.multinomial(n - n_states, shares)
    state_idx = np.repeat(np.arange(n_states), counts)

    centres = rng.uniform(0.0, side_km, size=(n_states, 2))
    xy = centres[state_idx] + rng.normal(0.0, sigma_km, size=(n, 2))

    diff = xy[:, None, :] - xy[None, :, :]
    straight = np.sqrt((diff ** 2).sum(axis=2))
    road = np.rint(road_factor * straight)
    off = ~np.eye(n, dtype=bool)
    road[off] = np.maximum(road[off], 1.0)     # distinct districts are never 0 km apart
    np.fill_diagonal(road, 0.0)                 # left 0 so the loader's U[10,20] rule is used

    width = len(str(n))
    ids = [f"D{k + 1:0{max(3, width)}d}" for k in range(n)]
    states = [f"S{s + 1:02d}" for s in state_idx]
    population = np.rint(rng.lognormal(np.log(2e6), 0.6, size=n)).astype(np.int64)

    districts = pd.DataFrame({
        "district_id": ids,
        "name": [f"District {k + 1}" for k in range(n)],
        "state": states,
        "population": population,
    })
    distances = pd.DataFrame(road.astype(np.int64), index=ids, columns=ids)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        districts.to_excel(xw, sheet_name="districts", index=False)
        distances.to_excel(xw, sheet_name="distances", index=True, index_label="district_id")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data/districts.xlsx")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--states", type=int, default=36)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t = time.perf_counter()
    make_placeholder(a.path, n=a.n, n_states=a.states, seed=a.seed)
    print(f"wrote {a.path} ({a.n} districts, {a.states} states) in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
