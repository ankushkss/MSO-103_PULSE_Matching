"""District data loader (README Section 2.1).

Reads the workbook, validates it, reorders the matrix to the `districts` sheet,
fills the diagonal with U[10,20] km where it is 0 or blank, and marks blank
off-diagonal cells as unreachable (inf). The parsed raw matrix is cached as
`<workbook>.cache.npz` and reused while the workbook is unchanged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("pulse")

REQUIRED_COLS = ("district_id", "name", "state")


class DistrictDataError(ValueError):
    pass


@dataclass
class Districts:
    ids: np.ndarray            # str, length K
    names: np.ndarray          # str
    state: np.ndarray          # int state code per district
    state_names: np.ndarray    # str, indexed by state code
    population: Optional[np.ndarray]
    R: np.ndarray              # float km, inf = unreachable, R[k,l] supplier k -> requester l
    R_max: float               # largest finite entry
    n_blank: int               # blank off-diagonal cells
    n_asym: int                # pairs with R[k,l] != R[l,k]

    @property
    def K(self) -> int:
        return len(self.ids)


def _clean_id(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _read_raw(path: str):
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DistrictDataError(f"cannot open {path}: {e}") from e
    for sheet in ("districts", "distances"):
        if sheet not in xls.sheet_names:
            raise DistrictDataError(f"{path}: missing sheet {sheet!r} (found {xls.sheet_names})")

    dist = pd.read_excel(xls, sheet_name="districts", dtype=object)
    dist.columns = [str(c).strip() for c in dist.columns]
    missing = [c for c in REQUIRED_COLS if c not in dist.columns]
    if missing:
        raise DistrictDataError(f"`districts` sheet is missing columns {missing}")
    dist = dist.dropna(how="all")
    if dist[list(REQUIRED_COLS)].isna().any().any():
        raise DistrictDataError("`districts` sheet has blank district_id, name or state")
    ids = np.array([_clean_id(v) for v in dist["district_id"]], dtype=object)
    if len(set(ids)) != len(ids):
        dup = pd.Series(ids)[pd.Series(ids).duplicated()].tolist()[:5]
        raise DistrictDataError(f"duplicate district_id values, e.g. {dup}")
    names = np.array([str(v).strip() for v in dist["name"]], dtype=object)
    states_raw = np.array([str(v).strip() for v in dist["state"]], dtype=object)
    population = None
    if "population" in dist.columns and dist["population"].notna().all():
        population = pd.to_numeric(dist["population"], errors="coerce").to_numpy(float)
        if np.isnan(population).any() or (population < 0).any():
            raise DistrictDataError("`population` must be non-negative numbers")
    elif "population" in dist.columns:
        log.warning("`population` column has blanks; requesters will be placed uniformly")

    mat = pd.read_excel(xls, sheet_name="distances", index_col=0, dtype=object)
    mat.index = [_clean_id(v) for v in mat.index]
    mat.columns = [_clean_id(v) for v in mat.columns]
    mat = mat.loc[[i for i in mat.index if i != "nan"]]          # trailing blank rows
    mat = mat[[c for c in mat.columns if not c.startswith("Unnamed")]]
    for axis_name, labels in (("row", list(mat.index)), ("column", list(mat.columns))):
        if len(set(labels)) != len(labels):
            raise DistrictDataError(f"duplicate {axis_name} IDs in `distances`")
        extra = set(labels) - set(ids)
        lacking = set(ids) - set(labels)
        if extra or lacking:
            raise DistrictDataError(
                f"`distances` {axis_name} IDs do not match the `districts` sheet: "
                f"{len(extra)} unknown (e.g. {sorted(extra)[:3]}), "
                f"{len(lacking)} missing (e.g. {sorted(lacking)[:3]})")
    mat = mat.reindex(index=list(ids), columns=list(ids))

    blank = mat.isna() | mat.map(lambda v: isinstance(v, str) and v.strip() == "")
    num = mat.mask(blank).apply(pd.to_numeric, errors="coerce")
    nonnum = num.isna() & ~blank
    if nonnum.to_numpy().any():
        r, c = np.argwhere(nonnum.to_numpy())[0]
        raise DistrictDataError(f"non-numeric distance at ({ids[r]}, {ids[c]}): {mat.iat[r, c]!r}")
    raw = num.to_numpy(dtype=float)              # NaN = blank
    if (raw[~np.isnan(raw)] < 0).any():
        raise DistrictDataError("negative distances in `distances`")
    off = ~np.eye(len(ids), dtype=bool)
    zero_off = off & (raw == 0)
    if zero_off.any():
        r, c = np.argwhere(zero_off)[0]
        raise DistrictDataError(
            f"{int(zero_off.sum())} off-diagonal zero distances, e.g. ({ids[r]}, {ids[c]}); "
            "two different districts cannot be 0 km apart (leave the cell blank if unknown)")
    return ids, names, states_raw, population, raw


def _cache_path(path: str) -> str:
    return path + ".cache.npz"


def _load_raw_cached(path: str, use_cache: bool):
    st = os.stat(path)
    key = np.array([st.st_mtime_ns, st.st_size], dtype=np.int64)
    cpath = _cache_path(path)
    if use_cache and os.path.exists(cpath):
        try:
            z = np.load(cpath, allow_pickle=True)
            if np.array_equal(z["key"], key):
                pop = z["population"] if z["has_pop"] else None
                return z["ids"], z["names"], z["states"], pop, z["raw"]
        except Exception:  # noqa: BLE001  (corrupt cache: just rebuild)
            pass
    log.info("reading %s (first load can take a while for an 800 x 800 sheet)", path)
    ids, names, states, pop, raw = _read_raw(path)
    if use_cache:
        try:
            np.savez_compressed(cpath, key=key, ids=ids, names=names, states=states,
                                population=pop if pop is not None else np.zeros(0),
                                has_pop=pop is not None, raw=raw)
        except OSError as e:
            log.warning("could not write cache %s: %s", cpath, e)
    return ids, names, states, pop, raw


def load_districts(path: str, district_seed: int = 12345,
                   r_intra_range: tuple = (10.0, 20.0), use_cache: bool = True) -> Districts:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ids, names, states_raw, population, raw = _load_raw_cached(path, use_cache)
    K = len(ids)
    R = raw.astype(float).copy()

    diag = np.diag(R).copy()
    fill = np.isnan(diag) | (diag == 0)
    rng = np.random.default_rng(district_seed)
    draws = rng.uniform(r_intra_range[0], r_intra_range[1], size=K)   # one per district, always drawn
    diag[fill] = draws[fill]
    np.fill_diagonal(R, diag)

    off = ~np.eye(K, dtype=bool)
    blank = off & np.isnan(R)
    n_blank = int(blank.sum())
    R[blank] = np.inf

    fin = np.isfinite(R) & np.isfinite(R.T)
    asym = fin & (R != R.T)
    n_asym = int(np.triu(asym, 1).sum())

    finite = R[np.isfinite(R)]
    if finite.size == 0:
        raise DistrictDataError("no finite distances")
    R_max = float(finite.max())

    state_names, state = np.unique(np.asarray(states_raw, dtype=str), return_inverse=True)
    if n_blank:
        log.warning("%d blank off-diagonal distances treated as unreachable", n_blank)
    if n_asym:
        log.info("%d district pairs have asymmetric distances", n_asym)
    return Districts(ids=np.asarray(ids, dtype=str), names=np.asarray(names, dtype=str),
                     state=state.astype(np.int64), state_names=state_names,
                     population=population, R=R, R_max=R_max,
                     n_blank=n_blank, n_asym=n_asym)
