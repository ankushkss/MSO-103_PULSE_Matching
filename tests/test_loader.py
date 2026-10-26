import numpy as np
import pandas as pd
import pytest

from pulse import Config, Simulation, load_districts
from pulse.districts import DistrictDataError


def test_placeholder_loads(districts):
    assert districts.K == 60
    assert np.isfinite(districts.R).all()
    d = districts.R.diagonal()
    assert ((d >= 10) & (d <= 20)).all()            # intra-district U[10,20]
    assert districts.R_max == districts.R[np.isfinite(districts.R)].max()


def _rewrite(src, dst, mutate):
    dist = pd.read_excel(src, sheet_name="districts")
    mat = pd.read_excel(src, sheet_name="distances", index_col=0)
    dist, mat = mutate(dist, mat)
    with pd.ExcelWriter(dst, engine="openpyxl") as xw:
        dist.to_excel(xw, sheet_name="districts", index=False)
        mat.to_excel(xw, sheet_name="distances", index=True, index_label="district_id")


def test_order_does_not_matter(small_xlsx, districts, tmp_path):
    rng = np.random.default_rng(0)

    def shuffle(dist, mat):
        rows = rng.permutation(mat.index)
        cols = rng.permutation(mat.columns)
        return dist, mat.loc[rows, cols]

    dst = str(tmp_path / "shuffled.xlsx")
    _rewrite(small_xlsx, dst, shuffle)
    d2 = load_districts(dst, use_cache=False)
    np.testing.assert_array_equal(d2.R, districts.R)


def test_blank_pair_is_unreachable_and_never_an_edge(small_xlsx, tmp_path, small_cfg):
    def blank_all_but_one_column(dist, mat):
        # requesters can then only be reached within their own district or from district 0
        mat = mat.astype(object)
        for i, rid in enumerate(mat.index):
            for j, cid in enumerate(mat.columns):
                if i != j and j != 0:
                    mat.iat[i, j] = None
        return dist, mat

    dst = str(tmp_path / "blank.xlsx")
    _rewrite(small_xlsx, dst, blank_all_but_one_column)
    d = load_districts(dst, use_cache=False)
    assert d.n_blank > 0 and np.isinf(d.R).sum() == d.n_blank
    sim = Simulation(small_cfg, d, seed=1)
    sim.run()
    km = sim.matches.get("km")
    assert np.isfinite(km).all()                     # no match ever used an unreachable pair


def test_off_diagonal_zero_is_rejected(small_xlsx, tmp_path):
    def zero(dist, mat):
        mat.iat[0, 1] = 0
        return dist, mat

    dst = str(tmp_path / "zero.xlsx")
    _rewrite(small_xlsx, dst, zero)
    with pytest.raises(DistrictDataError):
        load_districts(dst, use_cache=False)


def test_mismatched_ids_rejected(small_xlsx, tmp_path):
    def drop_row(dist, mat):
        return dist, mat.iloc[1:, :]

    dst = str(tmp_path / "bad.xlsx")
    _rewrite(small_xlsx, dst, drop_row)
    with pytest.raises(DistrictDataError):
        load_districts(dst, use_cache=False)


def test_cache_roundtrip(small_xlsx, districts):
    a = load_districts(small_xlsx, use_cache=True)
    b = load_districts(small_xlsx, use_cache=True)        # second call reads the cache
    np.testing.assert_array_equal(a.R, b.R)
    np.testing.assert_array_equal(a.R, districts.R)
