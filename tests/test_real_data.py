"""The shipped district workbook (data/districts.xlsx) loads and is usable."""
import os

import numpy as np
import pytest

from pulse.districts import load_districts

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "districts.xlsx")


@pytest.mark.skipif(not os.path.exists(PATH), reason="data/districts.xlsx not present")
def test_real_workbook_loads():
    d = load_districts(PATH, use_cache=False)
    K = len(d.ids)
    assert K >= 780
    assert len(d.state_names) >= 35
    diag = np.diag(d.R)
    assert ((diag >= 10) & (diag <= 20)).all()            # blank diagonal -> U[10, 20]
    fin = np.isfinite(d.R)
    assert np.array_equal(fin, fin.T)
    assert np.allclose(d.R[fin], d.R.T[fin])               # symmetric
    off = fin & ~np.eye(K, dtype=bool)
    assert (d.R[off] > 0).all()
    # island districts are cut off from the mainland
    islands = np.isin(d.state_names[d.state], ["Andaman and Nicobar", "Lakshadweep"])
    assert islands.any()
    assert not np.isfinite(d.R[np.ix_(islands, ~islands)]).any()
