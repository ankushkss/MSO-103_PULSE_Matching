import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pulse import Config, load_districts  # noqa: E402
from pulse.make_districts import make_placeholder  # noqa: E402


@pytest.fixture(scope="session")
def small_xlsx(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "small.xlsx"
    make_placeholder(str(path), n=60, n_states=8, seed=3)
    return str(path)


@pytest.fixture(scope="session")
def districts(small_xlsx):
    return load_districts(small_xlsx, use_cache=False)


@pytest.fixture
def small_cfg():
    # A small instance so every test runs in seconds.
    return Config(n_suppliers=60, arrivals_1=25, time_limit=10, candidate_k=0)
