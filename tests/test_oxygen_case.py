import importlib.util
import pathlib

import numpy as np

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "oxygen_case.py"
_spec = importlib.util.spec_from_file_location("oxygen_case", _p)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


def test_capped_proportional_redistributes_above_caps():
    x = oc.capped_proportional(100, np.array([1.0, 1.0, 2.0]), np.array([10.0, 100.0, 100.0]))
    assert np.isclose(x.sum(), 100) and np.isclose(x[0], 10)
    assert np.isclose(x[2], 2 * x[1])                 # uncapped ones keep their ratio


def test_capped_proportional_never_exceeds_total_caps():
    x = oc.capped_proportional(500, np.ones(3), np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(x, [10, 20, 30])


def test_pulse_serves_core_before_excess():
    D, need = np.array([100.0, 100.0]), np.array([20.0, 100.0])   # state 0 over-asks
    x = oc.pulse_core_excess(110, D, need)
    np.testing.assert_allclose(x, [110 * 20 / 120, 110 * 100 / 120])  # core 120 > 110: pro-rata core
    x = oc.pulse_core_excess(150, D, need)
    np.testing.assert_allclose(x, [50, 100])                       # core met, rest to excess


def test_real_data_loads_and_matches_printed_total():
    d = oc.load(str(pathlib.Path(__file__).resolve().parents[1] / "data" / "oxygen"))
    assert len(d) == 19 and d["actual"].sum() == 6722 and d["demand"].sum() == 6721
