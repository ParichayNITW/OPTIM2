import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from front_utils import get_ppm_for_dr, map_linefill_to_segments


def test_get_ppm_for_dr_handles_nan_and_missing_curves():
    assert get_ppm_for_dr(float('nan'), 10, {}) == 0.0
    df = pd.DataFrame({'%Drag Reduction': [0, 50], 'PPM': [0, 10]})
    curves = {10: df}
    # midpoint should interpolate to 5.0 and round to nearest 0.5
    assert get_ppm_for_dr(10, 25, curves) == 5.0


def test_map_linefill_to_segments_defaults_on_empty_df():
    stations = [{'L': 100.0}, {'L': 50.0}]
    kv, rho = map_linefill_to_segments(pd.DataFrame(), stations)
    assert kv == [1.0, 1.0]
    assert rho == [800.0, 800.0]
