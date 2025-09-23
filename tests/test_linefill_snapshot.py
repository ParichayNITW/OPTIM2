import pandas as pd
import pandas.testing as pdt

import pipeline_optimization_app as app


def test_get_linefill_snapshot_for_hour_returns_0600_snapshot():
    hours = [(7 + idx) % 24 for idx in range(24)]
    snapshots = [
        pd.DataFrame({"Station": ["S"], "DRA ppm": [float(hour)]}) for hour in hours
    ]

    result = app._get_linefill_snapshot_for_hour(snapshots, hours, target_hour=6)

    expected_idx = hours.index(6)
    pdt.assert_frame_equal(result, snapshots[expected_idx])
    assert result is not snapshots[expected_idx]


def test_get_linefill_snapshot_for_hour_returns_empty_when_missing_target():
    hours = [7, 8, 9]
    snapshots = [
        pd.DataFrame({"Station": ["S"], "DRA ppm": [float(hour)]}) for hour in hours
    ]

    result = app._get_linefill_snapshot_for_hour(snapshots, hours, target_hour=6)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_linefill_snapshot_for_hour_handles_absent_snapshots():
    result = app._get_linefill_snapshot_for_hour([], [7, 8, 9], target_hour=6)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
