import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pipeline_model


def test_sdh_history_clamps_when_queue_has_dra():
    pipeline_model._SDH_HISTORY.clear()

    first_sdh = pipeline_model._apply_sdh_history("S1", 100.0, True, True)
    assert first_sdh == 100.0

    # With DRA present, subsequent SDH values should not exceed the stored value.
    second_sdh = pipeline_model._apply_sdh_history("S1", 120.0, True, True)
    assert second_sdh == 100.0
    assert pipeline_model._SDH_HISTORY["S1"] == 100.0


def test_sdh_history_updates_without_queue_or_non_pump():
    pipeline_model._SDH_HISTORY.clear()
    uncapped_sdh = pipeline_model._apply_sdh_history("S1", 90.0, False, True)
    assert uncapped_sdh == 90.0
    assert pipeline_model._SDH_HISTORY["S1"] == 90.0

    non_pump_sdh = pipeline_model._apply_sdh_history("S2", 75.0, True, False)
    assert non_pump_sdh == 75.0
    assert "S2" not in pipeline_model._SDH_HISTORY
