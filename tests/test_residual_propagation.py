import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pipeline_model


def test_residual_matches_sdh_and_losses():
    pipeline_model._SDH_HISTORY.clear()
    # Use a clear set of inputs so the downstream residual must equal
    # SDH minus head loss minus the elevation difference with no rounding.
    sdh_effective = pipeline_model._apply_sdh_history("P1", 468.56, False, True)
    assert sdh_effective == 468.56

    residual_next = sdh_effective - 404.0 - 3.0
    bucket = int(round(residual_next))

    assert residual_next == 61.56
    assert bucket == 62
