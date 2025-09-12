import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import pipeline_model


def make_stn(is_pump: bool, L: float):
    return {"is_pump": is_pump, "L": L}


def test_pump_injection_resets_ppm():
    prev_ppm = 10.0
    reach_prev = 150.0
    stn = make_stn(True, 50.0)
    opt = {"nop": 1, "dra_ppm_main": 20.0}

    ppm, dra_len, reach_after, inj = pipeline_model._update_mainline_dra(
        prev_ppm, reach_prev, stn, opt
    )

    assert ppm == opt["dra_ppm_main"]
    assert math.isclose(dra_len, 50.0)
    assert math.isclose(reach_after, pipeline_model.MAX_DRA_KM - stn["L"])
    assert inj == opt["dra_ppm_main"]


def test_unpumped_segment_carries_ppm():
    prev_ppm = 15.0
    reach_prev = 120.0
    stn = make_stn(False, 40.0)
    opt = {"nop": 0, "dra_ppm_main": 0.0}

    ppm, dra_len, reach_after, inj = pipeline_model._update_mainline_dra(
        prev_ppm, reach_prev, stn, opt
    )

    assert ppm == prev_ppm
    assert math.isclose(dra_len, 40.0)
    assert math.isclose(reach_after, 80.0)
    assert inj == 0.0

def test_pump_without_injection_clears_ppm():
    prev_ppm = 12.0
    reach_prev = 80.0
    stn = make_stn(True, 60.0)
    opt = {"nop": 1, "dra_ppm_main": 0.0}

    ppm, dra_len, reach_after, inj = pipeline_model._update_mainline_dra(
        prev_ppm, reach_prev, stn, opt
    )

    assert ppm == 0.0
    assert dra_len == 0.0
    assert reach_after == 0.0
    assert inj == 0.0
