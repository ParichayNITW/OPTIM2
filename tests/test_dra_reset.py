import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline_model import _update_mainline_dra, MAX_DRA_KM


def test_pump_injection_resets_ppm():
    stn = {"L": 50.0, "is_pump": True}
    opt = {"dra_ppm_main": 5.0, "nop": 1}
    ppm, dra_len, reach, inj_ppm = _update_mainline_dra(10.0, 100.0, stn, opt)
    assert ppm == 5.0
    assert dra_len == 50.0
    assert reach == MAX_DRA_KM - 50.0
    assert inj_ppm == 5.0


def test_pump_no_injection_resets_to_zero():
    stn = {"L": 50.0, "is_pump": True}
    opt = {"dra_ppm_main": 0.0, "nop": 1}
    ppm, dra_len, reach, inj_ppm = _update_mainline_dra(10.0, 100.0, stn, opt)
    assert ppm == 0.0
    assert dra_len == 0.0
    assert reach == 0.0
    assert inj_ppm == 0.0


def test_unpumped_segment_carries_ppm_until_reach():
    stn = {"L": 50.0, "is_pump": False}
    opt = {"dra_ppm_main": 0.0, "nop": 0}
    ppm, dra_len, reach, inj_ppm = _update_mainline_dra(10.0, 150.0, stn, opt)
    assert ppm == 10.0
    assert dra_len == 50.0
    assert reach == 100.0
    assert inj_ppm == 0.0
