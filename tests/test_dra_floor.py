import math
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline_model import _update_mainline_dra


def test_floor_respected_without_injection_when_queue_already_meets_floor():
    queue = [(10.0, 2.0)]
    stn_data = {
        "d_inner": 0.7,
        "idx": 0,
        "kv": 3.0,
    }
    opt = {
        "dra_ppm_main": 0.0,
        "nop": 0,
    }
    segment_floor = {
        "length_km": 10.0,
        "dra_ppm": 1.0,
        "enforce_queue": True,
    }

    pumped, queue_after, inj_ppm_main, floor_requires_injection = _update_mainline_dra(
        queue,
        stn_data,
        opt,
        segment_length=10.0,
        flow_m3h=1000.0,
        hours=1.0,
        pump_running=False,
        pump_shear_rate=0.0,
        dra_shear_factor=0.0,
        shear_injection=False,
        is_origin=True,
        precomputed=None,
        segment_floor=segment_floor,
    )

    assert not floor_requires_injection, "Existing queue already meets the floor without injection"
    assert math.isclose(sum(float(length) for length, _ppm in pumped), 10.0, rel_tol=0, abs_tol=1e-9)
