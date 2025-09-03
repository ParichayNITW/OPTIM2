import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline_model import _parallel_segment_hydraulics


def test_loopline_dra_bypass_reduces_head_loss():
    flow = 1000.0
    kv = 1.0
    main = {
        "L": 20.0,
        "d_inner": 0.5,
        "rough": 4e-5,
        "dra": 0.0,
        "dra_len": 0.0,
    }
    loop = {
        "L": 10.0,
        "d_inner": 0.5,
        "rough": 4e-5,
        "dra": 0.0,
        "dra_len": 0.0,
    }
    hl_no, _, _ = _parallel_segment_hydraulics(flow, main, loop, kv)

    carry_prev = 20.0
    opt_dra = 30.0
    loop_dra = loop.copy()
    loop_dra["dra"] = carry_prev + opt_dra
    loop_dra["dra_len"] = loop["L"]
    hl_dra, main_stats, loop_stats = _parallel_segment_hydraulics(flow, main, loop_dra, kv)

    assert hl_dra < hl_no

    q_main = main_stats[3]
    q_loop = loop_stats[3]
    assert q_main > 0 and q_loop > 0
    assert math.isclose(q_main + q_loop, flow, rel_tol=1e-6)
