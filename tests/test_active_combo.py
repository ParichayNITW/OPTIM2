import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline_model import solve_pipeline_with_types


def run_case(a_eff, b_eff):
    station = {
        'name': 'PS1',
        'is_pump': True,
        'L': 0.0,
        'd': 0.25,
        'MinRPM': 1000,
        'DOL': 1000,
        'min_residual': 0.0,
        'pump_types': {
            'A': {
                'available': 1,
                'A': 0.0,
                'B': 0.0,
                'C': 100.0,
                'P': 0.0,
                'Q': 0.0,
                'R': 0.0,
                'S': 0.0,
                'T': a_eff,
                'MinRPM': 1000,
                'DOL': 1000,
                'power_type': 'Grid',
                'rate': 1.0,
            },
            'B': {
                'available': 1,
                'A': 0.0,
                'B': 0.0,
                'C': 100.0,
                'P': 0.0,
                'Q': 0.0,
                'R': 0.0,
                'S': 0.0,
                'T': b_eff,
                'MinRPM': 1000,
                'DOL': 1000,
                'power_type': 'Grid',
                'rate': 1.0,
            },
        },
    }
    stations = [station]
    terminal = {'min_residual': 50.0, 'elev': 0.0}
    FLOW = 100.0
    KV_list = [1.0]
    rho_list = [850.0]
    result = solve_pipeline_with_types(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=850.0,
        Ambient_temp=25.0,
        linefill_dict={},
        dra_reach_km=0.0,
        mop_kgcm2=None,
        hours=24.0,
    )
    return result


def test_selects_A_when_more_efficient():
    res = run_case(70.0, 60.0)
    assert res['stations_used'][0]['active_combo'] == {'A': 1, 'B': 0}


def test_selects_B_when_more_efficient():
    res = run_case(60.0, 70.0)
    assert res['stations_used'][0]['active_combo'] == {'A': 0, 'B': 1}
