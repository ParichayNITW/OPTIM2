import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline_model import solve_pipeline


def test_disabled_loops_allows_high_flow():
    stations = [
        {
            'name': 'S1',
            'is_pump': True,
            'max_pumps': 1,
            'MinRPM': 2000,
            'DOL': 2000,
            'A': 0.0,
            'B': 0.0,
            'C': 80.0,
            'P': 0.0,
            'Q': 0.0,
            'R': 0.0,
            'S': 0.0,
            'T': 0.0,
            'pump_types': {
                'A': {
                    'A': 0.0,
                    'B': 0.0,
                    'C': 80.0,
                    'DOL': 2000,
                    'P': 0.0,
                    'Q': 0.0,
                    'R': 0.0,
                    'S': 0.0,
                    'T': 0.0,
                }
            },
            'pump_combo': {'A': 1},
            'active_combo': {'A': 1},
            'L': 1.0,
            't': 0.01,
            'd': 1.0,
            'rough': 0.00004,
            'max_dr': 0.0,
            'elev': 0.0,
            'loopline': {
                'L': 1.0,
                't': 0.01,
                'd': 1.0,
                'rough': 0.00004,
                'max_dr': 0.0,
                'peaks': [{'loc': 0.5, 'elev': 200}],
            },
        },
        {
            'name': 'S2',
            'is_pump': False,
            'L': 1.0,
            't': 0.01,
            'd': 1.0,
            'rough': 0.00004,
            'max_dr': 0.0,
            'elev': 0.0,
            'loopline': {
                'L': 1.0,
                't': 0.01,
                'd': 1.0,
                'rough': 0.00004,
                'max_dr': 0.0,
                'peaks': [{'loc': 0.5, 'elev': 150}],
            },
        },
    ]
    terminal = {'name': 'T', 'elev': 0.0, 'min_residual': 0.0}
    FLOW = 1900.0
    KV_list = [0.0, 0.0]
    rho_list = [1000.0, 1000.0]
    result = solve_pipeline(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA=0.0,
        Price_HSD=1.0,
        Fuel_density=0.84,
        Ambient_temp=25.0,
        loop_usage_by_station=[0, 0],
        enumerate_loops=False,
    )
    assert not result.get('error'), result.get('message')
