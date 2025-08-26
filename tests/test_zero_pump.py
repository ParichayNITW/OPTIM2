import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pipeline_model


def test_station_with_min_pumps_has_no_zero_option():
    stations = [
        {
            'name': 'P1',
            'is_pump': True,
            'min_pumps': 1,
            'max_pumps': 1,
            'MinRPM': 1000,
            'DOL': 1000,
            'L': 0.01,
            'd': 0.2,
            'rough': 0.00004,
            'A': 0.0,
            'B': 0.0,
            'C': 10.0,
            'P': 0.0,
            'Q': 0.0,
            'R': 0.0,
            'S': 0.0,
            'T': 75.0,
            'power_type': 'Grid',
            'rate': 1.0,
        }
    ]
    terminal = {'name': 'T', 'elev': 0.0}
    FLOW = 100.0
    KV_list = [1000.0]
    rho_list = [850.0, 850.0]

    result = pipeline_model.solve_pipeline(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=1.0,
        Ambient_temp=25.0,
    )

    key = 'num_pumps_p1'
    assert result[key] == 1, 'Station with min_pumps >= 1 should not offer a zero-pump option'
    assert result['power_cost_p1'] > 0.0


def test_only_origin_requires_minimum_one_pump():
    stations = [
        {
            'name': 'P1',
            'is_pump': True,
            'min_pumps': 0,
            'max_pumps': 1,
            'MinRPM': 1000,
            'DOL': 1000,
            'L': 0.01,
            'd': 0.2,
            'rough': 0.00004,
            'A': 0.0,
            'B': 0.0,
            'C': 10.0,
            'P': 0.0,
            'Q': 0.0,
            'R': 0.0,
            'S': 0.0,
            'T': 75.0,
            'power_type': 'Grid',
            'rate': 1.0,
        },
        {
            'name': 'P2',
            'is_pump': True,
            'min_pumps': 0,
            'max_pumps': 1,
            'MinRPM': 1000,
            'DOL': 1000,
            'L': 0.01,
            'd': 0.2,
            'rough': 0.00004,
            'A': 0.0,
            'B': 0.0,
            'C': 0.0,
            'P': 0.0,
            'Q': 0.0,
            'R': 0.0,
            'S': 0.0,
            'T': 75.0,
            'power_type': 'Grid',
            'rate': 1.0,
        },
    ]
    terminal = {'name': 'T', 'elev': 0.0}
    FLOW = 100.0
    KV_list = [1000.0, 1000.0]
    rho_list = [850.0, 850.0, 850.0]

    result = pipeline_model.solve_pipeline(
        stations,
        terminal,
        FLOW,
        KV_list,
        rho_list,
        RateDRA=0.0,
        Price_HSD=0.0,
        Fuel_density=1.0,
        Ambient_temp=25.0,
    )

    assert result['num_pumps_p1'] == 1, 'Origin station should enforce at least one pump'
    assert result['num_pumps_p2'] == 0, 'Downstream station may bypass pumps when min_pumps is 0'

