import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pipeline_model as pm

HOURS = 24.0
PRICE_HSD = 100.0
FUEL_DENSITY = 820.0
RATE = 5.0


def compute_station_cost(flow, rpm, pump_def):
    details = pm._pump_head(pump_def, flow, rpm, sum(pump_def.get('combo', {}).values()) or pump_def.get('nop', 0))
    total = 0.0
    for d in details:
        eff = max(d['eff'], 1e-6)
        pump_bkw = (850.0 * flow * 9.81 * d['tdh']) / (3600.0 * 1000.0 * (eff / 100.0))
        mech_eff = 0.98 if d.get('power_type') == 'Diesel' else 0.95
        prime_kw = pump_bkw / mech_eff
        if d.get('power_type') == 'Diesel':
            sfc_val = d['data'].get('sfc', 0.0)
            fuel_per_kwh = (sfc_val * 1.34102) / FUEL_DENSITY if sfc_val else 0.0
            cost = prime_kw * HOURS * fuel_per_kwh * PRICE_HSD
        else:
            cost = prime_kw * HOURS * RATE
        d['power_cost'] = cost
        total += cost
    return total, details


def test_all_electric_cost_sum():
    pump_def = {
        'pump_types': {
            'A': {'A': 0.0, 'B': 0.0, 'C': 100.0, 'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0, 'T': 80.0, 'DOL': 1000.0, 'power_type': 'Electric'},
            'B': {'A': 0.0, 'B': 0.0, 'C': 80.0, 'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0, 'T': 70.0, 'DOL': 1000.0, 'power_type': 'Electric'},
        },
        'combo': {'A': 1, 'B': 1},
    }
    total, details = compute_station_cost(1000.0, 1000.0, pump_def)
    assert total == pytest.approx(sum(d['power_cost'] for d in details))


def test_mixed_diesel_electric_cost_sum():
    pump_def = {
        'pump_types': {
            'E': {'A': 0.0, 'B': 0.0, 'C': 90.0, 'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0, 'T': 75.0, 'DOL': 1000.0, 'power_type': 'Electric'},
            'D': {'A': 0.0, 'B': 0.0, 'C': 60.0, 'P': 0.0, 'Q': 0.0, 'R': 0.0, 'S': 0.0, 'T': 65.0, 'DOL': 1000.0, 'power_type': 'Diesel', 'sfc': 200.0},
        },
        'combo': {'E': 1, 'D': 1},
    }
    total, details = compute_station_cost(1000.0, 1000.0, pump_def)
    assert total == pytest.approx(sum(d['power_cost'] for d in details))
