import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pipeline_model as pm


def _build_sample_pipeline():
    stations = [
        {
            'name': 'S1',
            'is_pump': True,
            'max_pumps': 1,
            'MinRPM': 100,
            'DOL': 100,
            'A': 0,
            'B': 0,
            'C': 120,
            'L': 100.0,
            'd': 0.5,
            'loopline': {'L': 100.0, 'd': 0.5},
        },
        {
            'name': 'S2',
            'is_pump': True,
            'max_pumps': 1,
            'MinRPM': 100,
            'DOL': 100,
            'A': 0,
            'B': 0,
            'C': 0,
            'L': 100.0,
            'd': 0.5,
        },
    ]
    terminal = {'name': 'T', 'min_residual': 10, 'elev': 0}
    FLOW = 1000
    KV_list = [1e-6, 1e-6]
    rho_list = [1000, 1000]
    return stations, terminal, FLOW, KV_list, rho_list


def test_bypass_enables_feasible_solution(monkeypatch):
    stations, terminal, FLOW, KV_list, rho_list = _build_sample_pipeline()

    # Simulate previous behaviour: downstream requirement uses original flows
    base_flows = [FLOW]
    for st in stations:
        base_flows.append(base_flows[-1] - float(st.get('delivery', 0.0)) + float(st.get('supply', 0.0)))

    orig = pm._downstream_requirement

    def old_req(stations_arg, idx, terminal_arg, segment_flows_arg, KV_list_arg, flow_override=None):
        return orig(stations_arg, idx, terminal_arg, base_flows, KV_list_arg, flow_override)

    monkeypatch.setattr(pm, '_downstream_requirement', old_req)
    res_old = pm.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, 0, 0, 0.85, 25)
    assert res_old.get('error')

    # Restore new implementation which recomputes flows after bypass
    monkeypatch.setattr(pm, '_downstream_requirement', orig)
    res_new = pm.solve_pipeline(stations, terminal, FLOW, KV_list, rho_list, 0, 0, 0.85, 25)
    assert not res_new.get('error')
    # Bypass should reduce flow reaching station S2
    assert res_new['bypass_next_s1'] == 1
    assert res_new['pipeline_flow_s2'] < FLOW

