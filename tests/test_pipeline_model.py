import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pipeline_model as pm


def _base_origin_station():
    return {
        'name': 'Origin',
        'is_pump': True,
        'pump_types': {
            'A': {
                'available': 2,
                'name': 'Alpha',
                'head_data': [[0, 100.0]],
                'eff_data': [[0, 0.8]],
                'power_type': 'Grid',
                'rate': 1.0,
                'sfc': 0.0,
                'MinRPM': 1000,
                'DOL': 2000,
            },
            'B': {
                'available': 2,
                'name': 'Beta',
                'head_data': [[0, 90.0]],
                'eff_data': [[0, 0.9]],
                'power_type': 'Grid',
                'rate': 0.5,
                'sfc': 0.0,
                'MinRPM': 900,
                'DOL': 1900,
            },
        },
        'D': 0.5,
        't': 0.01,
        'rough': 0.000045,
        'L': 10.0,
    }

def _downstream_station():
    return {
        'name': 'S1',
        'D': 0.5,
        't': 0.01,
        'rough': 0.000045,
        'L': 100.0,
        'is_pump': False,
    }


def test_enumerates_combinations_and_selects_min_cost(monkeypatch):
    origin = _base_origin_station()
    stations = [origin, _downstream_station()]
    stations_copy = copy.deepcopy(stations)

    combos = set()

    def fake_solver(stations_combo, terminal, FLOW, kv, rho, RateDRA, Price_HSD, linefill, **kwargs):
        numA = sum(1 for s in stations_combo if s.get('is_pump') and 'Alpha' in s['name'])
        numB = sum(1 for s in stations_combo if s.get('is_pump') and 'Beta' in s['name'])
        combos.add((numA, numB))
        head_total = sum(s['head_data'][0][1] for s in stations_combo if s.get('is_pump'))
        eff_list = [s['eff_data'][0][1] for s in stations_combo if s.get('is_pump')]
        cost = numA * 100 + (2 - numB) * 10
        return {'total_cost': cost, 'head_total': head_total, 'eff_list': eff_list}

    monkeypatch.setattr(pm, 'solve_pipeline', fake_solver)

    result = pm.solve_pipeline_multi_origin(
        stations,
        terminal={},
        FLOW=100.0,
        KV_list=[1.0, 1.0],
        rho_list=[1000.0, 1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    expected = {
        (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)
    }
    assert combos == expected
    assert result['pump_combo'] == {'Beta': 2}
    assert result['head_total'] == 180.0
    assert result['eff_list'] == [0.9, 0.9]
    assert stations == stations_copy
    assert result['stations_used'][-1] == stations_copy[-1]


def test_only_type_a(monkeypatch):
    origin = _base_origin_station()
    origin['pump_types'] = {'A': origin['pump_types']['A']}
    stations = [origin, _downstream_station()]

    combos = set()

    def fake_solver(stations_combo, terminal, FLOW, kv, rho, RateDRA, Price_HSD, linefill, **kwargs):
        numA = sum(1 for s in stations_combo if s.get('is_pump'))
        combos.add((numA, 0))
        return {
            'total_cost': float(numA),
            'head_total': sum(s['head_data'][0][1] for s in stations_combo if s.get('is_pump')),
            'eff_list': [s['eff_data'][0][1] for s in stations_combo if s.get('is_pump')],
        }

    monkeypatch.setattr(pm, 'solve_pipeline', fake_solver)

    result = pm.solve_pipeline_multi_origin(
        stations,
        terminal={},
        FLOW=100.0,
        KV_list=[1.0, 1.0],
        rho_list=[1000.0, 1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    assert combos == {(1, 0), (2, 0)}
    assert result['pump_combo'] == {'Alpha': 1}
    assert result['head_total'] == 100.0
    assert result['eff_list'] == [0.8]


def test_only_type_b(monkeypatch):
    origin = _base_origin_station()
    origin['pump_types'] = {'B': origin['pump_types']['B']}
    stations = [origin, _downstream_station()]

    combos = set()

    def fake_solver(stations_combo, terminal, FLOW, kv, rho, RateDRA, Price_HSD, linefill, **kwargs):
        numB = sum(1 for s in stations_combo if s.get('is_pump'))
        combos.add((0, numB))
        return {
            'total_cost': float(numB),
            'head_total': sum(s['head_data'][0][1] for s in stations_combo if s.get('is_pump')),
            'eff_list': [s['eff_data'][0][1] for s in stations_combo if s.get('is_pump')],
        }

    monkeypatch.setattr(pm, 'solve_pipeline', fake_solver)

    result = pm.solve_pipeline_multi_origin(
        stations,
        terminal={},
        FLOW=100.0,
        KV_list=[1.0, 1.0],
        rho_list=[1000.0, 1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    assert combos == {(0, 1), (0, 2)}
    assert result['pump_combo'] == {'Beta': 1}
    assert result['head_total'] == 90.0
    assert result['eff_list'] == [0.9]


def test_no_feasible_combination_returns_error(monkeypatch):
    origin = _base_origin_station()
    stations = [origin, _downstream_station()]

    def failing_solver(*args, **kwargs):
        return {"error": True}

    monkeypatch.setattr(pm, "solve_pipeline", failing_solver)

    result = pm.solve_pipeline_multi_origin(
        stations,
        terminal={},
        FLOW=100.0,
        KV_list=[1.0, 1.0],
        rho_list=[1000.0, 1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    assert result["error"] is True
    assert "No feasible pump combination" in result["message"]
    attempts = {(c["A"], c["B"]) for c in result.get("attempted_combos", [])}
    expected = {
        (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)
    }
    assert attempts == expected


def test_propagates_common_failure_message(monkeypatch):
    origin = _base_origin_station()
    stations = [origin, _downstream_station()]

    def failing_solver(*args, **kwargs):
        return {"error": True, "message": "NEOS offline"}

    monkeypatch.setattr(pm, "solve_pipeline", failing_solver)

    result = pm.solve_pipeline_multi_origin(
        stations,
        terminal={},
        FLOW=100.0,
        KV_list=[1.0, 1.0],
        rho_list=[1000.0, 1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    assert result["error"] is True
    assert result["message"] == "NEOS offline"
    attempts = result.get("attempted_combos", [])
    assert len(attempts) == 8
    assert all(a.get("message") == "NEOS offline" for a in attempts)


def test_generate_origin_combinations_respects_max_total_and_casts():
    combos = pm.generate_origin_combinations(2.0, 3.0, max_total=3)
    assert combos == [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1)]


def test_solver_failure_returns_error(monkeypatch, capsys):
    station = {
        'name': 'Origin',
        'D': 0.5,
        't': 0.01,
        'rough': 0.000045,
        'L': 10.0,
        'is_pump': True,
        'head_data': [[0, 100], [1, 100], [2, 100]],
        'eff_data': [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
        'MinRPM': 1000,
        'DOL': 2000,
        'pump_types': {"A": {"available": 1, "name": "A", "head_data": [[0,100],[1,100],[2,100]],
                              "eff_data": [[0,1],[1,1],[2,1],[3,1],[4,1]], "power_type":"Grid", "rate":1.0,
                              "sfc":0.0, "MinRPM":1000, "DOL":2000}}
    }
    terminal = {'name': 'T', 'min_residual': 50.0}

    class DummyManager:
        def solve(self, *args, **kwargs):
            print('neo stdout')
            print('neo stderr', file=sys.stderr)
            raise ValueError('no Options line found')

    monkeypatch.setattr(pm, 'SolverManagerFactory', lambda *args, **kwargs: DummyManager())
    monkeypatch.setattr(pm, '_neos_available', lambda: True)

    res = pm.solve_pipeline(
        [station],
        terminal,
        FLOW=100.0,
        KV_list=[1.0],
        rho_list=[1000.0],
        RateDRA=0.0,
        Price_HSD=1.0,
    )

    out, err = capsys.readouterr()
    assert out == ''
    assert err == ''
    assert res['error'] is True
    assert 'no Options line found' in res['message']
    assert 'neo stdout' in res['solver_output']
    assert 'neo stderr' in res['solver_output']
