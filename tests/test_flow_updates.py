import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pipeline_model as pm


def _old_update(segment_flows, stations, idx, next_flow):
    flows = segment_flows.copy()
    flows[idx + 1] = next_flow
    for j in range(idx + 1, len(stations)):
        delivery_j = float(stations[j].get('delivery', 0.0))
        supply_j = float(stations[j].get('supply', 0.0))
        flows[j + 1] = flows[j] - delivery_j + supply_j
    return flows


def _sample_data():
    stations = [
        {'delivery': 100, 'supply': 0},
        {'delivery': 0, 'supply': 50},
        {'delivery': 0, 'supply': 0},
    ]
    flows = [1000.0]
    for stn in stations:
        prev = flows[-1]
        flows.append(prev - stn.get('delivery', 0.0) + stn.get('supply', 0.0))
    return stations, flows


def test_update_segment_flows_non_bypass():
    stations, segment_flows = _sample_data()
    idx = 0
    next_flow = 920.0
    expected = _old_update(segment_flows, stations, idx, next_flow)
    updated = pm._update_segment_flows(segment_flows, stations, idx, next_flow)
    assert updated == expected
    assert segment_flows == [1000.0, 900.0, 950.0, 950.0]


def test_update_segment_flows_bypass():
    stations, segment_flows = _sample_data()
    idx = 0
    next_flow = 800.0
    expected = _old_update(segment_flows, stations, idx, next_flow)
    updated = pm._update_segment_flows(segment_flows, stations, idx, next_flow)
    assert updated == expected
    assert segment_flows == [1000.0, 900.0, 950.0, 950.0]
