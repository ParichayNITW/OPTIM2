import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_optimization_app import INIT_DRA_COL, apply_dra_ppm, build_station_table


@pytest.mark.parametrize(
    "queue, expected_products, expected_volumes, expected_ppm",
    [
        (
            [
                {"volume": 70.0, "dra_ppm": 10.0},
                {"volume": 50.0, "dra_ppm": 25.0},
                {"volume": 60.0, "dra_ppm": 0.0},
                {"volume": 30.0, "dra_ppm": 15.0},
            ],
            ["Batch 1", "Batch 1", "Batch 2", "Batch 2", "Batch 3", "Batch 3"],
            [70.0, 50.0, 60.0, 20.0, 10.0, 30.0],
            [10.0, 25.0, 0.0, 15.0, 15.0, 0.0],
        ),
    ],
)
def test_apply_dra_ppm_splits_rows_at_queue_boundaries(
    queue, expected_products, expected_volumes, expected_ppm
) -> None:
    df = pd.DataFrame(
        [
            {
                "Product": "Batch 1",
                "Volume (m³)": 120.0,
                "Viscosity (cSt)": 2.0,
                "Density (kg/m³)": 820.0,
            },
            {
                "Product": "Batch 2",
                "Volume (m³)": 80.0,
                "Viscosity (cSt)": 3.0,
                "Density (kg/m³)": 830.0,
            },
            {
                "Product": "Batch 3",
                "Volume (m³)": 40.0,
                "Viscosity (cSt)": 4.0,
                "Density (kg/m³)": 840.0,
            },
        ]
    )
    df[INIT_DRA_COL] = 0.0

    result = apply_dra_ppm(df, queue)

    assert result["Product"].tolist() == expected_products
    assert result["Volume (m³)"].tolist() == pytest.approx(expected_volumes)
    assert result["DRA ppm"].tolist() == pytest.approx(expected_ppm)
    assert result[INIT_DRA_COL].tolist() == pytest.approx(expected_ppm)

    # Original batch volumes should be preserved across splits.
    restored = result.groupby("Product")["Volume (m³)"].sum().reindex(df["Product"].unique())
    assert restored.tolist() == pytest.approx(df["Volume (m³)"].tolist())


def test_build_station_table_includes_dra_profile_columns() -> None:
    """Station tables should expose inlet/outlet ppm and profile strings."""

    res = {
        'stations_used': [
            {
                'name': 'station_a',
                'orig_name': 'Station A',
                'is_pump': True,
            }
        ],
        'pipeline_flow_station_a': 1000.0,
        'pipeline_flow_in_station_a': 1000.0,
        'loopline_flow_station_a': 0.0,
        'pump_flow_station_a': 1000.0,
        'power_cost_station_a': 0.0,
        'dra_cost_station_a': 0.0,
        'dra_ppm_station_a': 12.0,
        'dra_ppm_loop_station_a': 0.0,
        'num_pumps_station_a': 1,
        'efficiency_station_a': 80.0,
        'pump_bkw_station_a': 100.0,
        'motor_kw_station_a': 110.0,
        'reynolds_station_a': 1.0,
        'head_loss_station_a': 5.0,
        'head_loss_kgcm2_station_a': 0.5,
        'velocity_station_a': 2.0,
        'residual_head_station_a': 30.0,
        'rh_kgcm2_station_a': 3.0,
        'sdh_station_a': 35.0,
        'sdh_kgcm2_station_a': 3.5,
        'maop_station_a': 80.0,
        'maop_kgcm2_station_a': 8.0,
        'drag_reduction_station_a': 40.0,
        'drag_reduction_loop_station_a': 0.0,
        'rho_station_a': 850.0,
        'velocity_loop_station_a': 0.0,
        'reynolds_loop_station_a': 0.0,
        'friction_station_a': 0.01,
        'friction_loop_station_a': 0.0,
        'maop_loop_station_a': 0.0,
        'maop_loop_kgcm2_station_a': 0.0,
        'coef_A_station_a': 0.0,
        'coef_B_station_a': 0.0,
        'coef_C_station_a': 0.0,
        'coef_P_station_a': 0.0,
        'coef_Q_station_a': 0.0,
        'coef_R_station_a': 0.0,
        'coef_S_station_a': 0.0,
        'coef_T_station_a': 0.0,
        'min_rpm_station_a': 900.0,
        'dol_station_a': 1200.0,
        'pump_details_station_a': [],
        'dra_profile_station_a': [
            {'length_km': 2.0, 'dra_ppm': 12.0},
            {'length_km': 1.0, 'dra_ppm': 0.0},
            {'length_km': 3.0, 'dra_ppm': 10.0},
        ],
        'dra_treated_length_station_a': 5.0,
        'dra_inlet_ppm_station_a': 12.0,
        'dra_outlet_ppm_station_a': 10.0,
    }

    base_stations = [
        {
            'name': 'Station A',
            'is_pump': True,
            'pump_names': ['Pump 1'],
            'max_pumps': 1,
            'min_pumps': 1,
            'L': 8.0,
        }
    ]

    df = build_station_table(res, base_stations)
    assert 'DRA Inlet PPM' in df.columns
    assert 'DRA Outlet PPM' in df.columns
    assert 'DRA Treated Length (km)' in df.columns
    assert 'DRA Untreated Length (km)' in df.columns
    assert 'DRA Profile (km@ppm)' in df.columns

    row = df.iloc[0]
    assert row['DRA Inlet PPM'] == pytest.approx(12.0, rel=1e-9)
    assert row['DRA Outlet PPM'] == pytest.approx(10.0, rel=1e-9)
    assert row['DRA Treated Length (km)'] == pytest.approx(5.0, rel=1e-9)
    assert row['DRA Untreated Length (km)'] == pytest.approx(3.0, rel=1e-9)
    profile_str = row['DRA Profile (km@ppm)']
    assert isinstance(profile_str, str)
    assert '2.00 km @ 12.00 ppm' in profile_str
    assert '1.00 km @ 0.00 ppm' in profile_str
    assert '3.00 km @ 10.00 ppm' in profile_str


def test_build_station_table_computes_defaults_when_solver_omits_metrics() -> None:
    """Fallback calculations should populate treated/untreated metrics."""

    res = {
        'stations_used': [
            {
                'name': 'station_b',
                'orig_name': 'Station B',
                'is_pump': True,
            }
        ],
        'pipeline_flow_station_b': 800.0,
        'loopline_flow_station_b': 0.0,
        'pump_flow_station_b': 800.0,
        'power_cost_station_b': 0.0,
        'dra_cost_station_b': 0.0,
        'dra_ppm_station_b': 5.0,
        'dra_ppm_loop_station_b': 0.0,
        'num_pumps_station_b': 1,
        'efficiency_station_b': 75.0,
        'pump_bkw_station_b': 90.0,
        'motor_kw_station_b': 95.0,
        'reynolds_station_b': 1.0,
        'head_loss_station_b': 4.0,
        'head_loss_kgcm2_station_b': 0.4,
        'velocity_station_b': 1.5,
        'residual_head_station_b': 25.0,
        'rh_kgcm2_station_b': 2.5,
        'sdh_station_b': 28.0,
        'sdh_kgcm2_station_b': 2.8,
        'maop_station_b': 70.0,
        'maop_kgcm2_station_b': 7.0,
        'drag_reduction_station_b': 30.0,
        'drag_reduction_loop_station_b': 0.0,
        'dra_profile_station_b': [
            {'length_km': 4.0, 'dra_ppm': 5.0},
            {'length_km': 6.0, 'dra_ppm': 0.0},
        ],
    }

    base_stations = [
        {
            'name': 'Station B',
            'is_pump': True,
            'pump_names': ['Pump X'],
            'max_pumps': 1,
            'min_pumps': 1,
            'L': 12.0,
        }
    ]

    df = build_station_table(res, base_stations)
    row = df.iloc[0]

    assert row['DRA Treated Length (km)'] == pytest.approx(4.0, rel=1e-9)
    assert row['DRA Untreated Length (km)'] == pytest.approx(8.0, rel=1e-9)
    assert row['DRA Inlet PPM'] == pytest.approx(5.0, rel=1e-9)
    assert row['DRA Outlet PPM'] == pytest.approx(0.0, rel=1e-9)
    profile_str = row['DRA Profile (km@ppm)']
    assert '4.00 km @ 5.00 ppm' in profile_str
    assert '6.00 km @ 0.00 ppm' in profile_str


def test_build_station_table_includes_zero_ppm_profile_for_floorless_station() -> None:
    """Stations without a floor should still report zero-ppm segments."""

    res = {
        'stations_used': [
            {
                'name': 'station_c',
                'orig_name': 'Station C',
                'is_pump': True,
            }
        ],
        'pipeline_flow_station_c': 900.0,
        'loopline_flow_station_c': 0.0,
        'pump_flow_station_c': 900.0,
        'power_cost_station_c': 0.0,
        'dra_cost_station_c': 0.0,
        'dra_ppm_station_c': 0.0,
        'dra_ppm_loop_station_c': 0.0,
        'num_pumps_station_c': 1,
        'efficiency_station_c': 70.0,
        'pump_bkw_station_c': 85.0,
        'motor_kw_station_c': 92.0,
        'reynolds_station_c': 1.0,
        'head_loss_station_c': 3.5,
        'head_loss_kgcm2_station_c': 0.35,
        'velocity_station_c': 1.7,
        'residual_head_station_c': 22.0,
        'rh_kgcm2_station_c': 2.2,
        'sdh_station_c': 24.0,
        'sdh_kgcm2_station_c': 2.4,
        'maop_station_c': 65.0,
        'maop_kgcm2_station_c': 6.5,
        'drag_reduction_station_c': 0.0,
        'drag_reduction_loop_station_c': 0.0,
        'dra_profile_station_c': [
            {'length_km': 2.5, 'dra_ppm': 0.0},
            {'length_km': 1.5, 'dra_ppm': 0.0},
        ],
        'dra_treated_length_station_c': 0.0,
        'dra_inlet_ppm_station_c': 0.0,
        'dra_outlet_ppm_station_c': 0.0,
    }

    base_stations = [
        {
            'name': 'Station C',
            'is_pump': True,
            'pump_names': ['Pump Zero'],
            'max_pumps': 1,
            'min_pumps': 1,
            'L': 4.0,
            'dra_floor_ppm_min': 0.0,
        }
    ]

    df = build_station_table(res, base_stations)
    row = df.iloc[0]

    assert row['DRA Treated Length (km)'] == pytest.approx(0.0, rel=1e-9)
    assert row['DRA Untreated Length (km)'] == pytest.approx(4.0, rel=1e-9)
    assert row['DRA Inlet PPM'] == pytest.approx(0.0, rel=1e-9)
    assert row['DRA Outlet PPM'] == pytest.approx(0.0, rel=1e-9)
    profile_str = row['DRA Profile (km@ppm)']
    assert profile_str.strip() != ''
    assert '2.50 km @ 0.00 ppm' in profile_str
    assert '1.50 km @ 0.00 ppm' in profile_str
