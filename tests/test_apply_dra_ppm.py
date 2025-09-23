import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_optimization_app import INIT_DRA_COL, apply_dra_ppm


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
