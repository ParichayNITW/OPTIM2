"""Tests for drag-reducer interpolation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import get_ppm_for_dr


def _sample_curve() -> dict[float, pd.DataFrame]:
    """Return a minimal drag-reducer curve for testing."""

    df = pd.DataFrame(
        {
            "%Drag Reduction": [0.0, 5.0, 10.0],
            "PPM": [0.0, 3.18, 6.0],
        }
    )
    return {3.0: df}


def test_get_ppm_for_dr_ceils_to_next_whole_ppm() -> None:
    """Ensure default rounding never rounds required ppm down."""

    data = _sample_curve()
    result = get_ppm_for_dr(3.0, 5.0, dra_curve_data=data)
    assert result == pytest.approx(4.0)


def test_get_ppm_for_dr_honours_custom_rounding_step() -> None:
    """A configurable increment should use ceiling to the next multiple."""

    data = _sample_curve()
    result = get_ppm_for_dr(3.0, 5.0, dra_curve_data=data, rounding_step=0.5)
    assert result == pytest.approx(3.5)


def test_get_ppm_for_dr_rounds_fractional_interpolations_upwards() -> None:
    """Interpolated values that fall between integers round up."""

    data = _sample_curve()
    interpolated = get_ppm_for_dr(3.0, 7.0, dra_curve_data=data)
    assert interpolated == pytest.approx(5.0)
