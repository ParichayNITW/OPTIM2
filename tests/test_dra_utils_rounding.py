"""Tests for Burger-equation drag reducer helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import get_ppm_for_dr


VELOCITY_MPS = 1.5
DIAMETER_M = 0.7
VISCOSITY_CST = 3.0


def _expected_ppm(dr_percent: float) -> float:
    """Compute the unrounded ppm from the Burger equation."""

    k1 = 0.1644
    k2 = -0.04705
    ft_per_m = 3.28083989501312
    velocity_ft_s = VELOCITY_MPS * ft_per_m
    diameter_ft = DIAMETER_M * ft_per_m
    exponent = 2.0 * ((dr_percent / 100.0) - k2) / k1
    return math.exp(exponent) * VISCOSITY_CST * (diameter_ft ** 0.2) / velocity_ft_s


def test_get_ppm_for_dr_ceils_to_next_whole_ppm() -> None:
    """Default rounding should ceil to the next whole ppm."""

    raw_value = _expected_ppm(5.0)
    assert raw_value < math.ceil(raw_value)
    result = get_ppm_for_dr(VISCOSITY_CST, 5.0, VELOCITY_MPS, DIAMETER_M)
    assert result == pytest.approx(math.ceil(raw_value))


def test_get_ppm_for_dr_honours_custom_rounding_step() -> None:
    """A custom rounding increment should ceil to that increment."""

    raw_value = _expected_ppm(10.0)
    assert 4.0 < raw_value < 5.0
    result = get_ppm_for_dr(
        VISCOSITY_CST,
        10.0,
        VELOCITY_MPS,
        DIAMETER_M,
        rounding_step=0.5,
    )
    assert result == pytest.approx(4.5)


def test_get_ppm_for_dr_rounds_fractional_values_upwards() -> None:
    """Fractional raw values must round up rather than down."""

    raw_value = _expected_ppm(11.0)
    assert 4.8 < raw_value < 5.0
    result = get_ppm_for_dr(VISCOSITY_CST, 11.0, VELOCITY_MPS, DIAMETER_M)
    assert result == pytest.approx(5.0)
