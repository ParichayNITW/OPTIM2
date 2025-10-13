"""Tests for Burger-equation drag reducer helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dra_utils import get_ppm_for_dr, get_ppm_for_dr_exact


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
    exponent = ((dr_percent / 100.0) - k2) / k1
    argument = math.exp(exponent)
    return (
        VISCOSITY_CST
        * (diameter_ft ** 0.4)
        * (argument ** 2)
        / (velocity_ft_s ** 2)
    )


def test_get_ppm_for_dr_ceils_to_next_whole_ppm() -> None:
    """Default rounding should ceil to the next whole ppm."""

    raw_value = _expected_ppm(15.0)
    assert 1.0 < raw_value < 2.0
    result = get_ppm_for_dr(VISCOSITY_CST, 15.0, VELOCITY_MPS, DIAMETER_M)
    assert result == pytest.approx(math.ceil(raw_value))


def test_get_ppm_for_dr_honours_custom_rounding_step() -> None:
    """A custom rounding increment should ceil to that increment."""

    raw_value = _expected_ppm(25.0)
    assert 6.0 < raw_value < 6.5
    result = get_ppm_for_dr(
        VISCOSITY_CST,
        25.0,
        VELOCITY_MPS,
        DIAMETER_M,
        rounding_step=0.5,
    )
    assert result == pytest.approx(6.5)


def test_get_ppm_for_dr_rounds_fractional_values_upwards() -> None:
    """Fractional raw values must round up rather than down."""

    raw_value = _expected_ppm(20.0)
    assert 3.0 < raw_value < 4.0
    result = get_ppm_for_dr(VISCOSITY_CST, 20.0, VELOCITY_MPS, DIAMETER_M)
    assert result == pytest.approx(4.0)


def test_get_ppm_for_dr_exact_matches_user_reference_case() -> None:
    """The Burger inversion should reproduce the user-supplied example."""

    viscosity = 15.0
    drag_reduction = 24.77987421383648
    velocity = 2.0131566518546617
    inner_diameter = 0.7461504

    ppm_exact = get_ppm_for_dr_exact(
        viscosity,
        drag_reduction,
        velocity,
        inner_diameter,
    )
    assert ppm_exact == pytest.approx(17.770130549595464, rel=1e-9, abs=1e-9)

    ppm_rounded = get_ppm_for_dr(
        viscosity,
        drag_reduction,
        velocity,
        inner_diameter,
    )
    assert ppm_rounded == pytest.approx(18.0)
