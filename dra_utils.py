"""Drag reducer (DRA) helper utilities based on the Burger equation."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

K1 = 0.1644
K2 = -0.04705
FT_PER_M = 3.28083989501312


def _normalise_rounding_step(step: float | None) -> float:
    """Return a validated rounding increment for PPM requirements."""

    try:
        step_value = 1.0 if step is None else float(step)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        step_value = 1.0
    if step_value <= 0.0:
        step_value = 1.0
    return step_value


def _round_cache_key(*values: float, precision: int = 6) -> Tuple[float, ...]:
    """Return a tuple suitable for memoisation keyed by rounded ``values``."""

    rounded = []
    for val in values:
        try:
            rounded.append(round(float(val), precision))
        except (TypeError, ValueError):
            rounded.append(0.0)
    return tuple(rounded)


def _velocity_ft_s(velocity_mps: float) -> float:
    try:
        vel = float(velocity_mps)
    except (TypeError, ValueError):
        return 0.0
    return max(vel, 0.0) * FT_PER_M


def _diameter_ft(diameter_m: float) -> float:
    try:
        dia = float(diameter_m)
    except (TypeError, ValueError):
        return 0.0
    return max(dia, 0.0) * FT_PER_M


def _compute_drag_reduction(visc: float, ppm: float, velocity_mps: float, diameter_m: float) -> float:
    """Return % drag reduction using the Burger equation."""

    try:
        visc_val = float(visc)
        ppm_val = float(ppm)
    except (TypeError, ValueError):
        return 0.0
    if visc_val <= 0.0 or ppm_val <= 0.0:
        return 0.0

    vel_ft_s = _velocity_ft_s(velocity_mps)
    dia_ft = _diameter_ft(diameter_m)
    if vel_ft_s <= 0.0 or dia_ft <= 0.0:
        return 0.0

    denom = visc_val * (dia_ft ** 0.4)
    if denom <= 0.0:
        return 0.0

    try:
        ratio = ppm_val / denom
    except ZeroDivisionError:
        return 0.0
    if ratio <= 0.0:
        return 0.0

    argument = vel_ft_s * math.sqrt(ratio)
    if argument <= 0.0:
        return 0.0

    try:
        dr_fraction = K1 * math.log(argument) + K2
    except (ValueError, OverflowError):
        return 0.0
    if not math.isfinite(dr_fraction):
        return 0.0
    return max(dr_fraction * 100.0, 0.0)


def _round_up(value: float, step: float) -> float:
    if value <= 0.0:
        return 0.0
    quotient = value / step
    nearest = round(quotient)
    if math.isclose(quotient, nearest, rel_tol=1e-9, abs_tol=1e-9):
        return nearest * step
    return math.ceil(quotient) * step


@lru_cache(maxsize=32768)
def _ppm_for_dr_cached(visc: float, dr_percent: float, velocity_mps: float, diameter_m: float, step: float) -> float:
    if dr_percent <= 0.0:
        return 0.0
    try:
        visc_val = float(visc)
        dr_val = float(dr_percent)
    except (TypeError, ValueError):
        return 0.0
    if visc_val <= 0.0:
        return 0.0

    vel_ft_s = _velocity_ft_s(velocity_mps)
    dia_ft = _diameter_ft(diameter_m)
    if vel_ft_s <= 0.0 or dia_ft <= 0.0:
        return 0.0

    dr_fraction = dr_val / 100.0
    exponent = (dr_fraction - K2) / K1
    try:
        argument = math.exp(exponent)
    except OverflowError:
        argument = float('inf')
    if not math.isfinite(argument) or argument <= 0.0:
        return 0.0

    denom = vel_ft_s ** 2
    if denom <= 0.0:
        return 0.0

    ppm_raw = argument * argument * visc_val * (dia_ft ** 0.4) / denom
    if not math.isfinite(ppm_raw):
        return 0.0
    return _round_up(ppm_raw, step)


@lru_cache(maxsize=32768)
def _ppm_for_dr_exact_cached(visc: float, dr_percent: float, velocity_mps: float, diameter_m: float) -> float:
    if dr_percent <= 0.0:
        return 0.0
    try:
        visc_val = float(visc)
        dr_val = float(dr_percent)
    except (TypeError, ValueError):
        return 0.0
    if visc_val <= 0.0:
        return 0.0

    vel_ft_s = _velocity_ft_s(velocity_mps)
    dia_ft = _diameter_ft(diameter_m)
    if vel_ft_s <= 0.0 or dia_ft <= 0.0:
        return 0.0

    dr_fraction = dr_val / 100.0
    exponent = (dr_fraction - K2) / K1
    try:
        argument = math.exp(exponent)
    except OverflowError:
        argument = float("inf")
    if not math.isfinite(argument) or argument <= 0.0:
        return 0.0

    denom = vel_ft_s ** 2
    if denom <= 0.0:
        return 0.0

    ppm_raw = argument * argument * visc_val * (dia_ft ** 0.4) / denom
    if not math.isfinite(ppm_raw):
        return 0.0
    return ppm_raw


@lru_cache(maxsize=32768)
def _dr_for_ppm_cached(visc: float, ppm: float, velocity_mps: float, diameter_m: float) -> float:
    try:
        ppm_val = float(ppm)
        visc_val = float(visc)
    except (TypeError, ValueError):
        return 0.0
    if ppm_val <= 0.0 or visc_val <= 0.0:
        return 0.0

    vel_ft_s = _velocity_ft_s(velocity_mps)
    dia_ft = _diameter_ft(diameter_m)
    if vel_ft_s <= 0.0 or dia_ft <= 0.0:
        return 0.0

    denom = visc_val * (dia_ft ** 0.4)
    if denom <= 0.0:
        return 0.0

    try:
        ratio = ppm_val / denom
    except ZeroDivisionError:
        return 0.0
    if ratio <= 0.0:
        return 0.0

    argument = vel_ft_s * math.sqrt(ratio)
    if argument <= 0.0:
        return 0.0

    try:
        dr_fraction = K1 * math.log(argument) + K2
    except (ValueError, OverflowError):
        return 0.0
    if not math.isfinite(dr_fraction):
        return 0.0
    return max(dr_fraction * 100.0, 0.0)


def get_ppm_for_dr(
    visc: float,
    dr_percent: float,
    velocity_mps: float,
    diameter_m: float,
    rounding_step: float | None = None,
) -> float:
    """Compute the PPM required for ``dr_percent`` drag reduction."""

    step_value = _normalise_rounding_step(rounding_step)
    key = _round_cache_key(visc, dr_percent, velocity_mps, diameter_m, step_value)
    return _ppm_for_dr_cached(*key)


def get_ppm_for_dr_exact(
    visc: float,
    dr_percent: float,
    velocity_mps: float,
    diameter_m: float,
) -> float:
    """Compute the unrounded PPM required for ``dr_percent`` drag reduction."""

    key = _round_cache_key(visc, dr_percent, velocity_mps, diameter_m)
    return _ppm_for_dr_exact_cached(*key)


def get_dr_for_ppm(
    visc: float,
    ppm: float,
    velocity_mps: float,
    diameter_m: float,
) -> float:
    """Compute % drag reduction delivered by ``ppm``."""

    key = _round_cache_key(visc, ppm, velocity_mps, diameter_m)
    return _dr_for_ppm_cached(*key)


def compute_drag_reduction(visc: float, ppm: float, velocity_mps: float, diameter_m: float) -> float:
    """Return effective % drag reduction for ``ppm`` at viscosity ``visc``."""

    key = _round_cache_key(visc, ppm, velocity_mps, diameter_m)
    return _dr_for_ppm_cached(*key)


__all__ = [
    "get_ppm_for_dr",
    "get_ppm_for_dr_exact",
    "get_dr_for_ppm",
    "compute_drag_reduction",
]
