
"""Drag reducer (DRA) helper utilities.

Adds inverse interpolation (ppm_to_dr) and keeps get_ppm_for_dr API.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Mapping of viscosity (cSt) to CSV file name
DRA_CSV_FILES: Dict[float, str] = {
    1: "1 cst.csv",
    2: "2 cst.csv",
    2.5: "2.5 cst.csv",
    3: "3 cst.csv",
    3.5: "3.5 cst.csv",
    4: "4 cst.csv",
    4.5: "4.5 cst.csv",
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv",
}

# Load the drag-reducer curves lazily at import time
DRA_CURVE_DATA: Dict[float, pd.DataFrame | None] = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            # Ensure required columns exist
            if "%Drag Reduction" in df.columns and "PPM" in df.columns:
                df = df[["%Drag Reduction", "PPM"]].dropna().sort_values("%Drag Reduction")
                DRA_CURVE_DATA[cst] = df.reset_index(drop=True)
            else:
                DRA_CURVE_DATA[cst] = None
        except Exception:
            DRA_CURVE_DATA[cst] = None
    else:
        DRA_CURVE_DATA[cst] = None


def _ppm_from_df(df: pd.DataFrame, dr: float) -> float:
    """Return the PPM value for ``dr`` using breakpoints in ``df``."""
    x = df['%Drag Reduction'].values.astype(float)
    y = df['PPM'].values.astype(float)
    if dr <= x[0]:
        return float(y[0])
    if dr >= x[-1]:
        return float(y[-1])
    return float(np.interp(dr, x, y))


def _dr_from_df(df: pd.DataFrame, ppm: float) -> float:
    """Return %Drag Reduction for a given ``ppm`` by inverse interpolation of ``df``."""
    x = df['%Drag Reduction'].values.astype(float)
    y = df['PPM'].values.astype(float)
    if ppm <= y[0]:
        return float(x[0])
    if ppm >= y[-1]:
        return float(x[-1])
    # Interpolate inverse: x(y)
    return float(np.interp(ppm, y, x))


def _nearest_bounds(visc: float, data: Dict[float, pd.DataFrame | None]) -> Tuple[float, float]:
    cst_list = sorted([c for c in data.keys() if data[c] is not None])
    if not cst_list:
        return (visc, visc)
    if visc <= cst_list[0]:
        return (cst_list[0], cst_list[0])
    if visc >= cst_list[-1]:
        return (cst_list[-1], cst_list[-1])
    lower = max(c for c in cst_list if c <= visc)
    upper = min(c for c in cst_list if c >= visc)
    return (lower, upper)


_DEFAULT_CURVE_SENTINEL = object()
_PPM_CACHE: Dict[tuple[float, ...], float] = {}
_DR_CACHE: Dict[tuple[float, ...], float] = {}
_PPM_BOUND_CACHE: Dict[tuple[float, ...], tuple[float, float]] = {}


def get_ppm_bounds(
    visc: float,
    dra_curve_data: Dict[float, pd.DataFrame | None] = _DEFAULT_CURVE_SENTINEL,
) -> tuple[float, float]:
    """Return the minimum and maximum PPM supported by available curves."""

    use_global = False
    if dra_curve_data is _DEFAULT_CURVE_SENTINEL or dra_curve_data is DRA_CURVE_DATA:
        dra_curve_data = DRA_CURVE_DATA
        use_global = True

    visc = float(visc)
    cache_key = _round_cache_key(visc)
    if use_global:
        cached = _PPM_BOUND_CACHE.get(cache_key)
        if cached is not None:
            return cached
    lower, upper = _nearest_bounds(visc, dra_curve_data)

    def _bounds_for(df: pd.DataFrame | None) -> tuple[float, float]:
        if df is None or df.empty:
            return (0.0, 0.0)
        ppm_vals = df["PPM"].values.astype(float)
        if ppm_vals.size == 0:
            return (0.0, 0.0)
        return (float(np.nanmin(ppm_vals)), float(np.nanmax(ppm_vals)))

    if lower not in dra_curve_data or dra_curve_data[lower] is None:
        return (0.0, 0.0)

    lower_bounds = _bounds_for(dra_curve_data[lower])
    if lower == upper:
        return lower_bounds

    upper_bounds = _bounds_for(dra_curve_data.get(upper))
    min_ppm = lower_bounds[0] if upper_bounds[0] == 0.0 else min(lower_bounds[0], upper_bounds[0])
    max_ppm = max(lower_bounds[1], upper_bounds[1])
    result = (min_ppm, max_ppm)
    if use_global:
        if len(_PPM_BOUND_CACHE) > 8192:
            _PPM_BOUND_CACHE.clear()
        _PPM_BOUND_CACHE[cache_key] = result
    return result


def _round_cache_key(*values: float, precision: int = 2) -> tuple[float, ...]:
    """Return a tuple suitable for memoisation keyed by rounded ``values``."""

    return tuple(round(float(val), precision) for val in values)


def _compute_ppm_for_dr(
    visc: float,
    dr: float,
    dra_curve_data: Dict[float, pd.DataFrame | None],
) -> float:
    """Internal helper implementing :func:`get_ppm_for_dr` without caching."""

    visc = float(visc)
    lower, upper = _nearest_bounds(visc, dra_curve_data)
    if lower not in dra_curve_data or dra_curve_data[lower] is None:
        return 0.0

    def round_ppm(val: float, step: float = 1.0) -> float:
        """Round ``val`` up to the next multiple of ``step``.

        ``step`` defaults to ``1.0`` so that injection requirements are
        expressed in whole PPM increments.  Callers that require different
        granularity may provide their preferred increment.
        """

        try:
            step_value = float(step)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            step_value = 1.0
        if step_value <= 0.0:
            step_value = 1.0
        if val <= 0.0:
            return 0.0
        return math.ceil(val / step_value) * step_value

    if lower == upper:
        return round_ppm(_ppm_from_df(dra_curve_data[lower], dr))

    df_lower = dra_curve_data[lower]
    df_upper = dra_curve_data[upper]
    ppm_lower = _ppm_from_df(df_lower, dr)
    ppm_upper = _ppm_from_df(df_upper, dr)
    ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
    return round_ppm(float(ppm_interp))


def get_ppm_for_dr(
    visc: float,
    dr: float,
    dra_curve_data: Dict[float, pd.DataFrame | None] = _DEFAULT_CURVE_SENTINEL,
) -> float:
    """Interpolate PPM for a given drag reduction and viscosity.

    Returns the PPM value rounded up to the next whole PPM (or custom
    increment).
    """

    if dra_curve_data is _DEFAULT_CURVE_SENTINEL or dra_curve_data is DRA_CURVE_DATA:
        dra_curve_data = DRA_CURVE_DATA
        key = _round_cache_key(visc, dr)
        cached = _PPM_CACHE.get(key)
        if cached is not None:
            return cached
        result = _compute_ppm_for_dr(visc, dr, dra_curve_data)
        if len(_PPM_CACHE) > 8192:
            _PPM_CACHE.clear()
        _PPM_CACHE[key] = result
        return result

    return _compute_ppm_for_dr(visc, dr, dra_curve_data)


def _compute_dr_for_ppm(
    visc: float,
    ppm: float,
    dra_curve_data: Dict[float, pd.DataFrame | None],
) -> float:
    """Internal helper implementing :func:`get_dr_for_ppm` without caching."""

    visc = float(visc)
    lower, upper = _nearest_bounds(visc, dra_curve_data)
    if lower not in dra_curve_data or dra_curve_data[lower] is None:
        return 0.0

    if lower == upper:
        return _dr_from_df(dra_curve_data[lower], ppm)

    df_lower = dra_curve_data[lower]
    df_upper = dra_curve_data[upper]
    dr_lower = _dr_from_df(df_lower, ppm)
    dr_upper = _dr_from_df(df_upper, ppm)
    dr_interp = np.interp(visc, [lower, upper], [dr_lower, dr_upper])
    return float(dr_interp)


def get_dr_for_ppm(
    visc: float,
    ppm: float,
    dra_curve_data: Dict[float, pd.DataFrame | None] = _DEFAULT_CURVE_SENTINEL,
) -> float:
    """Inverse: interpolate %Drag Reduction for a given PPM and viscosity."""

    if dra_curve_data is _DEFAULT_CURVE_SENTINEL or dra_curve_data is DRA_CURVE_DATA:
        dra_curve_data = DRA_CURVE_DATA
        key = _round_cache_key(visc, ppm)
        cached = _DR_CACHE.get(key)
        if cached is not None:
            return cached
        result = _compute_dr_for_ppm(visc, ppm, dra_curve_data)
        if len(_DR_CACHE) > 8192:
            _DR_CACHE.clear()
        _DR_CACHE[key] = result
        return result

    return _compute_dr_for_ppm(visc, ppm, dra_curve_data)


def compute_drag_reduction(visc: float, ppm: float) -> float:
    """Return effective % drag reduction for ``ppm`` at viscosity ``visc``."""
    if ppm <= 0:
        return 0.0
    return get_dr_for_ppm(visc, ppm)


__all__ = [
    "DRA_CSV_FILES",
    "DRA_CURVE_DATA",
    "get_ppm_for_dr",
    "get_dr_for_ppm",
    "compute_drag_reduction",
]
