
"""Drag reducer (DRA) helper utilities.

Adds inverse interpolation (ppm_to_dr) and keeps get_ppm_for_dr API.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Tuple, MutableMapping

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

# Cache of loaded drag-reducer curves.  Curves are loaded on demand via
# ``load_curve`` below.  We use an ``OrderedDict`` so we can enforce a small
# LRU-style cache and avoid unbounded growth if many viscosities are requested.
DRA_CURVE_DATA: OrderedDict[float, pd.DataFrame | None] = OrderedDict()


def load_curve(
    visc: float,
    cache: MutableMapping[float, pd.DataFrame | None] | None = None,
    max_size: int = 8,
) -> pd.DataFrame | None:
    """Return DataFrame for ``visc`` loading and caching it on demand.

    Parameters
    ----------
    visc:
        Viscosity in cSt for which to load the curve.
    cache:
        Mapping used to store cached curves.  Defaults to the module-level
        ``DRA_CURVE_DATA``.  If ``cache`` is an ``OrderedDict`` an LRU policy is
        applied keeping the size below ``max_size``.
    max_size:
        Maximum number of curves to retain in ``cache`` when it is an
        ``OrderedDict``.
    """

    cache = DRA_CURVE_DATA if cache is None else cache
    visc = float(visc)

    if visc in cache:
        if isinstance(cache, OrderedDict):
            cache.move_to_end(visc)
        return cache[visc]

    df: pd.DataFrame | None = None
    fname = DRA_CSV_FILES.get(visc)
    if fname and os.path.exists(fname):
        try:
            tmp = pd.read_csv(fname)
            if "%Drag Reduction" in tmp.columns and "PPM" in tmp.columns:
                df = (
                    tmp[["%Drag Reduction", "PPM"]]
                    .dropna()
                    .sort_values("%Drag Reduction")
                    .reset_index(drop=True)
                )
        except Exception:
            df = None

    cache[visc] = df
    if isinstance(cache, OrderedDict):
        cache.move_to_end(visc)
        while len(cache) > max_size:
            cache.popitem(last=False)

    return df


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


def _nearest_bounds(visc: float) -> Tuple[float, float]:
    """Return the viscosities bounding ``visc`` with available CSV files."""
    available = sorted(
        c for c, fname in DRA_CSV_FILES.items() if os.path.exists(fname)
    )
    if not available:
        return (visc, visc)
    if visc <= available[0]:
        return (available[0], available[0])
    if visc >= available[-1]:
        return (available[-1], available[-1])
    lower = max(c for c in available if c <= visc)
    upper = min(c for c in available if c >= visc)
    return (lower, upper)


def get_ppm_for_dr(
    visc: float,
    dr: float,
    dra_curve_data: MutableMapping[float, pd.DataFrame | None] | None = None,
) -> float:
    """Interpolate PPM for a given drag reduction and viscosity.

    Returns the PPM value rounded to the nearest 0.5.
    """
    cache = DRA_CURVE_DATA if dra_curve_data is None else dra_curve_data
    visc = float(visc)
    lower, upper = _nearest_bounds(visc)
    df_lower = load_curve(lower, cache)
    if df_lower is None:
        return 0.0

    def round_ppm(val: float, step: float = 0.5) -> float:
        return round(val / step) * step

    if lower == upper:
        return round_ppm(_ppm_from_df(df_lower, dr))

    df_upper = load_curve(upper, cache)
    if df_upper is None:
        return 0.0
    ppm_lower = _ppm_from_df(df_lower, dr)
    ppm_upper = _ppm_from_df(df_upper, dr)
    ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
    return round_ppm(float(ppm_interp))


def get_dr_for_ppm(
    visc: float,
    ppm: float,
    dra_curve_data: MutableMapping[float, pd.DataFrame | None] | None = None,
) -> float:
    """Inverse: interpolate %Drag Reduction for a given PPM and viscosity."""
    cache = DRA_CURVE_DATA if dra_curve_data is None else dra_curve_data
    visc = float(visc)
    lower, upper = _nearest_bounds(visc)
    df_lower = load_curve(lower, cache)
    if df_lower is None:
        return 0.0

    if lower == upper:
        return _dr_from_df(df_lower, ppm)

    df_upper = load_curve(upper, cache)
    if df_upper is None:
        return 0.0
    dr_lower = _dr_from_df(df_lower, ppm)
    dr_upper = _dr_from_df(df_upper, ppm)
    dr_interp = np.interp(visc, [lower, upper], [dr_lower, dr_upper])
    return float(dr_interp)


def compute_drag_reduction(visc: float, ppm: float) -> float:
    """Return effective % drag reduction for ``ppm`` at viscosity ``visc``."""
    if ppm <= 0:
        return 0.0
    return get_dr_for_ppm(visc, ppm)


__all__ = [
    "DRA_CSV_FILES",
    "DRA_CURVE_DATA",
    "load_curve",
    "get_ppm_for_dr",
    "get_dr_for_ppm",
    "compute_drag_reduction",
]
