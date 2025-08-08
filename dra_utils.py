
"""Drag reducer (DRA) helper utilities.

Adds inverse interpolation (ppm_to_dr) and keeps get_ppm_for_dr API.
"""

from __future__ import annotations

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


def get_ppm_for_dr(
    visc: float,
    dr: float,
    dra_curve_data: Dict[float, pd.DataFrame | None] = DRA_CURVE_DATA,
) -> float:
    """Interpolate PPM for a given drag reduction and viscosity.

    Returns the PPM value rounded to the nearest 0.5.
    """
    visc = float(visc)
    lower, upper = _nearest_bounds(visc, dra_curve_data)
    if lower not in dra_curve_data or dra_curve_data[lower] is None:
        return 0.0

    def round_ppm(val: float, step: float = 0.5) -> float:
        return round(val / step) * step

    if lower == upper:
        return round_ppm(_ppm_from_df(dra_curve_data[lower], dr))

    df_lower = dra_curve_data[lower]
    df_upper = dra_curve_data[upper]
    ppm_lower = _ppm_from_df(df_lower, dr)
    ppm_upper = _ppm_from_df(df_upper, dr)
    ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
    return round_ppm(float(ppm_interp))


def get_dr_for_ppm(
    visc: float,
    ppm: float,
    dra_curve_data: Dict[float, pd.DataFrame | None] = DRA_CURVE_DATA,
) -> float:
    """Inverse: interpolate %Drag Reduction for a given PPM and viscosity."""
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
