"""Drag reducer (DRA) helper utilities.

This module centralises loading of DRA performance curves and exposes
helpers for interpolating PPM values.  It is lightweight and avoids
importing the heavy optimisation stack so the Streamlit app can use it
without long import times.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

# Mapping of viscosity (cSt) to CSV file name
DRA_CSV_FILES: Dict[int, str] = {
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv",
}

# Load the drag-reducer curves lazily at import time
DRA_CURVE_DATA: Dict[int, pd.DataFrame | None] = {}
for cst, fname in DRA_CSV_FILES.items():
    if os.path.exists(fname):
        DRA_CURVE_DATA[cst] = pd.read_csv(fname)
    else:
        DRA_CURVE_DATA[cst] = None


def _ppm_from_df(df: pd.DataFrame, dr: float) -> float:
    """Return the PPM value for ``dr`` using breakpoints in ``df``."""

    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return float(y[0])
    if dr >= x[-1]:
        return float(y[-1])
    return float(np.interp(dr, x, y))


def get_ppm_for_dr(
    visc: float,
    dr: float,
    dra_curve_data: Dict[int, pd.DataFrame | None] = DRA_CURVE_DATA,
) -> float:
    """Interpolate PPM for a given drag reduction and viscosity.

    Parameters
    ----------
    visc: float
        Fluid viscosity in cSt.
    dr: float
        Required drag reduction percentage.
    dra_curve_data: dict
        Mapping of viscosity to DRA curve data. Defaults to the module level
        data loaded from CSV files.

    Returns
    -------
    float
        Interpolated PPM value rounded to the nearest 0.5.
    """

    cst_list = sorted([c for c in dra_curve_data.keys() if dra_curve_data[c] is not None])
    if not cst_list:
        return 0.0

    visc = float(visc)

    def round_ppm(val: float, step: float = 0.5) -> float:
        return round(val / step) * step

    if visc <= cst_list[0]:
        df = dra_curve_data[cst_list[0]]
        return round_ppm(_ppm_from_df(df, dr))
    if visc >= cst_list[-1]:
        df = dra_curve_data[cst_list[-1]]
        return round_ppm(_ppm_from_df(df, dr))

    lower = max(c for c in cst_list if c <= visc)
    upper = min(c for c in cst_list if c >= visc)
    df_lower = dra_curve_data[lower]
    df_upper = dra_curve_data[upper]
    ppm_lower = _ppm_from_df(df_lower, dr)
    ppm_upper = _ppm_from_df(df_upper, dr)
    ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
    return round_ppm(ppm_interp)
