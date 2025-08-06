import os
import numpy as np
import pandas as pd

# Mapping of standard DRA curve CSV files
DRA_CSV_FILES = {
    10: "10 cst.csv",
    15: "15 cst.csv",
    20: "20 cst.csv",
    25: "25 cst.csv",
    30: "30 cst.csv",
    35: "35 cst.csv",
    40: "40 cst.csv",
}

# Load any available curves.  Missing files are simply ignored so the
# optimiser can run in environments where the data are not present.
DRA_CURVE_DATA = {
    cst: pd.read_csv(fname)
    for cst, fname in DRA_CSV_FILES.items()
    if os.path.exists(fname)
}

def _ppm_from_df(df: pd.DataFrame, dr: float) -> float:
    """Return PPM for ``dr`` using the breakpoints in ``df``.

    If the dataframe is ``None`` or empty, ``0`` is returned so callers do not
    have to guard against missing DRA curve data.
    """
    if df is None or df.empty:
        return 0.0
    x = df['%Drag Reduction'].values
    y = df['PPM'].values
    if dr <= x[0]:
        return float(y[0])
    if dr >= x[-1]:
        return float(y[-1])
    return float(np.interp(dr, x, y))

def get_ppm_for_dr(visc, dr, dra_curve_data=DRA_CURVE_DATA):
    """Interpolate PPM for a given drag reduction and viscosity.

    The previous implementation assumed DRA curve data were always loaded and
    that viscosity values were finite.  When either assumption was violated the
    application crashed with ``ValueError: max() arg is an empty sequence``.  We
    now guard against missing data and ``NaN`` viscosities, returning ``0`` PPM
    which effectively disables DRA for that segment.
    """
    try:
        visc = float(visc)
    except (TypeError, ValueError):
        visc = float('nan')
    if not dra_curve_data or np.isnan(visc) or dr <= 0:
        return 0.0
    cst_list = sorted(dra_curve_data.keys())
    # Ensure we have actual dataframes for interpolation
    valid_lower = [c for c in cst_list if c <= visc and dra_curve_data.get(c) is not None]
    valid_upper = [c for c in cst_list if c >= visc and dra_curve_data.get(c) is not None]
    if not valid_lower or not valid_upper:
        # No surrounding curves: fall back to nearest available or 0 if none
        if not dra_curve_data:
            return 0.0
        nearest = cst_list[0]
        return _ppm_from_df(dra_curve_data.get(nearest), dr)
    lower = max(valid_lower)
    upper = min(valid_upper)
    if lower == upper:
        return _ppm_from_df(dra_curve_data[lower], dr)
    ppm_lower = _ppm_from_df(dra_curve_data[lower], dr)
    ppm_upper = _ppm_from_df(dra_curve_data[upper], dr)
    ppm_interp = np.interp(visc, [lower, upper], [ppm_lower, ppm_upper])
    # Round to nearest 0.5 ppm
    return float(round(ppm_interp / 0.5) * 0.5)

def map_linefill_to_segments(linefill_df: pd.DataFrame, stations):
    """Map linefill properties onto each pipeline segment.

    When the linefill table is empty the original helper attempted to index the
    dataframe and crashed with ``IndexError``.  This version defaults to
    viscosity ``1.0`` cSt and density ``800`` kg/m³, ensuring the optimiser can
    still proceed.
    """
    cumlen = [0]
    for stn in stations:
        cumlen.append(cumlen[-1] + stn.get("L", 0.0))
    if linefill_df is None or linefill_df.empty:
        default_visc = 1.0
        default_den = 800.0
        return [default_visc] * len(stations), [default_den] * len(stations)
    viscs, dens = [], []
    last_visc = linefill_df.iloc[-1]["Viscosity (cSt)"]
    last_den = linefill_df.iloc[-1]["Density (kg/m³)"]
    for i in range(len(stations)):
        seg_start = cumlen[i]
        mask = (
            (linefill_df["Start (km)"] <= seg_start) &
            (seg_start < linefill_df["End (km)"])
        )
        if mask.any():
            row = linefill_df[mask].iloc[0]
            viscs.append(row["Viscosity (cSt)"])
            dens.append(row["Density (kg/m³)"])
        else:
            viscs.append(last_visc)
            dens.append(last_den)
    return viscs, dens
