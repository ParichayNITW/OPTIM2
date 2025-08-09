
"""DRA utilities with inverse/forward interpolation.

CSV folder via env DRA_CURVE_DIR. Each CSV has columns: "%Drag Reduction", "PPM".
"""

from __future__ import annotations
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

DRA_CURVE_DIR = os.environ.get("DRA_CURVE_DIR", os.path.dirname(__file__))

DRA_CSV_FILES: Dict[float, str] = {
    1: "1 cst.csv", 2: "2 cst.csv", 2.5: "2.5 cst.csv", 3: "3 cst.csv",
    3.5: "3.5 cst.csv", 4: "4 cst.csv", 4.5: "4.5 cst.csv",
    5: "5 cst.csv", 6: "6 cst.csv", 8: "8 cst.csv",
    10: "10 cst.csv", 12: "12 cst.csv", 15: "15 cst.csv",
    20: "20 cst.csv", 25: "25 cst.csv", 30: "30 cst.csv",
    35: "35 cst.csv", 40: "40 cst.csv"
}

def _try_read(path:str):
    try:
        df = pd.read_csv(path)
        if "%Drag Reduction" in df.columns and "PPM" in df.columns:
            return df[["%Drag Reduction","PPM"]].dropna().sort_values("%Drag Reduction").reset_index(drop=True)
    except Exception:
        pass
    return None

DRA_CURVE_DATA: Dict[float, pd.DataFrame|None] = {}
for cst,fname in DRA_CSV_FILES.items():
    DRA_CURVE_DATA[cst] = _try_read(os.path.join(DRA_CURVE_DIR, fname))

def _nearest_bounds(visc: float):
    cands = sorted([k for k,v in DRA_CURVE_DATA.items() if v is not None])
    if not cands: return (visc,visc)
    if visc <= cands[0]: return (cands[0],cands[0])
    if visc >= cands[-1]: return (cands[-1],cands[-1])
    lo = max(c for c in cands if c<=visc)
    hi = min(c for c in cands if c>=visc)
    return lo,hi

def _ppm_from_df(df, dr):
    x = df["%Drag Reduction"].astype(float).values
    y = df["PPM"].astype(float).values
    if dr <= x[0]: return float(y[0])
    if dr >= x[-1]: return float(y[-1])
    return float(np.interp(dr, x, y))

def _dr_from_df(df, ppm):
    x = df["%Drag Reduction"].astype(float).values
    y = df["PPM"].astype(float).values
    if ppm <= y[0]: return float(x[0])
    if ppm >= y[-1]: return float(x[-1])
    return float(np.interp(ppm, y, x))

def get_ppm_for_dr(visc: float, dr: float) -> float:
    lo,hi = _nearest_bounds(float(visc))
    df_lo = DRA_CURVE_DATA.get(lo); df_hi = DRA_CURVE_DATA.get(hi)
    if df_lo is None: return 0.0
    if lo==hi: return round(_ppm_from_df(df_lo, dr)/0.5)*0.5
    p_lo = _ppm_from_df(df_lo, dr); p_hi = _ppm_from_df(df_hi, dr if df_hi is not None else dr)
    p = np.interp(float(visc), [lo,hi], [p_lo,p_hi])
    return round(p/0.5)*0.5

def get_dr_for_ppm(visc: float, ppm: float) -> float:
    lo,hi = _nearest_bounds(float(visc))
    df_lo = DRA_CURVE_DATA.get(lo); df_hi = DRA_CURVE_DATA.get(hi)
    if df_lo is None: return 0.0
    if lo==hi: return float(_dr_from_df(df_lo, ppm))
    d_lo = _dr_from_df(df_lo, ppm); d_hi = _dr_from_df(df_hi, ppm) if df_hi is not None else d_lo
    return float(np.interp(float(visc), [lo,hi], [d_lo,d_hi]))
