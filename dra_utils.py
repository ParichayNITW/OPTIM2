
from __future__ import annotations

import os
from typing import Dict
import numpy as np
import pandas as pd

DRA_CSV_FILES: Dict[float, str] = {
    1: "1 cst.csv", 2: "2 cst.csv", 2.5: "2.5 cst.csv", 3: "3 cst.csv",
    5: "5 cst.csv", 7.5: "7.5 cst.csv", 10: "10 cst.csv", 15: "15 cst.csv",
    20: "20 cst.csv", 25: "25 cst.csv", 30: "30 cst.csv", 35: "35 cst.csv", 40: "40 cst.csv",
}
_CACHE: Dict[str, pd.DataFrame] = {}

def _load_curve(path: str) -> pd.DataFrame:
    if path in _CACHE:
        return _CACHE[path]
    if os.path.exists(path):
        df = pd.read_csv(path)
        cols = {c.strip().lower(): c for c in df.columns}
        dr_col = cols.get('dr (%)') or cols.get('dr') or list(df.columns)[0]
        ppm_col = cols.get('ppm') or list(df.columns)[1]
        df = df[[dr_col, ppm_col]].copy()
        df.columns = ['DR', 'PPM']
    else:
        dr = np.linspace(0, 30, 16); ppm = dr * 12.0
        df = pd.DataFrame({'DR': dr, 'PPM': ppm})
    _CACHE[path] = df
    return df

def get_ppm_for_dr(kv_cst: float, dr_percent: float) -> float:
    dr = max(0.0, float(dr_percent))
    nearest = min(DRA_CSV_FILES.keys(), key=lambda k: abs(k - float(kv_cst)))
    df = _load_curve(DRA_CSV_FILES[nearest])
    x = df['DR'].to_numpy().astype(float); y = df['PPM'].to_numpy().astype(float)
    if dr <= x.min(): 
        return float(max(0.0, np.interp(dr, [x.min(), x.min()+1e-6], [y.min(), y.min()])))
    if dr >= x.max():
        return float(max(0.0, np.interp(dr, [x.max()-1e-6, x.max()], [y.max(), y.max()])))
    return float(max(0.0, np.interp(dr, x, y)))

def get_dr_for_ppm(kv_cst: float, ppm: float) -> float:
    p = max(0.0, float(ppm))
    nearest = min(DRA_CSV_FILES.keys(), key=lambda k: abs(k - float(kv_cst)))
    df = _load_curve(DRA_CSV_FILES[nearest])
    x = df['PPM'].to_numpy().astype(float); y = df['DR'].to_numpy().astype(float)
    if p <= x.min(): 
        return float(max(0.0, np.interp(p, [x.min(), x.min()+1e-6], [y.min(), y.min()])))
    if p >= x.max():
        return float(max(0.0, np.interp(p, [x.max()-1e-6, x.max()], [y.max(), y.max()])))
    return float(max(0.0, np.interp(p, x, y)))
