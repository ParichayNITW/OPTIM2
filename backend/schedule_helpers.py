"""Pure-Python linefill queue helpers — no Streamlit dependency.

All functions are faithful reimplementations of the equivalents in
pipeline_optimization_app.py but with every Streamlit reference removed.
They operate on plain pandas DataFrames / lists so they can be called
safely from FastAPI worker threads.

DO NOT import from pipeline_optimization_app here — that module has
module-level st.session_state access that crashes outside Streamlit.
"""

from __future__ import annotations

import os
import sys
from math import isclose, pi
from typing import Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Solver path — project root (desktop) or /solver (Docker)
# ---------------------------------------------------------------------------
_SOLVER_PATH = os.environ.get(
    "SOLVER_PATH",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _SOLVER_PATH not in sys.path:
    sys.path.insert(0, _SOLVER_PATH)

import pipeline_model  # noqa: E402  safe to import — no Streamlit code

INIT_DRA_COL = "Initial DRA (ppm)"

# ---------------------------------------------------------------------------
# Column name constants (must match frontend serialisation)
# ---------------------------------------------------------------------------
_COL_PRODUCT  = "Product"
_COL_VOLUME   = "Volume (m³)"
_COL_VISC     = "Viscosity (cSt)"
_COL_DENSITY  = "Density (kg/m³)"
_COL_INIT_DRA = INIT_DRA_COL

_LINEFILL_COLS = [_COL_PRODUCT, _COL_VOLUME, _COL_VISC, _COL_DENSITY, _COL_INIT_DRA]


# ---------------------------------------------------------------------------
# DataFrame construction from API payload lists
# ---------------------------------------------------------------------------

def df_from_linefill(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert a list-of-dicts linefill payload into a canonical DataFrame."""
    if not rows:
        return pd.DataFrame(columns=_LINEFILL_COLS)
    df = pd.DataFrame(rows)
    # Accept legacy keys from the frontend
    _alias(df, {"product": _COL_PRODUCT, "visc": _COL_VISC,
                "density": _COL_DENSITY, "init_dra": _COL_INIT_DRA,
                "volume": _COL_VOLUME})
    for col in _LINEFILL_COLS:
        if col not in df.columns:
            df[col] = 0.0 if col != _COL_PRODUCT else ""
    return df[_LINEFILL_COLS].copy().reset_index(drop=True)


def _alias(df: pd.DataFrame, mapping: dict[str, str]) -> None:
    """Rename columns in-place if the target name is not already present."""
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)


# ---------------------------------------------------------------------------
# ensure_initial_dra_column  (mirror of app version, no Streamlit)
# ---------------------------------------------------------------------------

def ensure_initial_dra_column(
    df: pd.DataFrame | None,
    *,
    default: float | None = 0.0,
    fill_blanks: bool = False,
) -> pd.DataFrame | None:
    if not isinstance(df, pd.DataFrame):
        return df

    if INIT_DRA_COL not in df.columns:
        df[INIT_DRA_COL] = default if default is not None else np.nan
        return df

    if not fill_blanks or default is None:
        return df

    col = df[INIT_DRA_COL]
    blank_mask = col.isna()
    if col.dtype == object:
        blank_mask |= col.astype(str).str.strip() == ""
    if blank_mask.any():
        df.loc[blank_mask, INIT_DRA_COL] = default
    return df


# ---------------------------------------------------------------------------
# _station_d_inner
# ---------------------------------------------------------------------------

def _station_d_inner(stn: dict) -> float:
    """Return internal pipe diameter (m) for a station config dict."""
    if stn.get("D") is not None:
        t = float(stn.get("t", 0.007) or 0.007)
        return max(float(stn.get("D") or 0.0) - 2.0 * t, 0.0)
    d = stn.get("d_inner") or stn.get("d", 0.711)
    return float(d or 0.711)


# ---------------------------------------------------------------------------
# _default_segment_slices
# ---------------------------------------------------------------------------

def _default_segment_slices(
    stations: list[dict], kv_list: list[float], rho_list: list[float]
) -> list[list[dict]]:
    if not stations:
        return []
    fallback_kv  = kv_list[0]  if kv_list  else 1.0
    fallback_rho = rho_list[0] if rho_list else 850.0
    slices: list[list[dict]] = []
    for idx, stn in enumerate(stations):
        length = float(stn.get("L", 0.0) or 0.0)
        kv  = kv_list[idx]  if idx < len(kv_list)  else fallback_kv
        rho = rho_list[idx] if idx < len(rho_list) else fallback_rho
        slices.append([{"length_km": length, "kv": float(kv), "rho": float(rho)}])
    return slices


# ---------------------------------------------------------------------------
# _compress_slice_entries
# ---------------------------------------------------------------------------

def _compress_slice_entries(entries: list[dict], tol: float = 1e-9) -> list[dict]:
    if not entries:
        return []
    compressed: list[dict] = [dict(entries[0])]
    for entry in entries[1:]:
        if (
            isclose(entry.get("kv", 0.0), compressed[-1].get("kv", 0.0), rel_tol=0.0, abs_tol=1e-9)
            and isclose(entry.get("rho", 0.0), compressed[-1].get("rho", 0.0), rel_tol=0.0, abs_tol=1e-9)
        ):
            compressed[-1]["length_km"] += entry.get("length_km", 0.0)
        else:
            compressed.append(dict(entry))
    return [e for e in compressed if e.get("length_km", 0.0) > tol]


# ---------------------------------------------------------------------------
# _weighted_properties
# ---------------------------------------------------------------------------

def _weighted_properties(
    entries: list[dict] | None,
    seg_length: float,
    default_kv: float,
    default_rho: float,
) -> tuple[float, float]:
    if seg_length <= 0.0:
        return 0.0, 0.0
    total_len = visc_sum = rho_sum = 0.0
    for entry in entries or []:
        try:
            seg_len = float(entry.get("length_km", 0.0) or 0.0)
        except (TypeError, ValueError):
            seg_len = 0.0
        if seg_len <= 0.0:
            continue
        kv_val  = float(entry.get("kv",  default_kv)  or default_kv)
        rho_val = float(entry.get("rho", default_rho) or default_rho)
        total_len += seg_len
        visc_sum  += seg_len * kv_val
        rho_sum   += seg_len * rho_val
    if total_len <= 0.0:
        return float(default_kv), float(default_rho)
    return visc_sum / total_len, rho_sum / total_len


# ---------------------------------------------------------------------------
# map_vol_linefill_to_segments
# ---------------------------------------------------------------------------

def map_vol_linefill_to_segments(
    vol_table: pd.DataFrame | None, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Convert a volumetric linefill DataFrame to per-segment fluid properties."""

    if not isinstance(vol_table, pd.DataFrame) or vol_table.empty:
        kv_list  = [1.0]   * len(stations)
        rho_list = [850.0] * len(stations)
        return kv_list, rho_list, _default_segment_slices(stations, kv_list, rho_list)

    if not stations:
        return [], [], []

    batches: list[dict[str, float]] = []
    col_vol = _COL_VOLUME if _COL_VOLUME in vol_table.columns else "Volume"
    for _, r in vol_table.iterrows():
        try:
            vol = float(r.get(col_vol, 0.0) or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol <= 0.0:
            continue
        try:
            visc = float(r.get(_COL_VISC, 0.0) or 0.0)
        except (TypeError, ValueError):
            visc = 0.0
        try:
            dens = float(r.get(_COL_DENSITY, 0.0) or 0.0)
        except (TypeError, ValueError):
            dens = 0.0
        batches.append({"volume_m3": vol, "kv": visc, "rho": dens})

    if not batches:
        kv_list  = [1.0]   * len(stations)
        rho_list = [850.0] * len(stations)
        return kv_list, rho_list, _default_segment_slices(stations, kv_list, rho_list)

    km_from_volume  = pipeline_model._km_from_volume
    volume_from_km  = pipeline_model._volume_from_km

    seg_kv: list[float]        = []
    seg_rho: list[float]       = []
    seg_slices: list[list[dict]] = []
    seg_lengths = [float(s.get("L", 0.0) or 0.0) for s in stations]

    i_batch = 0
    remaining_vol = batches[0]["volume_m3"]
    kv_cur  = batches[0]["kv"]
    rho_cur = batches[0]["rho"]

    for stn, L in zip(stations, seg_lengths):
        d_seg = _station_d_inner(stn)
        if d_seg <= 0:
            seg_kv.append(kv_cur)
            seg_rho.append(rho_cur)
            seg_slices.append([])
            continue

        seg_vol  = volume_from_km(L, d_seg)
        need_vol = seg_vol

        segment_entries: list[dict] = []
        while need_vol > 1e-9:
            if remaining_vol <= 1e-9:
                i_batch += 1
                if i_batch >= len(batches):
                    length_km = km_from_volume(need_vol, d_seg)
                    segment_entries.append({"length_km": length_km, "kv": kv_cur, "rho": rho_cur})
                    need_vol = 0.0
                    break
                remaining_vol = batches[i_batch]["volume_m3"]
                kv_cur  = batches[i_batch]["kv"]
                rho_cur = batches[i_batch]["rho"]
                if remaining_vol <= 1e-9:
                    continue

            take_vol  = min(need_vol, remaining_vol)
            take_km   = km_from_volume(take_vol, d_seg)
            if take_km > 0:
                segment_entries.append({"length_km": take_km, "kv": kv_cur, "rho": rho_cur})
            need_vol      -= take_vol
            remaining_vol -= take_vol

        if not segment_entries:
            segment_entries.append({"length_km": L, "kv": kv_cur, "rho": rho_cur})

        seg_slices.append(segment_entries)
        seg_kv.append(segment_entries[0]["kv"])
        avg_rho = (
            sum(e["length_km"] * e["rho"] for e in segment_entries) / L
            if L > 0 else segment_entries[0]["rho"]
        )
        seg_rho.append(avg_rho)

    return seg_kv, seg_rho, seg_slices


# ---------------------------------------------------------------------------
# combine_volumetric_profiles
# ---------------------------------------------------------------------------

def combine_volumetric_profiles(
    stations: list[dict],
    current_vol: pd.DataFrame,
    future_vol: pd.DataFrame,
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Return per-segment profiles using the higher weighted-viscosity hour."""

    kv_now,  rho_now,  slices_now  = map_vol_linefill_to_segments(current_vol, stations)
    kv_next, rho_next, slices_next = map_vol_linefill_to_segments(future_vol,  stations)

    kv_list: list[float]        = []
    rho_list: list[float]       = []
    segment_slices: list[list[dict]] = []

    for idx in range(len(stations)):
        kv_cur    = kv_now[idx]  if idx < len(kv_now)   else (kv_now[-1]  if kv_now  else 1.0)
        kv_future = kv_next[idx] if idx < len(kv_next)  else (kv_next[-1] if kv_next else kv_cur)
        rho_cur    = rho_now[idx]  if idx < len(rho_now)  else (rho_now[-1]  if rho_now  else 850.0)
        rho_future = rho_next[idx] if idx < len(rho_next) else (rho_next[-1] if rho_next else rho_cur)

        seg_length   = float(stations[idx].get("L", 0.0) or 0.0)
        entries_now  = slices_now[idx]  if idx < len(slices_now)  else []
        entries_next = slices_next[idx] if idx < len(slices_next) else []

        kv_w_now,  rho_w_now  = _weighted_properties(entries_now,  seg_length, kv_cur,    rho_cur)
        kv_w_next, rho_w_next = _weighted_properties(entries_next, seg_length, kv_future, rho_future)

        if kv_w_next > kv_w_now:
            kv_choice = kv_w_next;  rho_choice = rho_w_next;  chosen = entries_next
        else:
            kv_choice = kv_w_now;   rho_choice = rho_w_now;   chosen = entries_now

        segment_slices.append(_compress_slice_entries(chosen))
        kv_list.append(kv_choice)
        rho_list.append(rho_choice)

    return kv_list, rho_list, segment_slices


# ---------------------------------------------------------------------------
# apply_dra_ppm
# ---------------------------------------------------------------------------

def apply_dra_ppm(df: pd.DataFrame, dra_batches: list[dict]) -> pd.DataFrame:
    """Assign DRA ppm from dra_batches onto df rows by volume."""

    if df is None:
        return df

    df = ensure_initial_dra_column(df.copy(), default=0.0, fill_blanks=True)
    if "DRA ppm" not in df.columns:
        df["DRA ppm"] = df[INIT_DRA_COL]

    col_vol = _COL_VOLUME if _COL_VOLUME in df.columns else "Volume"

    queue: list[tuple[float, float]] = []
    for batch in dra_batches or []:
        try:
            vol = float(batch.get("volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol <= 1e-9:
            continue
        try:
            ppm_val = float(batch.get("dra_ppm", 0.0) or 0.0)
        except (TypeError, ValueError):
            ppm_val = 0.0
        if np.isnan(ppm_val):
            ppm_val = 0.0
        queue.append((vol, ppm_val))

    queue_idx = 0
    if queue:
        seg_remaining = queue[0][0];  ppm_cur = queue[0][1]
    else:
        seg_remaining = float("inf"); ppm_cur = 0.0

    def advance() -> None:
        nonlocal queue_idx, seg_remaining, ppm_cur
        queue_idx += 1
        if queue_idx < len(queue):
            seg_remaining = queue[queue_idx][0]; ppm_cur = queue[queue_idx][1]
        else:
            seg_remaining = float("inf"); ppm_cur = 0.0

    rows_out: list[dict] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            row_vol = float(row_dict.get(col_vol, 0.0) or 0.0)
        except (TypeError, ValueError):
            row_vol = 0.0

        if row_vol <= 1e-9:
            row_dict["DRA ppm"] = ppm_cur
            row_dict[INIT_DRA_COL] = ppm_cur
            rows_out.append(row_dict)
            continue

        remaining_row = row_vol
        while remaining_row > 1e-9:
            while queue_idx < len(queue) and seg_remaining <= 1e-9:
                advance()
            take = min(remaining_row, seg_remaining)
            if take <= 1e-9:
                take = remaining_row
            row_copy = row_dict.copy()
            row_copy[col_vol] = take
            row_copy["DRA ppm"] = ppm_cur
            row_copy[INIT_DRA_COL] = ppm_cur
            rows_out.append(row_copy)
            remaining_row -= take
            if queue_idx < len(queue):
                seg_remaining -= take
                if seg_remaining <= 1e-9:
                    advance()

    if not rows_out:
        return df.iloc[0:0].copy()
    return pd.DataFrame(rows_out, columns=df.columns).reset_index(drop=True)


# ---------------------------------------------------------------------------
# shift_vol_linefill
# ---------------------------------------------------------------------------

def shift_vol_linefill(
    vol_table: pd.DataFrame,
    pumped_m3: float,
    day_plan: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[dict[str, float]]]:
    """Remove pumped_m3 from terminal end, inject from day_plan at origin."""

    vol_table = ensure_initial_dra_column(vol_table.copy(), default=0.0, fill_blanks=True)
    vol_table[_COL_VOLUME] = vol_table[_COL_VOLUME].astype(float)

    remaining = pumped_m3
    idx = len(vol_table) - 1
    while remaining > 1e-9 and idx >= 0:
        v    = vol_table.at[idx, _COL_VOLUME]
        take = min(v, remaining)
        vol_table.at[idx, _COL_VOLUME] = v - take
        remaining -= take
        if vol_table.at[idx, _COL_VOLUME] <= 1e-9:
            vol_table = vol_table.drop(index=idx)
        idx -= 1
    vol_table = vol_table.reset_index(drop=True)

    injected_batches: list[dict[str, float]] = []

    if day_plan is not None:
        day_plan = ensure_initial_dra_column(day_plan.copy(), default=0.0, fill_blanks=True)
        day_plan[_COL_VOLUME] = day_plan[_COL_VOLUME].astype(float)

        inject_remaining = pumped_m3
        while inject_remaining > 1e-9 and len(day_plan) > 0:
            v        = day_plan.iloc[0][_COL_VOLUME]
            take     = min(v, inject_remaining)
            dra_ppm  = float(day_plan.iloc[0].get(INIT_DRA_COL, 0.0))

            new_row = pd.DataFrame([{
                _COL_PRODUCT:  day_plan.iloc[0].get(_COL_PRODUCT,  ""),
                _COL_VOLUME:   take,
                _COL_VISC:     day_plan.iloc[0].get(_COL_VISC,     0.0),
                _COL_DENSITY:  day_plan.iloc[0].get(_COL_DENSITY,  850.0),
                INIT_DRA_COL:  dra_ppm,
            }])
            vol_table = pd.concat([new_row, vol_table], ignore_index=True)
            injected_batches.insert(0, {"volume": float(take), "dra_ppm": dra_ppm})

            inject_remaining -= take
            if take >= v:
                day_plan = day_plan.drop(index=0).reset_index(drop=True)
            else:
                day_plan.at[0, _COL_VOLUME] = v - take

    return vol_table.reset_index(drop=True), day_plan, injected_batches
