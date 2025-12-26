"""Utilities for running the optimizer outside the Streamlit UI."""

from __future__ import annotations

import json
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

import pipeline_model


DRA_COLUMNS = (
    "Initial DRA (ppm)",
    "Initial DRA ppm",
    "DRA ppm",
    "dra_ppm",
    "initial_dra_ppm",
)


def ensure_initial_dra_column(
    df: pd.DataFrame | None,
    *,
    default: float | None = 0.0,
    fill_blanks: bool = False,
) -> pd.DataFrame | None:
    """Ensure ``df`` exposes the user-editable DRA ppm column."""

    if not isinstance(df, pd.DataFrame):
        return df

    col = "Initial DRA (ppm)"
    if col not in df.columns:
        df[col] = default if default is not None else 0.0
        return df

    if fill_blanks:
        mask = df[col].isna()
        if mask.any():
            df.loc[mask, col] = default if default is not None else 0.0
    return df


def _default_segment_slices(
    stations: list[dict], kv_list: list[float], rho_list: list[float]
) -> list[list[dict]]:
    """Return single-slice segment profiles for legacy distance tables."""

    if not stations:
        return []

    fallback_kv = kv_list[0] if kv_list else 1.0
    fallback_rho = rho_list[0] if rho_list else 850.0
    slices: list[list[dict]] = []
    for idx, stn in enumerate(stations):
        length = float(stn.get("L", 0.0) or 0.0)
        kv = kv_list[idx] if idx < len(kv_list) else fallback_kv
        rho = rho_list[idx] if idx < len(rho_list) else fallback_rho
        slices.append(
            [
                {
                    "length_km": length,
                    "kv": float(kv),
                    "rho": float(rho),
                }
            ]
        )
    return slices


def derive_segment_profiles(
    linefill_df: pd.DataFrame | None, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Return per-segment viscosity/density lists and batch slices."""

    kv_list, rho_list, segment_slices = map_linefill_to_segments(linefill_df, stations)
    return kv_list, rho_list, segment_slices


def map_linefill_to_segments(
    linefill_df: pd.DataFrame | None, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Map linefill properties onto each pipeline segment."""

    if linefill_df is None or len(linefill_df) == 0:
        kv_list = [1.0] * len(stations)
        rho_list = [850.0] * len(stations)
        segment_slices = _default_segment_slices(stations, kv_list, rho_list)
        return kv_list, rho_list, segment_slices

    cols = set(linefill_df.columns)

    if "Start (km)" not in cols or "End (km)" not in cols:
        if "Volume (m³)" in cols or "Volume" in cols:
            return map_vol_linefill_to_segments(linefill_df, stations)
        kv = float(linefill_df.iloc[-1].get("Viscosity (cSt)", 0.0))
        rho = float(linefill_df.iloc[-1].get("Density (kg/m³)", 0.0))
        kv_list = [kv] * len(stations)
        rho_list = [rho] * len(stations)
        segment_slices = _default_segment_slices(stations, kv_list, rho_list)
        return kv_list, rho_list, segment_slices

    cumlen = [0.0]
    for stn in stations:
        cumlen.append(cumlen[-1] + float(stn.get("L", 0.0) or 0.0))
    viscs = []
    dens = []
    for i in range(len(stations)):
        seg_start = cumlen[i]
        seg_end = cumlen[i + 1]
        found = False
        for _, row in linefill_df.iterrows():
            if row["Start (km)"] <= seg_start < row["End (km)"]:
                viscs.append(row["Viscosity (cSt)"])
                dens.append(row["Density (kg/m³)"])
                found = True
                break
        if not found:
            viscs.append(linefill_df.iloc[-1]["Viscosity (cSt)"])
            dens.append(linefill_df.iloc[-1]["Density (kg/m³)"])
    segment_slices = _default_segment_slices(stations, viscs, dens)
    return viscs, dens, segment_slices


def pipe_cross_section_area_m2(stations: list[dict]) -> float:
    """Return pipe internal cross-sectional area (m²) using the first station."""

    if not stations:
        return 0.0
    first = stations[0]
    diameter = first.get("D")
    if diameter is None:
        diameter = first.get("d", 0.711)
    t = float(first.get("t", 0.007))
    d_inner = max(float(diameter) - 2.0 * t, 0.0)
    return float((math.pi * d_inner**2) / 4.0)


def map_vol_linefill_to_segments(
    vol_table: pd.DataFrame | None, stations: list[dict]
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Convert a volumetric linefill table to per-segment fluid properties."""

    if not isinstance(vol_table, pd.DataFrame):
        return derive_segment_profiles(pd.DataFrame(), stations)

    if not stations:
        return [], [], []

    batches: list[dict[str, float]] = []
    for _, r in vol_table.iterrows():
        try:
            vol_raw = r.get("Volume (m³)")
        except AttributeError:
            vol_raw = None
        if vol_raw in (None, ""):
            vol_raw = r.get("Volume")
        try:
            vol = float(vol_raw)
        except (TypeError, ValueError):
            vol = 0.0
        if vol <= 0.0:
            continue

        try:
            visc = float(r.get("Viscosity (cSt)", 0.0))
        except (TypeError, ValueError):
            visc = 0.0
        try:
            dens = float(r.get("Density (kg/m³)", 0.0))
        except (TypeError, ValueError):
            dens = 0.0

        batches.append({"volume_m3": vol, "kv": visc, "rho": dens})

    if not batches:
        fallback_kv = 1.0
        fallback_rho = 850.0
        kv_list = [fallback_kv] * len(stations)
        rho_list = [fallback_rho] * len(stations)
        return kv_list, rho_list, _default_segment_slices(stations, kv_list, rho_list)

    area = pipe_cross_section_area_m2(stations)
    if area <= 0:
        fallback_kv = batches[0]["kv"] if batches else 1.0
        fallback_rho = batches[0]["rho"] if batches else 850.0
        kv_list = [fallback_kv] * len(stations)
        rho_list = [fallback_rho] * len(stations)
        return kv_list, rho_list, _default_segment_slices(stations, kv_list, rho_list)

    d_inner = math.sqrt((4.0 * area) / math.pi)
    km_from_volume = pipeline_model._km_from_volume

    for entry in batches:
        entry["len_km"] = km_from_volume(entry["volume_m3"], d_inner)

    seg_kv: list[float] = []
    seg_rho: list[float] = []
    seg_slices: list[list[dict]] = []
    seg_lengths = [float(s.get("L", 0.0) or 0.0) for s in stations]

    i_batch = 0
    remaining = batches[0]["len_km"] if batches else 0.0
    kv_cur = batches[0]["kv"] if batches else 1.0
    rho_cur = batches[0]["rho"] if batches else 850.0

    for length in seg_lengths:
        need = length
        if length <= 0:
            seg_kv.append(kv_cur)
            seg_rho.append(rho_cur)
            seg_slices.append([])
            continue

        segment_entries: list[dict] = []
        while need > 1e-9:
            if remaining <= 1e-9:
                i_batch += 1
                if i_batch >= len(batches):
                    segment_entries.append(
                        {"length_km": need, "kv": kv_cur, "rho": rho_cur}
                    )
                    need = 0.0
                    break
                remaining = batches[i_batch]["len_km"]
                kv_cur = batches[i_batch]["kv"]
                rho_cur = batches[i_batch]["rho"]
                if remaining <= 1e-9:
                    continue

            take = min(need, remaining)
            if take <= 0:
                break

            segment_entries.append({"length_km": take, "kv": kv_cur, "rho": rho_cur})
            need -= take
            remaining -= take

        if not segment_entries:
            segment_entries.append({"length_km": length, "kv": kv_cur, "rho": rho_cur})

        seg_kv.append(segment_entries[0]["kv"])
        seg_rho.append(segment_entries[0]["rho"])
        seg_slices.append(segment_entries)

    return seg_kv, seg_rho, seg_slices


def _extract_dra_ppm(row: Mapping[str, object]) -> float:
    for key in DRA_COLUMNS:
        if key in row:
            try:
                return float(row.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _linefill_batches_from_df(
    linefill_df: pd.DataFrame | None, stations: list[dict]
) -> list[dict]:
    if not isinstance(linefill_df, pd.DataFrame) or linefill_df.empty:
        return []

    cols = set(linefill_df.columns)
    batches: list[dict] = []
    if "Volume (m³)" in cols or "Volume" in cols:
        for _, row in linefill_df.iterrows():
            raw = row.get("Volume (m³)")
            if raw in (None, ""):
                raw = row.get("Volume")
            try:
                vol = float(raw)
            except (TypeError, ValueError):
                vol = 0.0
            if vol <= 0:
                continue
            batches.append(
                {
                    "volume": vol,
                    "dra_ppm": _extract_dra_ppm(row),
                }
            )
        return batches

    if "Start (km)" in cols and "End (km)" in cols:
        area = pipe_cross_section_area_m2(stations)
        if area <= 0:
            return []
        for _, row in linefill_df.iterrows():
            try:
                start_km = float(row.get("Start (km)", 0.0) or 0.0)
                end_km = float(row.get("End (km)", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            length_km = max(end_km - start_km, 0.0)
            if length_km <= 0:
                continue
            volume = length_km * 1000.0 * area
            batches.append(
                {
                    "volume": volume,
                    "dra_ppm": _extract_dra_ppm(row),
                }
            )
        return batches

    return []


def _update_station_defaults(stations: list[dict]) -> list[dict]:
    for idx, stn in enumerate(stations, start=1):
        stn.setdefault("name", f"Station {idx}")
        stn.setdefault("max_dr", 0.0)
        stn.setdefault("min_residual", 0.0)
        stn.setdefault("loopline", False)
        stn.setdefault("max_pumps", stn.get("available", 0))
        stn.setdefault("min_pumps", 0)
        if "D" not in stn and "d" in stn:
            stn["D"] = stn.get("d")

        if not stn.get("is_pump", False):
            continue

        pump_types = stn.get("pump_types") if isinstance(stn.get("pump_types"), Mapping) else None
        if pump_types:
            for ptype, pdata in pump_types.items():
                if not isinstance(pdata, Mapping):
                    continue
                pdata = dict(pdata)
                pdata.setdefault("available", pdata.get("count", 0))
                dfh = pdata.get("head_data")
                dfe = pdata.get("eff_data")
                if dfh is not None:
                    dfh = pd.DataFrame(dfh)
                if dfe is not None:
                    dfe = pd.DataFrame(dfe)
                if dfh is not None and len(dfh) >= 3:
                    qh = dfh.iloc[:, 0].values
                    hh = dfh.iloc[:, 1].values
                    coeff = np.polyfit(qh, hh, 2)
                    pdata["A"], pdata["B"], pdata["C"] = (
                        float(coeff[0]),
                        float(coeff[1]),
                        float(coeff[2]),
                    )
                if dfe is not None and len(dfe) >= 5:
                    qe = dfe.iloc[:, 0].values
                    ee = dfe.iloc[:, 1].values
                    coeff_e = np.polyfit(qe, ee, 4)
                    pdata["P"], pdata["Q"], pdata["R"], pdata["S"], pdata["T"] = (
                        float(coeff_e[0]),
                        float(coeff_e[1]),
                        float(coeff_e[2]),
                        float(coeff_e[3]),
                        float(coeff_e[4]),
                    )
                pdata["head_data"] = dfh.to_dict(orient="records") if isinstance(dfh, pd.DataFrame) else None
                pdata["eff_data"] = dfe.to_dict(orient="records") if isinstance(dfe, pd.DataFrame) else None
                stn.setdefault("pump_types", {})[ptype] = pdata
        else:
            dfh = stn.get("head_data")
            dfe = stn.get("eff_data")
            if dfh is not None:
                dfh = pd.DataFrame(dfh)
            if dfe is not None:
                dfe = pd.DataFrame(dfe)
            if dfh is not None and len(dfh) >= 3:
                qh = dfh.iloc[:, 0].values
                hh = dfh.iloc[:, 1].values
                coeff = np.polyfit(qh, hh, 2)
                stn["A"], stn["B"], stn["C"] = (
                    float(coeff[0]),
                    float(coeff[1]),
                    float(coeff[2]),
                )
            if dfe is not None and len(dfe) >= 5:
                qe = dfe.iloc[:, 0].values
                ee = dfe.iloc[:, 1].values
                coeff_e = np.polyfit(qe, ee, 4)
                stn["P"], stn["Q"], stn["R"], stn["S"], stn["T"] = (
                    float(coeff_e[0]),
                    float(coeff_e[1]),
                    float(coeff_e[2]),
                    float(coeff_e[3]),
                    float(coeff_e[4]),
                )
            if isinstance(dfh, pd.DataFrame):
                stn["head_data"] = dfh.to_dict(orient="records")
            if isinstance(dfe, pd.DataFrame):
                stn["eff_data"] = dfe.to_dict(orient="records")
    return stations


def prepare_case(case_data: Mapping[str, object]) -> dict:
    """Return normalized case inputs for the solver."""

    stations = [dict(stn) for stn in case_data.get("stations", [])]
    _update_station_defaults(stations)

    terminal = case_data.get("terminal", {}) or {}
    term_data = {
        "name": terminal.get("name", "Terminal"),
        "elev": float(terminal.get("elev", 0.0) or 0.0),
        "min_residual": float(terminal.get("min_residual", 50.0) or 0.0),
    }

    linefill_df = pd.DataFrame(case_data.get("linefill", []))
    linefill_vol = pd.DataFrame(case_data.get("linefill_vol", []))

    if isinstance(linefill_vol, pd.DataFrame) and not linefill_vol.empty:
        linefill_vol = ensure_initial_dra_column(linefill_vol, default=0.0, fill_blanks=True)
        try:
            kv_list, rho_list, segment_slices = map_vol_linefill_to_segments(linefill_vol, stations)
            linefill_df = linefill_vol
        except Exception:
            linefill_df = ensure_initial_dra_column(linefill_df, default=0.0, fill_blanks=True)
            kv_list, rho_list, segment_slices = derive_segment_profiles(linefill_df, stations)
    else:
        linefill_df = ensure_initial_dra_column(linefill_df, default=0.0, fill_blanks=True)
        kv_list, rho_list, segment_slices = derive_segment_profiles(linefill_df, stations)

    linefill_batches = _linefill_batches_from_df(linefill_df, stations)

    return {
        "stations": stations,
        "terminal": term_data,
        "linefill_df": linefill_df,
        "kv_list": kv_list,
        "rho_list": rho_list,
        "segment_slices": segment_slices,
        "linefill_batches": linefill_batches,
    }


def dump_json(data: object, path: str | None = None, *, pretty: bool = False) -> str:
    """Serialize results to JSON (optionally writing to ``path``)."""

    def _fallback(value: object):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, set):
            return list(value)
        return str(value)

    kwargs = {"default": _fallback}
    if pretty:
        kwargs.update({"indent": 2})
    payload = json.dumps(data, **kwargs)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    return payload


def merge_segment_floors(case_data: Mapping[str, object]) -> list[dict] | None:
    """Return enforceable segment floors from a saved case, if any."""

    floors = case_data.get("origin_lacing_segment_baseline")
    if isinstance(floors, Iterable) and not isinstance(floors, (str, bytes)):
        floors_list = [dict(entry) for entry in floors if isinstance(entry, Mapping)]
        return floors_list or None
    return None
