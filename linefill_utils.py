
from __future__ import annotations

from math import pi
from typing import List, Dict
import pandas as pd

def linefill_lengths(linefill: List[Dict], diameter: float) -> List[Dict]:
    A = (pi * diameter**2) / 4.0
    out = []; acc = 0.0
    for batch in linefill:
        vol = float(batch.get("volume", 0.0))
        L_km = (vol / max(A, 1e-9)) / 1000.0
        acc += L_km
        b = dict(batch); b['length_km'] = L_km; b['cum_length_km'] = acc
        out.append(b)
    return out

def shift_vol_linefill(vol_table: pd.DataFrame, pumped_m3: float, day_plan: pd.DataFrame | None):
    vol_table = vol_table.copy()
    vol_table["Volume (m³)"] = vol_table["Volume (m³)"].astype(float)
    take = pumped_m3
    i = len(vol_table) - 1
    while i >= 0 and take > 0:
        have = float(vol_table.at[i, "Volume (m³)"])
        rem = max(have - take, 0.0)
        take -= (have - rem)
        vol_table.at[i, "Volume (m³)"] = rem
        if rem <= 1e-9: vol_table = vol_table.drop(index=vol_table.index[i])
        i -= 1
    vol_table.reset_index(drop=True, inplace=True)

    if day_plan is not None:
        day_plan = day_plan.copy()
        day_plan["Volume (m³)"] = day_plan["Volume (m³)"].astype(float)
        add = pumped_m3
        while add > 0 and len(day_plan) > 0:
            head = day_plan.iloc[0].to_dict()
            vol = float(head.get("Volume (m³)", 0.0))
            take = min(vol, add)
            new_b = dict(head); new_b["Volume (m³)"] = take
            vol_table = pd.concat([pd.DataFrame([new_b]), vol_table], ignore_index=True)
            day_plan.at[0, "Volume (m³)"] = vol - take
            if day_plan.at[0, "Volume (m³)"] <= 1e-9:
                day_plan = day_plan.iloc[1:].reset_index(drop=True)
            add -= take
    return vol_table, day_plan
