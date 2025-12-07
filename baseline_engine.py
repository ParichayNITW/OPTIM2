from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List
from itertools import product   # ← THIS MUST BE PRESENT

from pipeline_model import _segment_hydraulics
from dra_utils import get_ppm_for_dr

def _single_pump_head_from_coeffs(A: float, B: float, C: float, flow: float) -> float:
    """
    Head of a single pump at DOL for the given bulk flow [m3/h],
    using the exact quadratic curve H(Q) = A*Q^2 + B*Q + C.

    No approximations; this is exactly what you encoded in your JSON.
    """
    h = A * flow * flow + B * flow + C
    return max(h, 0.0)


def _single_pump_head_from_pdata(pdata: Dict[str, Any], flow: float) -> float:
    """
    Extract A, B, C from a pump-type dictionary and compute head at DOL.
    """
    A = float(pdata.get("A", 0.0))
    B = float(pdata.get("B", 0.0))
    C = float(pdata.get("C", 0.0))
    return _single_pump_head_from_coeffs(A, B, C, flow)


def _max_head_at_dol(st_up: Dict[str, Any], flow: float) -> float:
    """
    Mathematically exact maximum station discharge head at DOL for the given
    bulk flow [m3/h], based SOLELY on the stored pump curves and pump counts.

    - Uses pump_types[*]["A","B","C"] when present.
    - Respects per-type "available" and station-level "min_pumps" / "max_pumps".
    - Assumes pumps at a station are in series (heads add at same flow),
      which is the same assumption used in the main optimizer.

    No arbitrary limits, no fudge factors.
    """
    try:
        q = float(flow or 0.0)
    except (TypeError, ValueError):
        q = 0.0

    if q <= 0.0:
        return 0.0

    pump_types = st_up.get("pump_types")
    # ---- MULTI-TYPE PATH (generic) -----------------------------------------
    if isinstance(pump_types, dict) and pump_types:
        type_keys: List[str] = list(pump_types.keys())
        avail_per_type: List[int] = []
        for k in type_keys:
            pdata = pump_types.get(k) or {}
            try:
                avail = int(pdata.get("available", 0) or 0)
            except (TypeError, ValueError):
                avail = 0
            avail_per_type.append(max(avail, 0))

        # Station-level pump bounds
        try:
            max_pumps_total = int(st_up.get("max_pumps", 0) or 0)
        except (TypeError, ValueError):
            max_pumps_total = 0

        if max_pumps_total <= 0:
            max_pumps_total = sum(avail_per_type)  # physical upper bound

        try:
            min_pumps_total = int(st_up.get("min_pumps", 0) or 0)
        except (TypeError, ValueError):
            min_pumps_total = 0

        best_head = 0.0

        # Enumerate all combinations of pump counts per type within availability
        ranges = [range(av + 1) for av in avail_per_type]
        for counts in product(*ranges):
            total_pumps = sum(counts)
            if total_pumps == 0:
                continue
            if total_pumps < min_pumps_total:
                continue
            if total_pumps > max_pumps_total:
                continue

            # All pumps are in series -> total head = sum(head_i)
            head = 0.0
            for k, n in zip(type_keys, counts):
                if n <= 0:
                    continue
                pdata = pump_types[k]
                h_single = _single_pump_head_from_pdata(pdata, q)
                head += n * h_single

            if head > best_head:
                best_head = head

        return max(best_head, 0.0)

    # ---- LEGACY SINGLE-TYPE PATH -------------------------------------------
    # For legacy stations without pump_types: use station-level A,B,C and max_pumps.
    A = float(st_up.get("A", 0.0))
    B = float(st_up.get("B", 0.0))
    C = float(st_up.get("C", 0.0))

    try:
        max_pumps = int(st_up.get("max_pumps", st_up.get("available", 0)) or 0)
    except (TypeError, ValueError):
        max_pumps = 0

    try:
        min_pumps = int(st_up.get("min_pumps", 0) or 0)
    except (TypeError, ValueError):
        min_pumps = 0

    if max_pumps <= 0:
        return 0.0

    h_single = _single_pump_head_from_coeffs(A, B, C, q)
    # Baseline engine wants the maximum achievable head -> use max_pumps.
    n = max(max_pumps, min_pumps, 0)
    return n * h_single



def schedule_baseline_injections(
    stations: List[Dict[str, Any]],
    terminal: Dict[str, Any],
    initial_linefill: List[Dict[str, Any]],
    flows: List[float],
    hours_per_slot: float,
    pump_shear_rate: float = 0.0,
    dra_reach_km: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Determine DRA injection actions to ensure flow feasibility.

    Parameters
    ----------
    stations : List of station dictionaries with pipeline geometry and pump/DRA info.
        Each station dict should include:
        - 'name': station name
        - 'L': segment length to next station (km)
        - 'D': outer pipe diameter (m)
        - 't': pipe wall thickness (m)
        - 'rough': pipe roughness (m)
        - 'elev': station elevation (m)
        - 'min_residual': minimum residual head at station (m)
        - 'max_dr': max drag reduction % (cap, >0 if injection available)
        - 'is_pump': True if station has pumping
        - Pump curve parameters as needed (e.g. 'DOL', 'min_pumps', 'max_pumps', 'A', 'B', 'C', etc.)
    terminal : dict of terminal constraints, including:
        - 'min_residual': required residual head at terminal (m)
        - 'elev': terminal elevation (m)
    initial_linefill : List of dicts with initial batches in pipeline, each dict:
        - 'volume': batch volume (m3) [first entry is nearest origin]
        - 'viscosity': kinematic viscosity of batch (cSt)
        - 'density': density of batch (kg/m3)
        - (optional 'dra_ppm': DRA concentration of batch)
    flows : list of flow rates (m3/h) for each time slot
    hours_per_slot : length of each time slot (hours)
    pump_shear_rate : global fraction [0..1] of DRA loss per pump (not used here)
    dra_reach_km : distance (km) that DRA remains effective (not used here)

    Returns
    -------
    actions : List of injection actions as dicts:
        [{'station': name, 'time': t, 'volume': V, 'dra_ppm': ppm}, ...]
    """
    # Build rolling linefill snapshots for each slot start
    snapshots: List[List[Dict[str, Any]]] = []
    current_linefill = deepcopy(initial_linefill)
    for flow in flows:
        snapshots.append(deepcopy(current_linefill))
        # Advance linefill by delivered volume
        delivered = flow * hours_per_slot
        remaining = delivered
        # Remove from tail of linefill
        while remaining > 0 and current_linefill:
            tail = current_linefill[-1]
            vol = float(tail.get("volume", 0.0))
            if vol > remaining:
                tail["volume"] = vol - remaining
                remaining = 0.0
            else:
                remaining -= vol
                current_linefill.pop()
        # Inject same volume at head as dummy to preserve pipeline volume (no DRA)
        added = delivered
        if added > 0:
            # Use a single dummy batch (could be split, but sum is preserved)
            current_linefill.insert(0, {"volume": added, "viscosity": 0.0, "density": 0.0, "dra_ppm": 0.0})

    # Precompute segment volumes and inner diameters
    num_segments = len(stations) - 1
    inner_diams = []
    segment_areas = []
    for i in range(num_segments):
        st = stations[i]
        d_inner = float(st["D"]) - 2.0 * float(st.get("t", 0.0))
        inner_diams.append(d_inner)
        if d_inner <= 0.0:
            segment_areas.append(0.0)
        else:
            area = math.pi * d_inner * d_inner / 4.0
            segment_areas.append(area)
    segment_volumes = []
    for i in range(num_segments):
        L = float(stations[i]["L"])
        area = segment_areas[i]
        segment_volumes.append(L * 1000.0 * area)  # m3

    # Compute volume-weighted viscosity and density for each segment and slot
    kv_matrix: List[List[float]] = [[0.0] * len(flows) for _ in range(num_segments)]
    rho_matrix: List[List[float]] = [[0.0] * len(flows) for _ in range(num_segments)]
    for k, snapshot in enumerate(snapshots):
        # Copy linefill for slicing
        queue = [batch.copy() for batch in snapshot]
        for i in range(num_segments):
            seg_vol = segment_volumes[i]
            if seg_vol <= 0.0 or not queue:
                kv = 0.0
                rho = 0.0
            else:
                needed = seg_vol
                visc_sum = 0.0
                dens_sum = 0.0
                while needed > 0 and queue:
                    batch = queue[0]
                    batch_vol = float(batch.get("volume", 0.0))
                    if batch_vol <= 0.0:
                        queue.pop(0)
                        continue
                    use_vol = min(batch_vol, needed)
                    visc = float(batch.get("viscosity", 0.0))
                    dens = float(batch.get("density", 0.0))
                    visc_sum += use_vol * visc
                    dens_sum += use_vol * dens
                    needed -= use_vol
                    if use_vol < batch_vol:
                        batch["volume"] = batch_vol - use_vol
                        break
                    else:
                        queue.pop(0)
                used = seg_vol - needed
                if used > 0.0:
                    kv = visc_sum / used
                    rho = dens_sum / used
                else:
                    kv = 0.0
                    rho = 0.0
            kv_matrix[i][k] = kv
            rho_matrix[i][k] = rho

    # Schedule DRA injections based on hydraulic feasibility
    actions: List[Dict[str, Any]] = []
    for k, flow in enumerate(flows):
        if flow <= 0.0:
            continue
        for i in range(num_segments):
            st_up = stations[i]
            if float(st_up.get("max_dr", 0.0)) <= 0.0:
                continue  # no injection facility at this station
            # Segment hydraulics (no DRA)
            L = float(st_up["L"])
            d_inner = inner_diams[i]
            rough = float(st_up.get("rough", 0.0))
            kv = kv_matrix[i][k]
            # Calculate head loss for this segment at flow (m of fluid)
            head_loss, *_ = _segment_hydraulics(flow, L, d_inner, rough, kv, 0.0, None)
            # Downstream residual head requirement
            if i + 1 < len(stations):
                st_down = stations[i+1]
                res_down = float(st_down.get("min_residual", 0.0))
                elev_down = float(st_down.get("elev", 0.0))
            else:
                res_down = float(terminal.get("min_residual", 0.0))
                elev_down = float(terminal.get("elev", 0.0))
            elev_up = float(st_up.get("elev", 0.0))
            # Required upstream head = friction + downstream head + elevation change
            required_head = head_loss + res_down + (elev_down - elev_up)
            # Available head at upstream station
            if st_up.get("is_pump", False):
                max_head = _max_head_at_dol(st_up, flow)
            else:
                max_head = 0.0
            suction_head = float(st_up.get("min_residual", 0.0))
            available_head = max_head + suction_head
            # Check shortfall
            gap = required_head - available_head
            if gap <= 0.0 or head_loss <= 0.0:
                continue
            # % drag reduction needed
            dr_pct = (gap / head_loss) * 100.0
            dr_pct = max(dr_pct, 0.0)
            dr_pct = min(dr_pct, 100.0)
            # Cap by station's max DR
            station_max_dr = float(st_up.get("max_dr", 0.0))
            if station_max_dr > 0.0:
                dr_pct = min(dr_pct, station_max_dr)
            if dr_pct <= 0.0:
                continue
            # Compute PPM from drag reduction
            ppm = get_ppm_for_dr(kv, dr_pct)
            ppm = max(ppm, 0.0)
            # Schedule injection timing and volume
            seg_vol = segment_volumes[i]
            travel_time = seg_vol / flow  # hours to push this volume
            inject_time = k * hours_per_slot - travel_time
            if inject_time < 0.0:
                inject_time = 0.0
            action = {
                "station": st_up.get("name"),
                "time": inject_time,
                "volume": seg_vol,
                "dra_ppm": ppm,
            }
            actions.append(action)

    # Sort actions by time and station for consistency
    actions.sort(key=lambda x: (x["time"], str(x["station"])))
    return actions
