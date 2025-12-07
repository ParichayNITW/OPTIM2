from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List

from pipeline_model import _segment_hydraulics, _max_head_at_dol
from dra_utils import get_ppm_for_dr

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
