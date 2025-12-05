"""Baseline DRA engine utilities for the dynamic optimizer."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List

from dra_utils import get_ppm_for_dr
from linefill_utils import advance_linefill
from pipeline_optimization_app import map_linefill_to_segments


def build_linefill_snapshots(
    stations: List[Dict],
    initial_linefill: List[Dict],
    flows: List[float],
    hours_per_slot: float,
) -> List[List[Dict]]:
    """Build rolling linefill snapshots for the entire horizon."""

    snapshots: List[List[Dict]] = []
    current = deepcopy(initial_linefill)

    for flow in flows:
        snapshots.append(deepcopy(current))
        delivered_volume = flow * hours_per_slot
        dummy_schedule: list[Any] = []
        advance_linefill(current, dummy_schedule, delivered_volume)

    return snapshots


def compute_segment_viscosities(
    snapshots: List[List[Dict]],
    stations: List[Dict],
):
    """Compute per-segment volume-weighted viscosity and density for snapshots."""

    kv_matrix: list[list[float]] = []
    rho_matrix: list[list[float]] = []

    for linefill in snapshots:
        _, kv_list, rho_list, seg_slices = map_linefill_to_segments(linefill, stations)

        kv_corrected: list[float] = []
        rho_corrected: list[float] = []

        for seg_idx, slices in enumerate(seg_slices):
            total_vol = 0.0
            visc_num = 0.0
            rho_num = 0.0

            for s in slices:
                length_km = s["length_km"]
                kv = s["kv"]
                rho = s["rho"]

                station = stations[seg_idx]
                diameter = station["D"] - 2 * station["t"]
                area = math.pi * diameter * diameter / 4.0
                vol = length_km * 1000.0 * area

                total_vol += vol
                visc_num += vol * kv
                rho_num += vol * rho

            kv_corrected.append(visc_num / total_vol if total_vol > 0 else 0.0)
            rho_corrected.append(rho_num / total_vol if total_vol > 0 else 0.0)

        kv_matrix.append(kv_corrected)
        rho_matrix.append(rho_corrected)

    return kv_matrix, rho_matrix


def compute_required_ppm(
    stations: List[Dict],
    flows: List[float],
    kv_matrix,
    rho_matrix,
    user_floor_ppm: float,
):
    """Compute minimum required PPM for each segment and time slot."""

    num_segments = len(stations) - 1
    num_slots = len(flows)

    ppm_required = [[0.0 for _ in range(num_slots)] for _ in range(num_segments)]

    for k in range(num_slots):
        flow = flows[k]
        for i in range(num_segments):
            upstream = stations[i]
            if upstream.get("max_dr", 0) <= 0:
                continue

            kv = kv_matrix[k][i]
            rho = rho_matrix[k][i]
            _ = rho  # reserved for future use
            _ = stations
            _ = flows
            _ = user_floor_ppm
            length_km = stations[i]["L"]
            _ = length_km

            dr_needed = max(0.0, min(70.0, (flow / 5000.0) * 10.0))
            ppm = get_ppm_for_dr(dr_needed, kv)

            ppm_required[i][k] = max(ppm, user_floor_ppm)

    return ppm_required


def backpropagate_injection(
    snapshots: List[List[Dict]],
    ppm_required,
):
    """Backpropagate the minimum DRA ppm to inject at each station per slot."""

    num_segments = len(ppm_required)
    num_slots = len(ppm_required[0]) if ppm_required else 0

    baseline_injection = [[0.0 for _ in range(num_slots)] for _ in range(num_segments)]

    for seg in range(num_segments):
        for k in range(num_slots):
            baseline_injection[seg][k] = max(baseline_injection[seg][k], ppm_required[seg][k])

    return baseline_injection
