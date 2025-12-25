"""Command-line entrypoint for the pipeline optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pipeline_model
from pipeline_case_utils import (
    dump_json,
    merge_segment_floors,
    prepare_case,
)


def _load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pipeline optimizer from a saved case JSON.",
    )
    parser.add_argument(
        "--case",
        required=True,
        type=Path,
        help="Path to a case JSON exported from the UI.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for solver results JSON.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Override the simulation hours (default from case or 24).",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Override the solver start time (HH:MM).",
    )
    parser.add_argument(
        "--dra-reach-km",
        type=float,
        default=None,
        help="Override DRA reach in km.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    case_data = _load_case(args.case)
    prepared = prepare_case(case_data)

    stations = prepared["stations"]
    terminal = prepared["terminal"]
    kv_list = prepared["kv_list"]
    rho_list = prepared["rho_list"]
    segment_slices = prepared["segment_slices"]
    linefill = prepared["linefill_batches"]

    hours = args.hours if args.hours is not None else float(case_data.get("hours", 24.0) or 24.0)
    start_time = args.start_time if args.start_time is not None else str(case_data.get("start_time", "00:00"))
    dra_reach_km = (
        args.dra_reach_km
        if args.dra_reach_km is not None
        else float(case_data.get("dra_reach_km", 0.0) or 0.0)
    )

    flow = float(case_data.get("FLOW", 1000.0) or 0.0)
    rate_dra = float(case_data.get("RateDRA", 500.0) or 0.0)
    price_hsd = float(case_data.get("Price_HSD", 70.0) or 0.0)
    fuel_density = float(case_data.get("Fuel_density", 820.0) or 0.0)
    ambient_temp = float(case_data.get("Ambient_temp", 25.0) or 0.0)
    mop_kgcm2 = float(case_data.get("MOP_kgcm2", 100.0) or 0.0)
    pump_shear_rate = float(case_data.get("pump_shear_rate", 0.0) or 0.0)

    segment_floors = merge_segment_floors(case_data)

    has_types = any(stn.get("pump_types") for stn in stations)
    solver = pipeline_model.solve_pipeline_with_types if has_types else pipeline_model.solve_pipeline

    result = solver(
        stations,
        terminal,
        flow,
        kv_list,
        rho_list,
        segment_slices,
        rate_dra,
        price_hsd,
        fuel_density,
        ambient_temp,
        linefill,
        dra_reach_km,
        mop_kgcm2,
        hours,
        start_time=start_time,
        pump_shear_rate=pump_shear_rate,
        segment_floors=segment_floors,
    )

    payload = dump_json(result, str(args.output) if args.output else None, pretty=args.pretty)
    if args.output is None:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
