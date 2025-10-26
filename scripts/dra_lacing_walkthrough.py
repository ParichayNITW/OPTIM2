"""Generate a step-by-step DRA lacing walkthrough for an A–B–C pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_model import (
    _merge_queue,
    _prepare_dra_queue_consumption,
    _queue_total_length,
    _segment_profile_from_queue,
    _take_queue_front,
    _trim_queue_front,
    _update_mainline_dra,
    _volume_from_km,
)


def _normalise_queue(
    entries: Sequence[Sequence[float]] | Sequence[dict],
) -> tuple[tuple[float, float], ...]:
    normalised: list[tuple[float, float]] = []
    for entry in entries:
        if not entry:
            continue
        if isinstance(entry, dict):
            length = float(entry.get("length_km", 0.0) or 0.0)
            ppm = float(entry.get("dra_ppm", 0.0) or 0.0)
        else:
            length = float(entry[0])
            ppm = float(entry[1]) if len(entry) > 1 else 0.0
        if length <= 0:
            continue
        normalised.append((length, ppm))
    merged = _merge_queue(normalised)
    return tuple((float(length), float(ppm)) for length, ppm in merged if float(length) > 0)


def _format_profile(
    profile: Iterable[tuple[float, float]],
    *,
    limit: int | None = None,
) -> str:
    parts: list[str] = []
    for length, ppm in profile:
        length = float(length)
        ppm = float(ppm)
        if length <= 1e-9:
            continue
        parts.append(f"{length:4.1f} km @ {ppm:4.1f} ppm")
    if not parts:
        return "0.0 km (untreated)"
    if limit is not None and limit > 0 and len(parts) > limit:
        remaining = len(parts) - limit
        display = parts[:limit] + [f"… (+{remaining} more)"]
    else:
        display = parts
    return "; ".join(display)


def main() -> None:
    flow_m3h = 1000.0
    hours = 1.0
    stn_a = {
        "idx": 0,
        "name": "Station A",
        "L": 40.0,
        "d_inner": 0.33,
        "kv": 3.0,
        "dra_injector_position": "downstream",
    }
    stn_b = {
        "idx": 1,
        "name": "Station B",
        "L": 60.0,
        "d_inner": 0.33,
        "kv": 3.0,
        "dra_injector_position": "downstream",
    }

    opt_a = {"dra_ppm_main": 8.0, "nop": 1}
    opt_b = {"dra_ppm_main": 4.0, "nop": 1}

    queue_full: tuple[tuple[float, float], ...] = ((100.0, 0.0),)

    header = (
        "| Hour | A injection (ppm) | B injection (ppm) | A→B treated profile | "
        "B→C treated profile | Downstream queue after B | Treated length (km) | "
        "Linefill volume (m³) |"
    )
    separator = (
        "| ---: | ---: | ---: | --- | --- | --- | ---: | ---: |"
    )
    print(header)
    print(separator)

    for hour in range(1, 7):
        queue_inlet_a = queue_full
        pre_a = _prepare_dra_queue_consumption(
            queue_inlet_a, stn_a["L"], flow_m3h, hours, stn_a["d_inner"]
        )
        dra_seg_a, queue_after_list_a, inj_a, _ = _update_mainline_dra(
            queue_inlet_a,
            stn_a,
            opt_a,
            stn_a["L"],
            flow_m3h,
            hours,
            pump_running=True,
            pump_shear_rate=0.0,
            dra_shear_factor=0.0,
            shear_injection=False,
            is_origin=True,
            precomputed=pre_a,
        )
        queue_after_full_a = _normalise_queue(queue_after_list_a)
        queue_after_inlet_a = _trim_queue_front(queue_after_full_a, stn_a["L"])

        total_prev_full = _queue_total_length(queue_after_full_a)
        total_prev_inlet = _queue_total_length(queue_after_inlet_a)
        upstream_length = max(total_prev_full - total_prev_inlet, 0.0)
        prefix_entries = _take_queue_front(queue_after_full_a, upstream_length)

        pre_b = _prepare_dra_queue_consumption(
            queue_after_inlet_a, stn_b["L"], flow_m3h, hours, stn_b["d_inner"]
        )
        dra_seg_b, queue_after_list_b, inj_b, _ = _update_mainline_dra(
            queue_after_inlet_a,
            stn_b,
            opt_b,
            stn_b["L"],
            flow_m3h,
            hours,
            pump_running=True,
            pump_shear_rate=0.20,
            dra_shear_factor=0.0,
            shear_injection=False,
            is_origin=False,
            precomputed=pre_b,
        )
        queue_after_body_b = _normalise_queue(queue_after_list_b)
        combined_after_b = tuple(prefix_entries) + tuple(queue_after_body_b)
        queue_after_full_b = _normalise_queue(combined_after_b)
        queue_after_inlet_b = _trim_queue_front(queue_after_full_b, stn_b["L"])

        queue_full = queue_after_full_b

        profile_a = _segment_profile_from_queue(queue_after_full_b, 0.0, stn_a["L"])
        profile_b = _segment_profile_from_queue(queue_after_full_b, stn_a["L"], stn_b["L"])

        linefill_summary = _format_profile(queue_after_full_b, limit=4)
        a_profile_summary = _format_profile(profile_a, limit=4)
        b_profile_summary = _format_profile(profile_b, limit=4)

        treated_length = sum(
            float(length)
            for length, ppm in queue_after_full_b
            if float(ppm) > 1e-9
        )
        linefill_volume = _volume_from_km(
            _queue_total_length(queue_after_full_b), stn_a["d_inner"]
        )

        print(
            f"| {hour:>4} | {inj_a:>7.2f} | {inj_b:>7.2f} | {a_profile_summary} | "
            f"{b_profile_summary} | {linefill_summary} | {treated_length:>6.1f} | "
            f"{linefill_volume:>10.1f} |"
        )

        queue_full = queue_after_full_b
        queue_inlet_a = queue_after_inlet_b


if __name__ == "__main__":
    main()
