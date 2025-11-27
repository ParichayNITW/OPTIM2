"""Generate 3-hourly DRA profiles for the S1-S2-S3-T example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline_model import (
    _merge_queue,
    _segment_profile_from_queue,
    _take_queue_front,
    _trim_queue_front,
    _update_mainline_dra,
    _volume_from_km,
)


diameter = 0.8
segment_lengths = (80.0, 60.0, 60.0)
flow_m3h = _volume_from_km(segment_lengths[0], diameter)  # pump 80 km of pipe per hour
hours = 1.0


@dataclass
class Case:
    idx: int
    s1_pump: bool
    s1_dra: bool
    s2_pump: bool
    s2_dra: bool

    @property
    def inj1(self) -> float:
        return 6.0 if self.s1_dra else 0.0

    @property
    def inj2(self) -> float:
        return 6.0 if self.s2_dra else 0.0


station1 = {"idx": 0, "is_pump": True, "d_inner": diameter}
station2 = {"idx": 1, "is_pump": True, "d_inner": diameter}


cases: list[Case] = []
idx = 1
for s1_pump in (False, True):
    for s1_dra in (False, True):
        for s2_pump in (False, True):
            for s2_dra in (False, True):
                cases.append(Case(idx, s1_pump, s1_dra, s2_pump, s2_dra))
                idx += 1


def _as_profile_string(profile: Iterable[tuple[float, float]]) -> str:
    parts: list[str] = []
    last_ppm: float | None = None
    acc_length = 0.0
    for length, ppm in profile:
        length = float(length)
        ppm = float(ppm)
        if length <= 0:
            continue
        if last_ppm is None:
            last_ppm = ppm
            acc_length = length
            continue
        if abs(ppm - last_ppm) < 1e-9:
            acc_length += length
        else:
            parts.append(f"{acc_length:.0f} km @ {last_ppm:.1f} ppm")
            last_ppm = ppm
            acc_length = length
    if last_ppm is not None and acc_length > 0:
        parts.append(f"{acc_length:.0f} km @ {last_ppm:.1f} ppm")
    return ", ".join(parts) or "0 km @ 0 ppm"


def run_case(case: Case) -> list[dict[str, str]]:
    queue = [
        {"length_km": 140.0, "dra_ppm": 5.0},
        {"length_km": 60.0, "dra_ppm": 0.0},
    ]
    hourly_rows: list[dict[str, str]] = []

    for hour in range(1, 4):
        dra_segments_s1, queue_after_s1, _, _ = _update_mainline_dra(
            queue,
            station1,
            {"nop": 1 if case.s1_pump else 0, "dra_ppm_main": case.inj1},
            segment_lengths[0],
            flow_m3h,
            hours,
            pump_running=case.s1_pump,
            pump_shear_rate=1.0,
            is_origin=True,
        )

        merged_after_s1 = _merge_queue(
            tuple(
                (float(entry["length_km"]), float(entry["dra_ppm"]))
                for entry in queue_after_s1
                if float(entry.get("length_km", 0.0)) > 0.0
            )
        )
        profile_s1 = list(
            (float(length), float(ppm))
            for length, ppm in _segment_profile_from_queue(merged_after_s1, 0.0, segment_lengths[0])
        )

        prefix_s1 = _take_queue_front(merged_after_s1, segment_lengths[0])
        inlet_s2 = _trim_queue_front(merged_after_s1, segment_lengths[0])

        dra_segments_s2, queue_after_s2, _, _ = _update_mainline_dra(
            [
                {"length_km": float(length), "dra_ppm": float(ppm)}
                for length, ppm in inlet_s2
                if float(length) > 0.0
            ],
            station2,
            {"nop": 1 if case.s2_pump else 0, "dra_ppm_main": case.inj2},
            segment_lengths[1],
            flow_m3h,
            hours,
            pump_running=case.s2_pump,
            pump_shear_rate=1.0,
        )

        merged_after_s2 = _merge_queue(
            tuple(prefix_s1)
            + tuple(
                (float(entry["length_km"]), float(entry["dra_ppm"]))
                for entry in queue_after_s2
                if float(entry.get("length_km", 0.0)) > 0.0
            )
        )

        profile_s2 = list(
            (float(length), float(ppm))
            for length, ppm in _segment_profile_from_queue(
                merged_after_s2, segment_lengths[0], segment_lengths[1]
            )
        )
        profile_s3 = list(
            (float(length), float(ppm))
            for length, ppm in _segment_profile_from_queue(
                merged_after_s2, segment_lengths[0] + segment_lengths[1], segment_lengths[2]
            )
        )

        queue = [
            {"length_km": float(length), "dra_ppm": float(ppm)}
            for length, ppm in merged_after_s2
            if float(length) > 0.0
        ]

        hourly_rows.append(
            {
                "Hour": f"Hour {hour}",
                "After S1 profile": _as_profile_string(profile_s1),
                "After S2 profile": _as_profile_string(profile_s2),
                "After S3 profile": _as_profile_string(profile_s3),
            }
        )

    return hourly_rows


def build_markdown_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(" | ".join(row[h] for h in headers))
    return "\n".join(lines)


def main() -> None:
    output_lines: list[str] = []
    for case in cases:
        hourly_rows = run_case(case)
        output_lines.append(
            f"### Case {case.idx}: S1 Pump {'ON' if case.s1_pump else 'OFF'}, "
            f"S1 DRA {'ON' if case.s1_dra else 'OFF'}, "
            f"S2 Pump {'ON' if case.s2_pump else 'OFF'}, "
            f"S2 DRA {'ON' if case.s2_dra else 'OFF'}"
        )
        output_lines.append("")
        output_lines.append(build_markdown_table(hourly_rows))
        output_lines.append("")
    Path("examples").mkdir(exist_ok=True)
    output_path = Path("examples/hourly_output_tables.md")
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    from pathlib import Path

    main()
