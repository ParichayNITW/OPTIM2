
"""Generate DRA profile combinations for a three-station pipeline scenario."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Sequence, Tuple

import pipeline_model as pm


@dataclass(frozen=True)
class Scenario:
    """Static description of the reference pipeline scenario."""

    length_km: float
    volume_m3: float
    segment_lengths: Sequence[float]
    baseline_queue: Sequence[Tuple[float, float]]
    dra_min_ppm: float
    dra_max_ppm: float
    dra_selection_ppm: float
    kv: float = 3.0

    def inner_diameter(self) -> float:
        """Return the inner diameter implied by line volume and length."""

        length_m = max(self.length_km, 0.0) * 1000.0
        if length_m <= 0:
            return 0.0
        area = self.volume_m3 / length_m
        if area <= 0.0:
            return 0.0
        return math.sqrt(4.0 * area / math.pi)


def _format_segments(profile: Sequence[Tuple[float, float]]) -> str:
    """Return a formatted representation of ``profile`` for console output."""

    if not profile:
        return "(no treated footage)"
    return "; ".join(f"{length:.2f} km @ {ppm:.2f} ppm" for length, ppm in profile)


def _normalise_segments(profile: Iterable[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """Round and merge ``profile`` entries for stable comparisons."""

    merged: list[Tuple[float, float]] = []
    for length, ppm in profile:
        length_val = round(float(length), 6)
        ppm_val = round(float(ppm), 6)
        if length_val <= 0.0:
            continue
        if merged and abs(merged[-1][1] - ppm_val) <= 1e-9:
            prev_len, prev_ppm = merged[-1]
            merged[-1] = (round(prev_len + length_val, 6), prev_ppm)
        else:
            merged.append((length_val, ppm_val))
    return tuple((round(length, 3), round(ppm, 3)) for length, ppm in merged)


def _segment_profiles_from_queue(
    queue: Sequence[Tuple[float, float]],
    segment_lengths: Sequence[float],
) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    """Return the profile covering each segment in ``segment_lengths``."""

    profiles: list[Tuple[Tuple[float, float], ...]] = []
    upstream = 0.0
    for seg_len in segment_lengths:
        profile = pm._segment_profile_from_queue(queue, upstream, seg_len)
        profiles.append(_normalise_segments(profile))
        upstream += seg_len
    return tuple(profiles)


def generate_combination_profiles(
    scenario: Scenario,
) -> list[dict[str, object]]:
    """Return DRA profiles for every pump/DRA combination over three hours."""

    d_inner = scenario.inner_diameter()
    segments = list(float(seg) for seg in scenario.segment_lengths)
    stations = [
        {"name": "Station 1", "idx": 0, "d_inner": d_inner, "kv": scenario.kv},
        {"name": "Station 2", "idx": 1, "d_inner": d_inner, "kv": scenario.kv},
        {"name": "Station 3", "idx": 2, "d_inner": d_inner, "kv": scenario.kv},
    ]

    initial_queue = [
        (float(length), float(ppm))
        for length, ppm in scenario.baseline_queue
        if float(length) > 0.0
    ]

    results: list[dict[str, object]] = []

    for s1_dra, s2_dra, s2_pump, s3_pump in product([False, True], repeat=4):
        inj_s1 = scenario.dra_selection_ppm if s1_dra else 0.0
        inj_s2 = scenario.dra_selection_ppm if s2_dra else 0.0

        case_entry: dict[str, object] = {
            "S1 DRA": "On" if s1_dra else "Off",
            "S2 DRA": "On" if s2_dra else "Off",
            "S2 Pump": "On" if s2_pump else "Off",
            "S3 Pump": "On" if s3_pump else "Off",
            "profiles": [],
        }

        queue_state = list(initial_queue)
        history: list[dict[str, object]] = []

        initial_profiles = _segment_profiles_from_queue(queue_state, segments)
        history.append(
            {
                "time": "07:00",
                "segments": initial_profiles,
            }
        )

        flow_m3h = scenario.volume_m3 / (scenario.length_km / 5.0) if scenario.length_km > 0 else 0.0
        hours = 1.0

        for step in range(3):
            hour_profiles: list[Tuple[Tuple[float, float], ...]] = []
            for idx, (station, seg_len) in enumerate(zip(stations, segments)):
                pump_running = True if idx == 0 else (s2_pump if idx == 1 else s3_pump)
                inj_ppm = inj_s1 if idx == 0 else (inj_s2 if idx == 1 else 0.0)
                opt = {"dra_ppm_main": inj_ppm}
                dra_segments, queue_state, *_ = pm._update_mainline_dra(
                    queue_state,
                    station,
                    opt,
                    seg_len,
                    flow_m3h,
                    hours,
                    pump_running=pump_running,
                    pump_shear_rate=0.0,
                    dra_shear_factor=0.0,
                    shear_injection=False,
                    is_origin=(idx == 0),
                    segment_floor=None,
                )
                hour_profiles.append(_normalise_segments(dra_segments))
            history.append(
                {
                    "time": f"{8 + step:02d}:00",
                    "segments": tuple(hour_profiles),
                }
            )

        case_entry["profiles"] = history
        results.append(case_entry)

    return results


def main() -> None:
    """Entry point used when executing the module as a script."""

    scenario = Scenario(
        length_km=200.0,
        volume_m3=120_000.0,
        segment_lengths=(80.0, 60.0, 60.0),
        baseline_queue=((140.0, 5.0), (60.0, 0.0)),
        dra_min_ppm=3.0,
        dra_max_ppm=6.0,
        dra_selection_ppm=6.0,
    )
    profiles = generate_combination_profiles(scenario)

    for entry in profiles:
        header = (
            f"S1 DRA: {entry['S1 DRA']}, "
            f"S2 DRA: {entry['S2 DRA']}, "
            f"S2 Pump: {entry['S2 Pump']}, "
            f"S3 Pump: {entry['S3 Pump']}"
        )
        print(header)
        for snapshot in entry["profiles"]:
            print(f"  {snapshot['time']}")
            for idx, segment in enumerate(snapshot["segments"], start=1):
                print(f"    S{idx}: {_format_segments(segment)}")
        print()


if __name__ == "__main__":
    main()
