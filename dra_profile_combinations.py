
"""Enumerate DRA profiles for arbitrary pump/DRA combinations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Sequence, Tuple

import pipeline_model as pm


@dataclass(frozen=True)
class Station:
    """Description of a pipeline station used for simulation."""

    name: str
    length_km: float
    has_pump: bool = True
    has_dra: bool = False
    pump_forced_on: bool = False
    dra_ppm_on: float | None = None
    kv: float | None = None


@dataclass(frozen=True)
class Scenario:
    """Static description of a pipeline used for combination generation."""

    length_km: float
    volume_m3: float
    stations: Sequence[Station]
    baseline_queue: Sequence[Tuple[float, float]]
    dra_min_ppm: float
    dra_max_ppm: float
    km_per_hour: float

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
    *,
    hours: int = 3,
) -> list[dict[str, object]]:
    """Return DRA profiles for every pump/DRA combination over ``hours`` steps."""

    d_inner = scenario.inner_diameter()
    segments = [float(stn.length_km) for stn in scenario.stations]

    pm_stations = [
        {
            "name": station.name,
            "idx": idx,
            "d_inner": d_inner,
            "kv": float(station.kv if station.kv is not None else 3.0),
        }
        for idx, station in enumerate(scenario.stations)
    ]

    initial_queue = [
        (float(length), float(ppm))
        for length, ppm in scenario.baseline_queue
        if float(length) > 0.0
    ]

    if scenario.length_km > 0.0 and scenario.km_per_hour > 0.0:
        diameter = d_inner
        area = math.pi * (diameter ** 2) / 4.0 if diameter > 0.0 else 0.0
        flow_m3h = scenario.km_per_hour * 1000.0 * area if area > 0.0 else 0.0
    else:
        flow_m3h = 0.0

    state_labels: list[str] = []
    state_variants: list[list[bool]] = []
    for station in scenario.stations:
        if station.has_dra:
            state_labels.append(f"{station.name} DRA")
            state_variants.append([False, True])
        if station.has_pump and not station.pump_forced_on:
            state_labels.append(f"{station.name} Pump")
            state_variants.append([False, True])
    if not state_labels:
        state_labels = ["(no toggles)"]
        state_variants = [[False]]

    results: list[dict[str, object]] = []

    for states in product(*state_variants):
        state_map = dict(zip(state_labels, states, strict=False))
        case_entry: dict[str, object] = {label: ("On" if state_map[label] else "Off") for label in state_labels}
        case_entry.setdefault("profiles", [])

        queue_state = list(initial_queue)
        history: list[dict[str, object]] = []

        initial_profiles = _segment_profiles_from_queue(queue_state, segments)
        history.append({"time": "07:00", "segments": initial_profiles})

        for step in range(hours):
            hour_profiles: list[Tuple[Tuple[float, float], ...]] = []
            for idx, (station_cfg, seg_len) in enumerate(zip(pm_stations, segments)):
                logical = scenario.stations[idx]
                pump_label = f"{logical.name} Pump"
                pump_running = True if logical.pump_forced_on else state_map.get(pump_label, logical.pump_forced_on)

                dra_label = f"{logical.name} DRA"
                dra_on = state_map.get(dra_label, False)
                dra_ppm = logical.dra_ppm_on if logical.dra_ppm_on is not None else scenario.dra_max_ppm
                inj_ppm = float(dra_ppm if dra_on else 0.0)

                opt = {"dra_ppm_main": inj_ppm}

                dra_segments, queue_state, *_ = pm._update_mainline_dra(
                    queue_state,
                    station_cfg,
                    opt,
                    seg_len,
                    flow_m3h,
                    1.0,
                    pump_running=pump_running,
                    pump_shear_rate=0.0,
                    dra_shear_factor=0.0,
                    shear_injection=False,
                    is_origin=(idx == 0),
                    segment_floor=None,
                )

                hour_profiles.append(_normalise_segments(dra_segments))

            history.append({"time": f"{8 + step:02d}:00", "segments": tuple(hour_profiles)})

        case_entry["profiles"] = history
        results.append(case_entry)

    return results


def main() -> None:
    """Entry point used when executing the module as a script."""

    scenario = Scenario(
        length_km=200.0,
        volume_m3=120_000.0,
        stations=(
            Station("Station 1", 80.0, has_pump=True, has_dra=True, pump_forced_on=True, dra_ppm_on=6.0, kv=3.0),
            Station("Station 2", 60.0, has_pump=True, has_dra=True, dra_ppm_on=6.0, kv=3.0),
            Station("Station 3", 60.0, has_pump=True, has_dra=False),
        ),
        baseline_queue=((140.0, 5.0), (60.0, 0.0)),
        dra_min_ppm=3.0,
        dra_max_ppm=6.0,
        km_per_hour=5.0,
    )
    profiles = generate_combination_profiles(scenario)

    for entry in profiles:
        header_parts = [f"{label}: {state}" for label, state in entry.items() if label != "profiles"]
        header = ", ".join(header_parts)
        print(header)
        for snapshot in entry["profiles"]:
            print(f"  {snapshot['time']}")
            for idx, segment in enumerate(snapshot["segments"], start=1):
                print(f"    {scenario.stations[idx - 1].name}: {_format_segments(segment)}")
        print()


if __name__ == "__main__":
    main()
