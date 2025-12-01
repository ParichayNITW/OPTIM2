"""Reproduce the screenshot walk-through scenario for the DRA profile.

The scenario described in the user-provided screenshot uses:

* Three segments: 80 km, 60 km and 69 km (total 209 km).
* Initial linefill at 05:00: first 80 km @ 5 ppm, remainder @ 0 ppm.
* DRA injection: 5 ppm only for the first 30 minutes (05:00–05:30),
  then 0 ppm afterwards.
* Throughput chosen so that every 30 minutes exactly 5.88 km of product
  moves along the pipe.

This script advances the queue in 30-minute increments by trimming the
downstream tail (delivered volume) and prepending the upstream injected
slug for that interval.  After each hour it prints the per-segment DRA
profiles using the same helpers the optimiser employs
(`_segment_profile_from_queue` and friends) so we can compare the live
logic to the manual table in the screenshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_model


@dataclass
class Step:
    """Half-hourly injection schedule entry."""

    label: str
    injected_ppm: float


def _prepend_slice(
    queue: Sequence[tuple[float, float]],
    *,
    length_km: float,
    ppm: float,
) -> list[tuple[float, float]]:
    """Return ``queue`` with a new head slice merged in if ppm matches."""

    merged = pipeline_model._merge_queue([(length_km, ppm)] + list(queue))
    return [(float(length), float(ppm_val)) for length, ppm_val in merged if float(length) > 0.0]


def advance_queue(
    queue: Sequence[tuple[float, float]],
    pumped_length_km: float,
    injected_ppm: float,
) -> list[tuple[float, float]]:
    """Trim delivered product and prepend the newly injected slug.

    The queue stores slices from upstream (index 0) to downstream.  When the
    line moves forward, we trim the tail (delivered volume) and then prepend
    the injected slice for the current interval at the head.
    """

    trimmed, leftover = pipeline_model._trim_queue_tail(queue, pumped_length_km)
    if leftover > 1e-9:
        raise ValueError(f"pumped_length_km exceeds queue length by {leftover:.6f} km")
    return _prepend_slice(trimmed, length_km=pumped_length_km, ppm=injected_ppm)


def segment_profiles(
    queue: Sequence[tuple[float, float]],
    segments: Iterable[float],
) -> list[tuple[float, tuple[tuple[float, float], ...]]]:
    """Return raw segment profiles for the supplied queue."""

    profiles: list[tuple[float, tuple[tuple[float, float], ...]]] = []
    offset = 0.0
    queue_tuple = tuple(queue)
    for seg_length in segments:
        profile = pipeline_model._segment_profile_from_queue(queue_tuple, offset, seg_length)
        profiles.append((float(seg_length), profile))
        offset += float(seg_length)
    return profiles


def main() -> None:
    segments = (80.0, 60.0, 69.0)
    pumped_per_step_km = 5.88
    queue: list[tuple[float, float]] = [(80.0, 5.0), (129.0, 0.0)]

    schedule = [
        Step("05:00", injected_ppm=5.0),
        Step("05:30", injected_ppm=5.0),
        Step("06:00", injected_ppm=0.0),
        Step("06:30", injected_ppm=0.0),
        Step("07:00", injected_ppm=0.0),
        Step("07:30", injected_ppm=0.0),
        Step("08:00", injected_ppm=0.0),
        Step("08:30", injected_ppm=0.0),
        Step("09:00", injected_ppm=0.0),
    ]

    print("Initial queue (05:00)")
    for seg_len, profile in segment_profiles(queue, segments):
        print(f"  Segment {seg_len:.0f} km: {profile}")

    for idx, step in enumerate(schedule[1:], start=1):
        queue = advance_queue(queue, pumped_per_step_km, step.injected_ppm)

        # Print at each whole hour boundary only (06:00, 07:00, ...)
        if idx % 2 == 0:  # every 60 minutes
            print(f"\nProfile at {step.label}")
            for seg_len, profile in segment_profiles(queue, segments):
                print(f"  Segment {seg_len:.0f} km: {profile}")


if __name__ == "__main__":
    main()
