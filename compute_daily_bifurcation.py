"""Compute daily Paradip→Balasore and Balasore→Haldia linefill volumes.

The tool reads whitespace-separated batch volumes for each day from
``data/daily_batch_volumes.txt`` (one line per day, batches ordered from
Paradip to Haldia) and writes ``data/daily_bifurcation.csv`` with the split
between the two pipeline legs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

PIPE_AREA_M2 = 0.437263
PARADIP_TO_BALASORE_KM = 158
BALASORE_TO_HALDIA_KM = 170
BATCH_FILE = Path("data/daily_batch_volumes.txt")
OUTPUT_FILE = Path("data/daily_bifurcation.csv")


@dataclass
class SegmentVolumes:
    """Volumes held in each segment for a single day."""

    day: int
    paradip_to_balasore_m3: float
    balasore_to_haldia_m3: float
    total_linefill_m3: float


def _segment_capacities(area_m2: float, length_km: float) -> float:
    """Convert a pipe length to its volumetric capacity in cubic metres."""

    if area_m2 <= 0:
        raise ValueError("Pipe area must be positive")
    if length_km <= 0:
        raise ValueError("Segment length must be positive")
    return area_m2 * length_km * 1000  # km → m


def _split_linefill(batches: Iterable[float], seg1_cap: float, seg2_cap: float) -> Tuple[float, float]:
    """Allocate batch volumes into two consecutive segments."""

    remaining1 = seg1_cap
    remaining2 = seg2_cap
    seg1 = seg2 = 0.0
    for volume in batches:
        if remaining1 > 0:
            take = min(volume, remaining1)
            seg1 += take
            volume -= take
            remaining1 -= take
        if volume > 0 and remaining2 > 0:
            take = min(volume, remaining2)
            seg2 += take
            volume -= take
            remaining2 -= take
        if volume > 0:
            # Any excess would sit beyond Haldia; keep the split capped.
            continue
    return seg1, seg2


def load_batches(batch_file: Path) -> List[List[float]]:
    """Parse whitespace-separated batch volumes per day."""

    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    days: List[List[float]] = []
    for line in batch_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        days.append([float(part) for part in line.split()])
    return days


def compute_daily_splits(batch_file: Path = BATCH_FILE) -> List[SegmentVolumes]:
    """Return split volumes for each day in ``batch_file``."""

    seg1_cap = _segment_capacities(PIPE_AREA_M2, PARADIP_TO_BALASORE_KM)
    seg2_cap = _segment_capacities(PIPE_AREA_M2, BALASORE_TO_HALDIA_KM)
    results: List[SegmentVolumes] = []
    for day, batches in enumerate(load_batches(batch_file), start=1):
        seg1, seg2 = _split_linefill(batches, seg1_cap, seg2_cap)
        results.append(
            SegmentVolumes(
                day=day,
                paradip_to_balasore_m3=seg1,
                balasore_to_haldia_m3=seg2,
                total_linefill_m3=sum(batches),
            )
        )
    return results


def write_bifurcation_csv(results: List[SegmentVolumes], output_file: Path = OUTPUT_FILE) -> None:
    """Write the split volumes to ``output_file`` in CSV format."""

    header = "day,paradip_to_balasore_m3,balasore_to_haldia_m3,total_linefill_m3"
    lines = [
        f"{entry.day},{entry.paradip_to_balasore_m3:.3f},{entry.balasore_to_haldia_m3:.3f},{entry.total_linefill_m3:.3f}"
        for entry in results
    ]
    output_file.write_text("\n".join([header, *lines]))


def main() -> None:
    results = compute_daily_splits()
    write_bifurcation_csv(results)
    print(f"Wrote {len(results)} daily splits to {OUTPUT_FILE}")
    if results:
        first = results[0]
        print(
            f"Day 1: {first.paradip_to_balasore_m3:.2f} m³ in Paradip→Balasore, "
            f"{first.balasore_to_haldia_m3:.2f} m³ in Balasore→Haldia (total {first.total_linefill_m3:.2f} m³)"
        )


if __name__ == "__main__":
    main()
