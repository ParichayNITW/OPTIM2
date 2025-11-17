#!/usr/bin/env python3
"""Dump the optimiser's ppm-vs-km output for the 150 km ABC pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dra_profile_combinations import Scenario, Station, generate_combination_profiles


def _format_profile(profile: tuple[tuple[float, float], ...]) -> str:
    if not profile:
        return "(empty)"
    return " + ".join(f"{length:.1f} km @ {ppm:.1f} ppm" for length, ppm in profile)


def main() -> None:
    scenario = Scenario(
        length_km=150.0,
        volume_m3=90_000.0,
        stations=(
            Station(
                "Segment A→B",
                100.0,
                has_pump=True,
                pump_forced_on=True,
                has_dra=True,
                dra_ppm_on=3.0,
            ),
            Station(
                "Segment B→C",
                50.0,
                has_pump=True,
                has_dra=True,
                dra_ppm_on=3.0,
            ),
        ),
        baseline_queue=((150.0, 0.0),),
        dra_min_ppm=2.0,
        dra_max_ppm=3.0,
        km_per_hour=5.0,
        pump_shear_rate=1.0,
    )

    profiles = generate_combination_profiles(scenario, hours=2)

    for entry in profiles:
        states = [f"{label}: {state}" for label, state in entry.items() if label != "profiles"]
        print(" | ".join(states))
        print("Time | Segment A→B | Segment B→C")
        print("-----|-------------|-------------")
        for snapshot in entry["profiles"]:
            seg_profiles = [
                _format_profile(snapshot["segments"][idx])
                for idx in range(len(scenario.stations))
            ]
            print(
                f"{snapshot['time']} | "
                + " | ".join(seg_profiles)
            )
        print()


if __name__ == "__main__":
    main()
