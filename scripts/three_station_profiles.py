#!/usr/bin/env python3
"""Dump the optimiser's ppm-vs-km output for a 3-station scenario."""

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
        length_km=30.0,
        volume_m3=18_000.0,
        stations=(
            Station("Alpha", 10.0, has_pump=True, pump_forced_on=True, has_dra=True, dra_ppm_on=5.0),
            Station("Bravo", 10.0, has_pump=True, has_dra=True, dra_ppm_on=8.0),
            Station("Charlie", 10.0, has_pump=True, has_dra=True, dra_ppm_on=10.0),
        ),
        baseline_queue=((10.0, 0.0), (10.0, 15.0), (10.0, 5.0), (10.0, 0.0)),
        dra_min_ppm=0.0,
        dra_max_ppm=10.0,
        km_per_hour=10.0,
        pump_shear_rate=1.0,
    )

    profiles = generate_combination_profiles(scenario, hours=2)

    for entry in profiles:
        states = [f"{label}: {state}" for label, state in entry.items() if label != "profiles"]
        print(" | ".join(states))
        print("Time | Alpha | Bravo | Charlie")
        print("-----|-------|-------|--------")
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
