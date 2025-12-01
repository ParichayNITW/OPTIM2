"""Generate DRA profile example for Paradip and Balasore.

This small helper mirrors the current optimiser behaviour when no
upstream DRA slug is present: each station reports the full downstream
segment as untreated (0 ppm).  It constructs the first five hourly
entries using the Paradip→Balasore (158 km) and Balasore→Haldia (170 km)
segments requested by the user.
"""

from __future__ import annotations

import pandas as pd

PARADIP_TO_BALASORE_KM = 158.0
BALASORE_TO_HALDIA_KM = 170.0


def build_zero_dra_profile(hours: int = 5) -> pd.DataFrame:
    """Return a DataFrame showing zero-ppm DRA coverage for each hour.

    When the upstream DRA queue is empty, the optimiser's profile output
    still enumerates each segment so the untreated length is explicit.
    This helper mirrors that by stamping a "0 ppm" profile across the
    segment length for both Paradip and Balasore.
    """

    if hours < 0:
        raise ValueError("hours must be non-negative")

    rows: list[dict[str, object]] = []
    for hr in range(hours):
        hour_label = f"{hr:02d}:00"
        rows.append(
            {
                "Hour": hour_label,
                "Station": "Paradip",
                "DRA Profile (km@ppm)": f"{PARADIP_TO_BALASORE_KM:.2f} km @ 0.00 ppm",
            }
        )
        rows.append(
            {
                "Hour": hour_label,
                "Station": "Balasore",
                "DRA Profile (km@ppm)": f"{BALASORE_TO_HALDIA_KM:.2f} km @ 0.00 ppm",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    profile_df = build_zero_dra_profile()
    print(profile_df.to_string(index=False))


if __name__ == "__main__":
    main()
