"""Helper utilities for schedule-related formatting."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def kv_rho_from_vol(
    vol_table: pd.DataFrame | Iterable[dict] | None,
    stations: list[dict],
) -> tuple[list[float], list[float], list[list[dict]]]:
    """Return viscosity, density and slice profiles for ``vol_table``.

    The helper mirrors the volumetric scheduling logic used by the Streamlit
    application.  It delegates to :func:`pipeline_optimization_app.
    map_vol_linefill_to_segments` so the returned segment slices stay in sync
    with the backend hydraulic solver expectations.

    Parameters
    ----------
    vol_table:
        Volumetric linefill description.  The value is coerced to a
        :class:`~pandas.DataFrame` when possible.  When ``None`` or empty the
        helper returns conservative defaults for all segments.
    stations:
        Sequence of station dictionaries describing the pipeline geometry.

    Returns
    -------
    tuple(list[float], list[float], list[list[dict]])
        ``kv_list`` and ``rho_list`` per segment plus ``segment_slices`` – a
        list of ``{"length_km", "kv", "rho"}`` dictionaries for each
        segment.
    """

    from pipeline_optimization_app import map_vol_linefill_to_segments

    if vol_table is None:
        df = pd.DataFrame()
    elif isinstance(vol_table, pd.DataFrame):
        df = vol_table
    else:
        df = pd.DataFrame(list(vol_table))

    return map_vol_linefill_to_segments(df, stations)


__all__ = ["kv_rho_from_vol"]
