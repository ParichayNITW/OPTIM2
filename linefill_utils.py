"""Utility functions for handling pipeline linefill tables.

Provides helpers to translate volume based linefill information into
length-wise positions along the pipeline and to update the linefill after a
certain throughput has been delivered.  Each linefill entry may carry an
optional ``dra_ppm`` field denoting the drag-reducing agent concentration in
parts-per-million for that batch.
"""

from __future__ import annotations

from math import pi
from typing import List, Dict


def linefill_lengths(linefill: List[Dict], diameter: float) -> List[Dict]:
    """Return length information for each product batch in ``linefill``.

    Parameters
    ----------
    linefill:
        List of dictionaries each describing a batch with keys ``volume``
        (m³) and optional metadata such as ``product``, ``viscosity``,
        ``density`` and ``dra_ppm``.  The first element is assumed to be the
        batch closest to the originating station.
    diameter:
        Inner diameter of the pipeline in metres.

    Returns
    -------
    List[Dict]
        ``linefill`` augmented with ``length_km`` plus ``length_km_start`` and
        ``length_km_end`` giving the occupied interval measured from the
        origin.
    """
    if diameter <= 0:
        raise ValueError("Pipe diameter must be positive")
    area = pi * (diameter ** 2) / 4.0
    result = []
    cum_len = 0.0
    for entry in linefill:
        vol = float(entry.get("volume", 0.0))
        length_km = vol / area / 1000.0
        new_entry = entry.copy()
        new_entry.update(
            {
                "length_km": length_km,
                "length_km_start": cum_len,
                "length_km_end": cum_len + length_km,
            }
        )
        result.append(new_entry)
        cum_len += length_km
    return result


def advance_linefill(linefill: List[Dict], schedule: List[Dict], delivered: float) -> List[Dict]:
    """Update ``linefill`` after ``delivered`` m³ has left the pipeline.

    The same volume is injected at the origin according to ``schedule``.  Both
    ``linefill`` and ``schedule`` are modified in-place and the updated
    ``linefill`` is returned for convenience.  Any ``dra_ppm`` values present
    on the batches are preserved and shifted along with the volumes.
    """
    remaining = delivered
    # Remove delivered volume from the terminal side (end of list)
    while remaining > 0 and linefill:
        tail = linefill[-1]
        vol = float(tail.get("volume", 0.0))
        if vol > remaining:
            tail["volume"] = vol - remaining
            remaining = 0
        else:
            remaining -= vol
            linefill.pop()
    # Inject new product batches at the origin side (front of list)
    added = delivered
    while added > 0 and schedule:
        head = schedule[0]
        vol = float(head.get("volume", 0.0))
        take = min(vol, added)
        new_batch = head.copy()
        new_batch["volume"] = take
        linefill.insert(0, new_batch)
        head["volume"] = vol - take
        if head["volume"] <= 0:
            schedule.pop(0)
        added -= take
    return linefill
