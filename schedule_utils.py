"""Helper utilities for schedule-related formatting."""
from __future__ import annotations

_MINUTES_PER_DAY = 24 * 60


def _coerce_to_float(value: float | int | None, default: float) -> float:
    """Return ``value`` as a float, falling back to ``default`` when invalid."""

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = float(default)
    if coerced != coerced:  # NaN check without importing math
        coerced = float(default)
    return coerced


def _format_minutes(total_minutes: int) -> str:
    """Format minutes since midnight as ``HHMM`` using 24-hour time."""

    total_minutes %= _MINUTES_PER_DAY
    hours, minutes = divmod(total_minutes, 60)
    return f"{int(hours):02d}{int(minutes):02d}"


def format_time_range(start_hour: float | int, block_hours: float | int | None = None) -> str:
    """Return an ``HHMM-HHMM`` string for a schedule interval.

    ``start_hour`` may exceed 24 or be negative; it is normalised to the 0-24h
    window. ``block_hours`` defaults to 1 hour and is coerced to a positive
    float. Both arguments may be numeric or numeric strings.
    """

    start_val = _coerce_to_float(start_hour, 0.0)
    duration = 1.0 if block_hours is None else _coerce_to_float(block_hours, 1.0)
    if duration <= 0:
        duration = 1.0

    start_minutes = int(round((start_val % 24.0) * 60.0)) % _MINUTES_PER_DAY
    duration_minutes = int(round(duration * 60.0))
    if duration_minutes <= 0:
        duration_minutes = 60
    end_minutes = (start_minutes + duration_minutes) % _MINUTES_PER_DAY

    return f"{_format_minutes(start_minutes)}-{_format_minutes(end_minutes)}"


__all__ = ["format_time_range"]
