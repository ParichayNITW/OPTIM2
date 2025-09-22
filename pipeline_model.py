"""Pipeline Optima"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Mapping
from itertools import product
import math

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba may be unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from dra_utils import get_ppm_for_dr, get_dr_for_ppm

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def head_to_kgcm2(head_m: float, rho: float) -> float:
    """Convert a head value in metres to kg/cm²."""
    return head_m * rho / 10000.0


def _km_from_volume(volume_m3: float, diameter_m: float) -> float:
    """Return the pipeline length in kilometres occupied by ``volume_m3``."""

    try:
        volume = float(volume_m3)
    except (TypeError, ValueError):
        return 0.0
    try:
        diameter = float(diameter_m)
    except (TypeError, ValueError):
        return 0.0
    if diameter <= 0:
        return 0.0
    area = math.pi * (diameter ** 2) / 4.0
    if area <= 0:
        return 0.0
    return volume / area / 1000.0


def _volume_from_km(length_km: float, diameter_m: float) -> float:
    """Return the volume in cubic metres for ``length_km`` of pipe."""

    try:
        length = float(length_km)
    except (TypeError, ValueError):
        return 0.0
    try:
        diameter = float(diameter_m)
    except (TypeError, ValueError):
        return 0.0
    if diameter <= 0:
        return 0.0
    area = math.pi * (diameter ** 2) / 4.0
    if area <= 0:
        return 0.0
    return length * 1000.0 * area


def generate_type_combinations(maxA: int = 3, maxB: int = 3) -> list[tuple[int, int]]:
    """Return all feasible pump count combinations for two pump types."""
    combos = [
        (a, b)
        for a in range(maxA + 1)
        for b in range(maxB + 1)
        if a + b > 0
    ]
    return sorted(combos, key=lambda x: (x[0] + x[1], x))


def _normalise_speed_suffix(label: str) -> str:
    """Return a normalised suffix for per-type speed fields."""

    if not isinstance(label, str):
        label = str(label or "")
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    cleaned = cleaned.strip("_")
    return cleaned.upper() if cleaned else "TYPE"


def _coerce_float(value, default: float = 0.0) -> float:
    """Convert *value* to ``float`` when possible, otherwise return ``default``."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_rpm(
    value,
    *,
    ptype: str | None = None,
    default: float = 0.0,
    prefer: str = 'min',
) -> float:
    """Return an RPM value from ``value`` handling scalars and mappings."""

    if isinstance(value, Mapping):
        if ptype is not None:
            specific = value.get(ptype)
            if specific is not None:
                return _coerce_float(specific, default)
        numeric: list[float] = []
        for val in value.values():
            if isinstance(val, Mapping):
                continue
            try:
                numeric.append(float(val))
            except (TypeError, ValueError):
                continue
        if not numeric:
            return float(default)
        if prefer == 'max':
            return max(numeric)
        if prefer == 'min':
            return min(numeric)
        return numeric[0]
    return _coerce_float(value, default)


def _station_min_rpm(
    stn: Mapping[str, object],
    ptype: str | None = None,
    default: float = 0.0,
) -> float:
    """Return the minimum permissible RPM for ``stn`` (optionally per type)."""

    return _extract_rpm(stn.get('MinRPM'), ptype=ptype, default=default, prefer='min')


def _station_max_rpm(
    stn: Mapping[str, object],
    ptype: str | None = None,
    default: float = 0.0,
) -> float:
    """Return the maximum permissible RPM for ``stn`` (optionally per type)."""

    return _extract_rpm(stn.get('DOL'), ptype=ptype, default=default, prefer='max')

# ---------------------------------------------------------------------------
# Loop enumeration utilities
# ---------------------------------------------------------------------------

def _generate_loop_cases(num_loops: int) -> list[list[int]]:
    """Return a small set of representative loop-usage combinations.

    This helper produces a variety of loop-use vectors that are independent of
    pipe diameters.  Each element in a returned list corresponds to a looped
    segment and takes one of the following values:

      * ``0`` – loop disabled (flow only through the mainline)
      * ``1`` – loop used in parallel with the mainline (flows split)
      * ``2`` – loop used in bypass mode (loop rejoins downstream of the next pump)
      * ``3`` – loop-only mode (flow only through the loopline)

    The enumeration intentionally limits the number of combinations so that
    optimisation remains tractable.  When there is only a single looped
    segment it returns four cases.  For two loops it returns the most
    relevant permutations.  When there are more than two loops the helper
    constructs a handful of representative cases: all off, all parallel,
    each individual loop used in parallel, first bypass, last bypass and
    all loop-only.
    """
    if num_loops <= 0:
        return [[]]
    # One loop: off, parallel, bypass and loop-only
    if num_loops == 1:
        return [[0], [1], [2], [3]]
    # Two loops: no-loop, both parallel, first bypass, second parallel,
    # first parallel only, and both loop-only
    if num_loops == 2:
        return [[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [3, 3]]
    # More loops: all off, all parallel, each single loop in parallel,
    # first bypass, last bypass, and all loop-only
    combos: list[list[int]] = []
    combos.append([0] * num_loops)
    combos.append([1] * num_loops)
    for i in range(num_loops):
        c = [0] * num_loops
        c[i] = 1
        combos.append(c)
    # first bypass
    c = [0] * num_loops
    c[0] = 2
    combos.append(c)
    # last bypass
    c = [0] * num_loops
    c[-1] = 2
    combos.append(c)
    combos.append([3] * num_loops)
    # Remove duplicates while preserving order
    unique: list[list[int]] = []
    for c in combos:
        if c not in unique:
            unique.append(c)
    return unique

# ---------------------------------------------------------------------------
# Custom loop-case enumeration respecting pipe diameters
# ---------------------------------------------------------------------------

def _generate_loop_cases_by_diameter(num_loops: int, equal_diameter: bool) -> list[list[int]]:
    """Generate loop usage patterns tailored to pipe diameter equality.

    When ``equal_diameter`` is ``True`` the returned cases correspond to
    combinations required by Case‑1 in the problem description: no loops,
    parallel loops on all segments and each individual loop in parallel.  For
    instance, with two loops this yields four cases: `[0, 0]`, `[1, 1]`,
    `[0, 1]` and `[1, 0]`.  Bypass and loop‑only modes are not returned
    because they are irrelevant when the loop and mainline diameters are
    identical.

    When ``equal_diameter`` is ``False`` the returned cases reflect Case‑2:
    no loops, loop‑only across the entire pipeline and a bypass case.  With
    multiple loops the bypass directive applies only to the first looped
    segment because the specification assumes that the loop bypasses the
    next pump and rejoins the mainline downstream of that station.  Additional
    loops are disabled in this case.  For a single loop this yields three
    cases: `[0]`, `[3]` and `[2]`; for two loops: `[0, 0]`, `[3, 3]` and
    `[2, 0]`.  When more than two loops exist the patterns generalise to
    `[0, 0, ...]`, `[3, 3, ...]` and `[2, 0, 0, ...]`.
    """
    if num_loops <= 0:
        return [[]]
    if equal_diameter:
        # Case‑1: only consider off/on combinations without bypass or loop-only.
        cases: list[list[int]] = []
        # All loops off
        cases.append([0] * num_loops)
        # All loops on (parallel)
        cases.append([1] * num_loops)
        # Each loop individually on
        for i in range(num_loops):
            c = [0] * num_loops
            c[i] = 1
            if c not in cases:
                cases.append(c)
        return cases
    else:
        # Case‑2: consider mainline‑only, loop‑only and bypass for first loop.
        cases: list[list[int]] = []
        # All loops off (mainline only)
        cases.append([0] * num_loops)
        # All loops loop‑only
        cases.append([3] * num_loops)
        # Bypass on first loop and others off
        c = [0] * num_loops
        c[0] = 2
        cases.append(c)
        return cases

# ---------------------------------------------------------------------------
# Fine-grained loop-case enumeration based on per-loop diameter equality
# ---------------------------------------------------------------------------

def _generate_loop_cases_by_flags(flags: list[bool]) -> list[list[int]]:
    """Return loop usage cases for pipelines with mixed diameter equality.

    ``flags`` contains one boolean per looped segment indicating whether the
    loopline diameter matches the mainline (``True``) or differs (``False``).

    Behaviour follows the two cases described in the problem statement:

    * **Case‑1** – When all flags are ``True`` the returned patterns are the
      same as :func:`_generate_loop_cases_by_diameter` with
      ``equal_diameter=True``: all loops off, all parallel and each loop
      individually in parallel.
    * **Case‑2** – When at least one flag is ``False`` the solver considers
      only three global scenarios for the differing loops: all loops off
      (mainline only), all differing loops in loop-only mode with equal-diameter
      loops disabled, and a bypass on the first differing loop with all other
      loops disabled.  Equal-diameter loops may additionally operate in
      parallel while all differing loops remain off.

    This tailored enumeration avoids invalid combinations such as bypassing
    multiple loops simultaneously or running unequal pipes in parallel.
    When ``flags`` is empty an empty pattern is returned.
    """
    if not flags:
        return [[]]
    if all(flags):
        # All loops have equal diameter → reuse simpler helper
        return _generate_loop_cases_by_diameter(len(flags), True)

    combos: list[list[int]] = []
    n = len(flags)
    # Base case: all loops off
    combos.append([0] * n)

    # Equal-diameter loops may run in parallel when differing loops are off
    eq_positions = [i for i, eq in enumerate(flags) if eq]
    if eq_positions:
        # All equal loops on in parallel
        all_eq_parallel = [1 if eq else 0 for eq in flags]
        combos.append(all_eq_parallel)
        # Each equal loop individually on
        for i in eq_positions:
            c = [0] * n
            c[i] = 1
            combos.append(c)

    # Case-2 scenarios for differing loops
    diff_positions = [i for i, eq in enumerate(flags) if not eq]
    if diff_positions:
        # All differing loops in loop-only mode (others off)
        loop_only = [3 if not eq else 0 for eq in flags]
        combos.append(loop_only)
        # Bypass only the first differing loop
        bypass_first = [0] * n
        bypass_first[diff_positions[0]] = 2
        combos.append(bypass_first)

    # Remove duplicates while preserving order
    unique: list[list[int]] = []
    for c in combos:
        if c not in unique:
            unique.append(c)
    return unique

# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

RPM_STEP = 25
DRA_STEP = 2
MAX_DRA_KM = 250.0
# Default scaling applied to the coarse search step sizes.  This multiplier
# mirrors the legacy behaviour where the coarse pass used five times the
# refinement step.
COARSE_MULTIPLIER = 5.0
# Residual head precision (decimal places) used when bucketing states during the
# dynamic-programming search.  Using a modest precision keeps the state space
# tractable while still providing near-global optimality.
RESIDUAL_ROUND = 0
V_MIN = 0.5
V_MAX = 2.5

# Limit the number of dynamic-programming states carried forward after
# each station.  ``STATE_TOP_K`` bounds the total states retained while
# ``STATE_COST_MARGIN`` allows keeping any state whose cost lies within
# this many currency units of the best state for the current station.
STATE_TOP_K = 50
STATE_COST_MARGIN = 5000.0

def _allowed_values(min_val: int, max_val: int, step: int) -> list[int]:
    if min_val > max_val:
        return [min_val]
    vals = list(range(min_val, max_val + 1, step))
    if vals[-1] != max_val:
        vals.append(max_val)
    return vals
_QUEUE_CONSUMPTION_CACHE: dict[
    tuple,
    tuple[float, tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]],
] = {}


def _prepare_dra_queue_consumption(
    queue: list[dict] | list[tuple] | tuple | None,
    segment_length: float,
    flow_m3h: float,
    hours: float,
    d_inner: float,
) -> tuple[float, tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Return pumped length along with consumed slices and downstream remainder."""

    try:
        segment_length = float(segment_length)
    except (TypeError, ValueError):
        segment_length = 0.0
    if segment_length < 0:
        segment_length = 0.0
    try:
        flow_m3h = float(flow_m3h)
    except (TypeError, ValueError):
        flow_m3h = 0.0
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        hours = 0.0
    if hours < 0:
        hours = 0.0
    try:
        d_inner = float(d_inner)
    except (TypeError, ValueError):
        d_inner = 0.0

    pumped_length_calc = _km_from_volume(flow_m3h * hours, d_inner) if d_inner > 0 else 0.0
    pumped_length = max(0.0, pumped_length_calc)

    current_queue: list[tuple[float, float]] = []
    key_entries: list[tuple[float, float]] = []
    if queue:
        for raw in queue:
            if isinstance(raw, Mapping):
                length = float(raw.get("length_km", 0.0) or 0.0)
                ppm_val = float(raw.get("dra_ppm", 0.0) or 0.0)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                length = float(raw[0] or 0.0)
                ppm_val = float(raw[1] or 0.0)
            else:
                continue
            if length <= 0:
                continue
            current_queue.append((length, ppm_val))
            key_entries.append((round(length, 6), round(ppm_val, 6)))

    cache_key = (
        tuple(key_entries),
        round(segment_length, 6),
        round(flow_m3h, 4),
        round(hours, 4),
        round(d_inner, 6),
    )
    cached = _QUEUE_CONSUMPTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    remaining = pumped_length
    incoming_slices: list[tuple[float, float]] = []
    queue_remainder: list[tuple[float, float]] = []
    for length, ppm_val in current_queue:
        length = float(length)
        ppm_val = float(ppm_val)
        if remaining > 0 and length > 0:
            take = min(length, remaining)
            if take > 0:
                incoming_slices.append((take, ppm_val))
                length -= take
                remaining -= take
        if length > 0:
            queue_remainder.append((length, ppm_val))

    if remaining > 0:
        incoming_slices.append((remaining, 0.0))

    result = (pumped_length, tuple(incoming_slices), tuple(queue_remainder))
    if len(_QUEUE_CONSUMPTION_CACHE) > 4096:
        _QUEUE_CONSUMPTION_CACHE.clear()
    _QUEUE_CONSUMPTION_CACHE[cache_key] = result
    return result


def _merge_queue(
    entries: list[tuple[float, float]] | tuple[tuple[float, float], ...]
) -> list[tuple[float, int]]:
    """Return ``entries`` with adjacent slices of equal concentration merged."""

    merged: list[tuple[float, int]] = []
    for ent in entries:
        if not ent:
            continue
        if isinstance(ent, (list, tuple)):
            length = float(ent[0] if len(ent) > 0 else 0.0)
            ppm_float = float(ent[1] if len(ent) > 1 else 0.0)
        else:
            length = float(ent)
            ppm_float = 0.0
        if length <= 0:
            continue
        ppm_val = int(round(ppm_float)) if ppm_float > 0 else 0
        if merged and merged[-1][1] == ppm_val:
            prev_len, prev_ppm = merged[-1]
            merged[-1] = (prev_len + length, prev_ppm)
        else:
            merged.append((length, ppm_val))
    return merged


def _queue_total_length(
    queue_entries: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None,
) -> float:
    """Return the cumulative treated length represented by ``queue_entries``."""

    if not queue_entries:
        return 0.0
    total = 0.0
    for length, _ppm in queue_entries:
        try:
            length_val = float(length or 0.0)
        except (TypeError, ValueError):
            length_val = 0.0
        if length_val > 0:
            total += length_val
    return total


def _trim_queue_front(
    queue_entries: list[tuple[float, float]]
    | tuple[tuple[float, float], ...],
    trim_length: float,
) -> tuple[tuple[float, float], ...]:
    """Return ``queue_entries`` shortened by ``trim_length`` from the head."""

    remaining = max(float(trim_length or 0.0), 0.0)
    if remaining <= 0:
        return tuple(
            (
                float(length),
                float(ppm),
            )
            for length, ppm in queue_entries
            if float(length or 0.0) > 0
        )

    trimmed: list[tuple[float, float]] = []
    for length, ppm in queue_entries:
        length_val = float(length or 0.0)
        if length_val <= 0:
            continue
        ppm_val = float(ppm or 0.0)
        if remaining > 0:
            if remaining >= length_val - 1e-9:
                remaining -= length_val
                continue
            leftover = length_val - remaining
            if leftover > 1e-9:
                trimmed.append((leftover, ppm_val))
            remaining = 0.0
        else:
            trimmed.append((length_val, ppm_val))

    if not trimmed:
        return ()

    merged_trimmed = _merge_queue(trimmed)
    return tuple(
        (
            float(length),
            float(ppm),
        )
        for length, ppm in merged_trimmed
        if float(length or 0.0) > 0
    )


def _take_queue_front(
    queue_entries: list[tuple[float, float]]
    | tuple[tuple[float, float], ...],
    take_length: float,
) -> tuple[tuple[float, float], ...]:
    """Return the leading ``take_length`` kilometres from ``queue_entries``."""

    remaining = max(float(take_length or 0.0), 0.0)
    if remaining <= 0:
        return ()

    taken: list[tuple[float, float]] = []
    for length, ppm in queue_entries:
        length_val = float(length or 0.0)
        if length_val <= 0:
            continue
        ppm_val = float(ppm or 0.0)
        portion = min(length_val, remaining)
        if portion > 0:
            taken.append((portion, ppm_val))
            remaining -= portion
        if remaining <= 1e-9:
            break

    if not taken:
        return ()

    return tuple(
        (
            float(length),
            float(ppm),
        )
        for length, ppm in taken
        if float(length or 0.0) > 0
    )


def _update_mainline_dra(
    queue: list[dict] | list[tuple] | tuple | None,
    stn_data: dict,
    opt: dict,
    segment_length: float,
    flow_m3h: float,
    hours: float,
    *,
    pump_running: bool = False,
    pump_shear_rate: float = 0.0,
    dra_shear_factor: float = 0.0,
    shear_injection: bool = False,
    is_origin: bool = False,
    precomputed: tuple[
        float,
        tuple[tuple[float, float], ...],
        tuple[tuple[float, float], ...],
    ] | None = None,
) -> tuple[list[tuple[float, int]], tuple[tuple[float, int], ...], int]:
    """Advance the mainline DRA queue for ``segment_length`` kilometres.

    Parameters
    ----------
    queue:
        Ordered list describing the downstream DRA distribution.  Each element
        should provide ``length_km`` and ``dra_ppm`` keys (either as a mapping
        or two-item iterable) with the head of the queue at index ``0``.
    stn_data:
        Station metadata containing at least ``d_inner`` for pumped-volume
        calculations.  Optional keys such as ``kv`` and
        ``dra_injector_position`` refine the DRA mixing behaviour.
    opt:
        Chosen operating option which must include ``dra_ppm_main`` and the
        number of operating pumps ``nop``.
    segment_length:
        Length (km) of the current segment requiring hydraulic evaluation.
    flow_m3h / hours:
        Throughput and timestep used to determine the pumped distance.
    pump_running:
        ``True`` when the station's pumps are active for this option.
    dra_shear_factor:
        Fractional reduction applied to upstream drag reduction when pumps are
        running.  Values are clamped to ``[0, 1]``.
    shear_injection:
        When ``True`` the injected slug experiences the same shear as the
        upstream fluid regardless of injector position.
    is_origin:
        ``True`` when handling the origin station.  A running origin pump with
        no injection outputs untreated fluid.

    Returns
    -------
    tuple
        ``(dra_segments, queue_after, inj_ppm_main)`` where ``dra_segments``
        is an ordered list of ``(length_km, ppm)`` describing the portion of
        the queue covering ``segment_length``.  ``queue_after`` provides the
        updated downstream queue after pumping ``flow_m3h * hours`` and
        ``inj_ppm_main`` echoes the injected concentration for reporting.
    """

    inj_ppm_main = int(opt.get("dra_ppm_main", 0) or 0)
    if not is_origin:
        idx_val = stn_data.get('idx')
        if isinstance(idx_val, (int, float)):
            is_origin = int(idx_val) == 0
    segment_length = max(float(segment_length) if segment_length is not None else 0.0, 0.0)
    flow_m3h = float(flow_m3h or 0.0)
    hours = max(float(hours or 0.0), 0.0)

    d_inner = float(stn_data.get("d_inner") or stn_data.get("d") or 0.0)

    if precomputed is None:
        pumped_length, incoming_slices, queue_remainder = _prepare_dra_queue_consumption(
            queue,
            segment_length,
            flow_m3h,
            hours,
            d_inner,
        )
    else:
        pumped_length, incoming_slices, queue_remainder = precomputed

    local_shear = max(0.0, min(float(dra_shear_factor or 0.0), 1.0))
    global_shear = max(0.0, min(float(pump_shear_rate or 0.0), 1.0))
    if pump_running:
        shear = 1.0 - (1.0 - local_shear) * (1.0 - global_shear)
    else:
        shear = local_shear
    kv = float(stn_data.get("kv", 3.0) or 3.0)
    injector_pos = str(stn_data.get("dra_injector_position", "")).lower()
    apply_injection_shear = pump_running and (shear_injection or injector_pos == "upstream")

    pumped_slices: list[tuple[float, float]] = []

    if not pump_running:
        inj_effective = float(inj_ppm_main)
        if inj_effective > 0:
            for length, base_ppm in incoming_slices:
                length = float(length)
                if length <= 0:
                    continue
                pumped_slices.append((length, float(base_ppm) + inj_effective))
        else:
            pumped_slices.extend(incoming_slices)
    else:
        inj_requested = max(float(inj_ppm_main), 0.0)
        inj_curve_ok = False
        inj_dr = 0.0
        if inj_requested > 0:
            try:
                inj_dr = float(get_dr_for_ppm(kv, inj_requested))
                inj_curve_ok = inj_dr > 0
            except Exception:
                inj_curve_ok = False
                inj_dr = 0.0
        if inj_curve_ok and inj_dr > 0 and apply_injection_shear and shear > 0:
            inj_dr *= 1.0 - shear
            if inj_dr < 0:
                inj_dr = 0.0
        if inj_curve_ok:
            if inj_dr > 0:
                try:
                    inj_effective = float(get_ppm_for_dr(kv, inj_dr))
                except Exception:
                    inj_curve_ok = False
                    inj_effective = inj_requested * (1.0 - shear if apply_injection_shear and shear > 0 else 1.0)
            else:
                inj_effective = 0.0
        else:
            inj_effective = inj_requested
            if apply_injection_shear and shear > 0 and inj_effective > 0:
                inj_effective *= 1.0 - shear
        base_curve_cache: dict[float, tuple[bool, float]] = {}
        for length, base_ppm in incoming_slices:
            if length <= 0:
                continue
            base_curve_ok = False
            base_dr = 0.0
            sheared_dr = 0.0
            if is_origin and inj_ppm_main <= 0:
                base_curve_ok = True
                base_sheared = 0.0
            else:
                base_ppm = float(base_ppm)
                if base_ppm > 0:
                    key = round(base_ppm, 6)
                    cached = base_curve_cache.get(key)
                    if cached is None:
                        try:
                            base_dr_val = float(get_dr_for_ppm(kv, base_ppm))
                            cached = (base_dr_val > 0, base_dr_val if base_dr_val > 0 else 0.0)
                        except Exception:
                            cached = (False, 0.0)
                        base_curve_cache[key] = cached
                    base_curve_ok, base_dr = cached
                else:
                    base_curve_ok, base_dr = True, 0.0
                if base_curve_ok:
                    sheared_dr = base_dr * (1.0 - shear) if shear > 0 else base_dr
                    if sheared_dr < 0:
                        sheared_dr = 0.0
                    if sheared_dr > 0:
                        try:
                            base_sheared = float(get_ppm_for_dr(kv, sheared_dr))
                        except Exception:
                            base_curve_ok = False
                            base_sheared = base_ppm * (1.0 - shear) if shear > 0 else base_ppm
                            sheared_dr = 0.0
                    else:
                        base_sheared = 0.0
                else:
                    base_sheared = base_ppm * (1.0 - shear) if shear > 0 else base_ppm
                    sheared_dr = 0.0
            if inj_ppm_main <= 0:
                combined_ppm = base_sheared
            else:
                if base_sheared <= 1e-9:
                    combined_ppm = base_sheared + inj_effective
                elif base_curve_ok and inj_curve_ok:
                    combined_dr = sheared_dr + inj_dr
                    if combined_dr > 0:
                        try:
                            combined_ppm = float(get_ppm_for_dr(kv, combined_dr))
                        except Exception:
                            combined_ppm = base_sheared + inj_effective
                    else:
                        combined_ppm = 0.0
                else:
                    combined_ppm = base_sheared + inj_effective
            pumped_slices.append((length, combined_ppm))

    segment_cover_entries: list[tuple[float, float]] = []
    downstream_entries: list[tuple[float, float]] = []
    remaining_seg = segment_length
    for length, ppm_val in pumped_slices:
        length = float(length)
        if length <= 0:
            continue
        take = 0.0
        if remaining_seg > 0:
            take = min(length, remaining_seg)
            if take > 0:
                segment_cover_entries.append((take, ppm_val))
                remaining_seg -= take
        leftover = length - take
        if leftover > 0:
            downstream_entries.append((leftover, ppm_val))

    queue_remainder_list: list[list[float]] = [
        [float(length or 0.0), float(ppm_val or 0.0)]
        for length, ppm_val in queue_remainder
        if float(length or 0.0) > 0
    ]

    if remaining_seg > 0 and queue_remainder_list:
        idx = 0
        while remaining_seg > 0 and idx < len(queue_remainder_list):
            length, ppm_val = queue_remainder_list[idx]
            take = min(length, remaining_seg)
            if take > 0:
                segment_cover_entries.append((take, ppm_val))
                queue_remainder_list[idx][0] = length - take
                remaining_seg -= take
            if queue_remainder_list[idx][0] <= 0:
                queue_remainder_list[idx][0] = 0.0
                idx += 1
            elif remaining_seg <= 0:
                break
            else:
                idx += 1

    trimmed_remainder: list[tuple[float, float]] = [
        (length, ppm_val)
        for length, ppm_val in queue_remainder_list
        if length > 0
    ]

    dra_segments: list[tuple[float, int]] = []
    for length, ppm_val in segment_cover_entries:
        if length <= 0:
            continue
        ppm_int = int(round(ppm_val)) if ppm_val > 0 else 0
        if ppm_int > 0:
            if dra_segments and abs(dra_segments[-1][1] - ppm_int) <= 0:
                prev_len, _ = dra_segments[-1]
                dra_segments[-1] = (prev_len + length, ppm_int)
            else:
                dra_segments.append((length, ppm_int))

    combined_entries: list[tuple[float, float]] = [
        (float(length), float(ppm_val))
        for length, ppm_val in (
            segment_cover_entries + downstream_entries + trimmed_remainder
        )
        if float(length) > 0
    ]

    merged_queue = _merge_queue(combined_entries)
    queue_after = [
        {'length_km': length, 'dra_ppm': ppm}
        for length, ppm in merged_queue
    ]
    return dra_segments, queue_after, inj_ppm_main


@njit(cache=True, fastmath=True)
def _segment_hydraulics(
    flow_m3h: float,
    L: float,
    d_inner: float,
    rough: float,
    kv: float,
    dra_perc: float,
    dra_length: float | None = None,
) -> tuple[float, float, float, float]:
    """Return (head_loss, velocity, reynolds, friction_factor).

    ``dra_length`` expresses the portion of the segment length ``L`` (in km)
    that experiences drag reduction.  If ``dra_length`` is ``None`` or greater
    than ``L`` the drag reduction is assumed to act over the full length.  When
    the value is ``0`` only the base friction is applied.
    """

    flow_m3h = np.float64(flow_m3h)
    L = np.float64(L)
    d_inner = np.float64(d_inner)
    rough = np.float64(rough)
    kv = np.float64(kv)
    dra_perc = np.float64(dra_perc)

    g = np.float64(9.81)
    flow_m3s = flow_m3h / np.float64(3600.0)
    area = np.pi * d_inner ** np.float64(2.0) / np.float64(4.0)
    v = flow_m3s / area if area > 0 else np.float64(0.0)
    Re = v * d_inner / (kv * np.float64(1e-6)) if kv > 0 else np.float64(0.0)
    if Re > 0:
        if Re < 4000:
            f = np.float64(64.0) / Re
        else:
            arg = (rough / d_inner / np.float64(3.7)) + (np.float64(5.74) / (Re ** np.float64(0.9)))
            f = np.float64(0.25) / (np.log10(arg) ** np.float64(2.0)) if arg > 0 else np.float64(0.0)
    else:
        f = np.float64(0.0)

    if dra_length is None:
        hl_dra = f * ((L * np.float64(1000.0)) / d_inner) * (
            v ** np.float64(2.0) / (np.float64(2.0) * g)
        ) * (1 - dra_perc / np.float64(100.0))
        return hl_dra, v, Re, f
    else:
        dlen = np.float64(dra_length)
        if dlen >= L:
            head_loss = f * ((L * np.float64(1000.0)) / d_inner) * (
                v ** np.float64(2.0) / (np.float64(2.0) * g)
            ) * (1 - dra_perc / np.float64(100.0))
        elif dlen <= np.float64(0.0):
            head_loss = f * ((L * np.float64(1000.0)) / d_inner) * (
                v ** np.float64(2.0) / (np.float64(2.0) * g)
            )
        else:
            hl_dra = f * ((dlen * np.float64(1000.0)) / d_inner) * (
                v ** np.float64(2.0) / (np.float64(2.0) * g)
            ) * (1 - dra_perc / np.float64(100.0))
            hl_nodra = f * (((L - dlen) * np.float64(1000.0)) / d_inner) * (
                v ** np.float64(2.0) / (np.float64(2.0) * g)
            )
            head_loss = hl_dra + hl_nodra

    return head_loss, v, Re, f


@njit(cache=True, fastmath=True)
def _parallel_segment_hydraulics(
    flow_m3h: float,
    main_L: float,
    main_d_inner: float,
    main_rough: float,
    main_dra: float,
    main_dra_len: float,
    loop_L: float,
    loop_d_inner: float,
    loop_rough: float,
    loop_dra: float,
    loop_dra_len: float,
    kv: float,
) -> tuple[float, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Split ``flow_m3h`` between mainline and loopline so both see equal head loss."""

    flow_m3h = np.float64(flow_m3h)
    main_L = np.float64(main_L)
    main_d_inner = np.float64(main_d_inner)
    main_rough = np.float64(main_rough)
    main_dra = np.float64(main_dra)
    main_dra_len = np.float64(main_dra_len)
    loop_L = np.float64(loop_L)
    loop_d_inner = np.float64(loop_d_inner)
    loop_rough = np.float64(loop_rough)
    loop_dra = np.float64(loop_dra)
    loop_dra_len = np.float64(loop_dra_len)
    kv = np.float64(kv)

    def calc(line_flow: float, L: float, d_inner: float, rough: float, dra: float, dra_len: float) -> tuple[float, float, float, float]:
        return _segment_hydraulics(line_flow, L, d_inner, rough, kv, dra, dra_len)

    lo = np.float64(0.0)
    hi = flow_m3h
    best = (np.float64(0.0), (np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)),
            (np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)))
    for _ in range(20):
        mid = (lo + hi) / np.float64(2.0)
        q_loop = mid
        q_main = flow_m3h - q_loop
        hl_main, v_main, Re_main, f_main = calc(q_main, main_L, main_d_inner, main_rough, main_dra, main_dra_len)
        hl_loop, v_loop, Re_loop, f_loop = calc(q_loop, loop_L, loop_d_inner, loop_rough, loop_dra, loop_dra_len)
        diff = hl_main - hl_loop
        best = (
            hl_main,
            (v_main, Re_main, f_main, q_main),
            (v_loop, Re_loop, f_loop, q_loop),
        )
        if abs(diff) < np.float64(1e-6):
            break
        if diff > 0:
            lo = mid
        else:
            hi = mid

    return best

# ---------------------------------------------------------------------------
# Multi‑segment parallel flow splitting
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _split_flow_two_segments(
    flow_m3h: float,
    kv: float,
    main1: tuple[float, float, float, float, float],
    main2: tuple[float, float, float, float, float],
    loop1: tuple[float, float, float, float, float],
    loop2: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Split ``flow_m3h`` between mainline and loopline over two segments."""

    flow_m3h = np.float64(flow_m3h)
    kv = np.float64(kv)

    lo = np.float64(0.0)
    hi = flow_m3h
    best = (np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0))
    for _ in range(25):
        q_loop = (lo + hi) / np.float64(2.0)
        q_main = flow_m3h - q_loop
        hl_m1, _, _, _ = _segment_hydraulics(q_main, main1[0], main1[1], main1[2], kv, main1[3], main1[4])
        hl_m2, _, _, _ = _segment_hydraulics(q_main, main2[0], main2[1], main2[2], kv, main2[3], main2[4])
        hl_main_total = hl_m1 + hl_m2
        hl_l1, _, _, _ = _segment_hydraulics(q_loop, loop1[0], loop1[1], loop1[2], kv, loop1[3], loop1[4])
        hl_l2, _, _, _ = _segment_hydraulics(q_loop, loop2[0], loop2[1], loop2[2], kv, loop2[3], loop2[4])
        hl_loop_total = hl_l1 + hl_l2
        diff = hl_main_total - hl_loop_total
        best = (q_main, q_loop, hl_m1, hl_l1, hl_m2, hl_l2)
        if abs(diff) < np.float64(1e-6):
            break
        if diff > 0:
            lo = q_loop
        else:
            hi = q_loop
    return best


def _pump_head(
    stn: dict,
    flow_m3h: float,
    rpm_map: Mapping[str, float | int],
    nop: int,
) -> list[dict]:
    """Return per-pump-type head and efficiency information.

    ``rpm_map`` should provide the operating speed for each pump type present
    in the station.  The return value is a list of dictionaries with keys
    ``tdh`` (total head contributed by that pump type), ``eff`` (efficiency at
    the operating point), ``count`` (number of pumps of that type),
    ``power_type`` and the original pump-type data under ``data``.  The pump
    ``ptype`` identifier and operating ``rpm`` are also included so callers can
    display detailed information for each pump type.  This allows callers to
    compute power and cost contributions for each pump type individually instead
    of relying on an averaged efficiency across all running pumps.
    """

    if nop <= 0:
        return []

    combo = (
        stn.get("active_combo")
        or stn.get("combo")
        or stn.get("pump_combo")
    )
    ptypes = stn.get("pump_types")
    results: list[dict] = []

    speed_map: dict[str, float | int]
    if rpm_map is None:
        speed_map = {}
    else:
        speed_map = dict(rpm_map)

    fallback_keys = ("*", "default", "__default__")
    numeric_values = [
        float(val)
        for val in speed_map.values()
        if isinstance(val, (int, float))
    ]
    if numeric_values:
        default_rpm = numeric_values[0]
    else:
        default_rpm = _station_max_rpm(stn)
        if default_rpm <= 0:
            default_rpm = _station_min_rpm(stn)
    default_rpm = float(default_rpm or 0.0)
    for key in fallback_keys:
        val = speed_map.get(key)
        if isinstance(val, (int, float)):
            default_rpm = float(val)
            break

    def resolve_rpm(ptype: str) -> float:
        val = speed_map.get(ptype)
        if isinstance(val, (int, float)):
            return float(val)
        for key in fallback_keys:
            val = speed_map.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return default_rpm

    if combo and ptypes:
        for ptype, count in combo.items():
            if not isinstance(count, (int, float)) or count <= 0:
                continue
            pdata = ptypes.get(ptype, {})
            rpm_val = resolve_rpm(ptype)
            if rpm_val <= 0:
                rpm_val = float(_station_max_rpm(stn, ptype=ptype) or default_rpm)
            dol = _extract_rpm(pdata.get("DOL"), default=0.0, prefer='max')
            if dol <= 0:
                dol = _station_max_rpm(stn, ptype=ptype, default=rpm_val)
            if dol <= 0:
                dol = rpm_val
            Q_equiv = flow_m3h * dol / rpm_val if rpm_val > 0 else flow_m3h
            A = pdata.get("A", 0.0)
            B = pdata.get("B", 0.0)
            C = pdata.get("C", 0.0)
            tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
            tdh_single = max(tdh_single, 0.0)
            speed_ratio_sq = (rpm_val / dol) ** 2 if dol else 0.0
            tdh_type = tdh_single * speed_ratio_sq * count
            P = pdata.get("P", 0.0)
            Qc = pdata.get("Q", 0.0)
            R = pdata.get("R", 0.0)
            S = pdata.get("S", 0.0)
            T = pdata.get("T", 0.0)
            eff_single = (
                P * Q_equiv ** 4
                + Qc * Q_equiv ** 3
                + R * Q_equiv ** 2
                + S * Q_equiv
                + T
            )
            eff_single = min(max(eff_single, 0.0), 100.0)
            results.append(
                {
                    "tdh": tdh_type,
                    "eff": eff_single,
                    "count": count,
                    "power_type": pdata.get("power_type", stn.get("power_type")),
                    "ptype": ptype,
                    "rpm": int(rpm_val),
                    "data": pdata,
                }
            )
        return results

    pump_type = stn.get("pump_type", "type1")
    rpm_single = resolve_rpm(pump_type)
    if rpm_single <= 0:
        rpm_single = float(default_rpm or _station_max_rpm(stn))
    dol = _station_max_rpm(stn, default=rpm_single if rpm_single > 0 else default_rpm)
    if dol <= 0:
        dol = rpm_single if rpm_single > 0 else default_rpm
    Q_equiv = flow_m3h * dol / rpm_single if rpm_single > 0 else flow_m3h
    A = stn.get("A", 0.0)
    B = stn.get("B", 0.0)
    C = stn.get("C", 0.0)
    tdh_single = A * Q_equiv ** 2 + B * Q_equiv + C
    tdh_single = max(tdh_single, 0.0)
    speed_ratio_sq = (rpm_single / dol) ** 2 if dol else 0.0
    tdh = tdh_single * speed_ratio_sq * nop
    P = stn.get("P", 0.0)
    Q = stn.get("Q", 0.0)
    R = stn.get("R", 0.0)
    S = stn.get("S", 0.0)
    T = stn.get("T", 0.0)
    eff = P * Q_equiv ** 4 + Q * Q_equiv ** 3 + R * Q_equiv ** 2 + S * Q_equiv + T
    eff = min(max(eff, 0.0), 100.0)
    results.append(
        {
            "tdh": tdh,
            "eff": eff,
            "count": nop,
            "power_type": stn.get("power_type"),
            "ptype": pump_type,
            "rpm": int(rpm_single),
            "data": stn,
        }
    )
    return results


def _compute_iso_sfc(pdata: dict, rpm: float, pump_bkw_total: float, rated_rpm: float, elevation: float, ambient_temp: float) -> float:
    """Compute SFC (gm/bhp-hr) using ISO 3046 approximation."""
    params = pdata.get('engine_params', {})
    rated_power = params.get('rated_power', 0.0)
    sfc50 = params.get('sfc50', 0.0)
    sfc75 = params.get('sfc75', 0.0)
    sfc100 = params.get('sfc100', 0.0)
    # Step 1: engine shaft power (kW)
    engine_kw = pump_bkw_total / 0.98 if pump_bkw_total > 0 else 0.0
    # Step 2: engine power based on operating speed
    engine_power = rated_power * (rpm / rated_rpm) if rated_rpm > 0 else 0.0
    # Step 3: determine ISO 3046 power adjustment factor (formula ref. D)
    T_ref = 298.15  # 25 °C in kelvin
    T_K = ambient_temp + 273.15
    m = 0.7
    n = 1.2
    alpha = (T_ref / T_K) ** m * np.exp(-n * elevation / 1000.0)
    engine_derated_power = engine_power * alpha
    # Step 4: load ratio
    load = engine_kw / engine_derated_power if engine_derated_power > 0 else 0.0
    load_perc = load * 100.0
    # Interpolate test bed SFC at current load
    if load_perc <= 50:
        sfc_tb = sfc50
    elif load_perc <= 75:
        sfc_tb = sfc50 + (sfc75 - sfc50) * (load_perc - 50) / 25.0
    elif load_perc <= 100:
        sfc_tb = sfc75 + (sfc100 - sfc75) * (load_perc - 75) / 25.0
    else:
        sfc_tb = sfc100
    # ISO 3046 fuel consumption adjustment factor (β) ~ 1/α for ref. D
    beta = 1.0 / alpha if alpha > 0 else 1.0
    sfc_site = sfc_tb * beta
    return sfc_site


def _build_pump_option_cache(
    stn_data: Mapping[str, object],
    opt: Mapping[str, object],
    *,
    flow_total: float,
    hours: float,
    start_time: str,
    ambient_temp: float,
    fuel_density: float,
    price_hsd: float,
) -> dict:
    """Return cached performance data for a pump operating option."""

    nop = int(opt.get('nop', 0) or 0)
    cache = {
        'pump_details': [],
        'tdh': 0.0,
        'efficiency': 0.0,
        'pump_bkw': 0.0,
        'prime_kw': 0.0,
        'power_cost': 0.0,
    }
    if nop <= 0 or flow_total <= 0:
        return cache

    pump_def = {
        'A': stn_data.get('coef_A', 0.0),
        'B': stn_data.get('coef_B', 0.0),
        'C': stn_data.get('coef_C', 0.0),
        'P': stn_data.get('coef_P', 0.0),
        'Q': stn_data.get('coef_Q', 0.0),
        'R': stn_data.get('coef_R', 0.0),
        'S': stn_data.get('coef_S', 0.0),
        'T': stn_data.get('coef_T', 0.0),
        'DOL': stn_data.get('dol', 0.0),
        'combo': stn_data.get('pump_combo'),
        'pump_types': stn_data.get('pump_types'),
        'active_combo': stn_data.get('active_combo'),
        'power_type': stn_data.get('power_type'),
        'sfc_mode': stn_data.get('sfc_mode'),
        'sfc': stn_data.get('sfc'),
        'engine_params': stn_data.get('engine_params', {}),
    }
    rpm_map_local: dict[str, float | int] = {}
    for source in (pump_def.get('rpm_map'), opt.get('rpm_map')):
        if isinstance(source, Mapping):
            for key, value in source.items():
                if isinstance(value, (int, float)):
                    rpm_map_local[key] = value
    fallback_rpm = opt.get('rpm') if isinstance(opt.get('rpm'), (int, float)) else 0
    combo_local = (
        pump_def.get('active_combo')
        or pump_def.get('combo')
        or pump_def.get('pump_combo')
    )
    if isinstance(combo_local, dict):
        for key, value in opt.items():
            if (
                isinstance(value, (int, float))
                and isinstance(key, str)
                and key.startswith('rpm_')
            ):
                ptype = key[4:]
                if ptype in combo_local:
                    rpm_map_local[ptype] = value
        for ptype, count in combo_local.items():
            if (
                isinstance(count, (int, float))
                and count > 0
                and ptype not in rpm_map_local
                and isinstance(fallback_rpm, (int, float))
            ):
                rpm_map_local[ptype] = fallback_rpm
    if (
        not rpm_map_local
        and isinstance(fallback_rpm, (int, float))
        and fallback_rpm > 0
    ):
        rpm_map_local = {'default': fallback_rpm}
    has_positive_rpm = any(
        isinstance(val, (int, float)) and val > 0 for val in rpm_map_local.values()
    )
    if has_positive_rpm:
        pump_details = _pump_head(pump_def, flow_total, rpm_map_local, nop)
    else:
        pump_details = []

    tdh = sum(p.get('tdh', 0.0) for p in pump_details)
    efficiency = (
        sum(p.get('eff', 0.0) * p.get('count', 0.0) for p in pump_details) / nop
        if pump_details
        else 0.0
    )

    pump_bkw_total = 0.0
    prime_kw_total = 0.0
    power_cost = 0.0
    rho_val = float(stn_data.get('rho', 0.0) or 0.0)
    for pinfo in pump_details:
        eff_local = max(min(pinfo.get('eff', 0.0), 100.0), 1e-6)
        tdh_local = max(pinfo.get('tdh', 0.0), 0.0)
        pump_bkw_i = (rho_val * flow_total * 9.81 * tdh_local) / (
            3600.0 * 1000.0 * (eff_local / 100.0)
        )
        pump_bkw_total += pump_bkw_i
        pdata = pinfo.get('data', {})
        rated_rpm = pdata.get('DOL', stn_data.get('dol', 0.0))
        rpm_operating = pinfo.get('rpm', opt.get('rpm', 0))
        if pinfo.get('power_type') == 'Diesel':
            mech_eff = 0.98
        else:
            mech_eff = 0.95 if rpm_operating >= rated_rpm else 0.91
        prime_kw_i = pump_bkw_i / mech_eff if mech_eff else 0.0
        prime_kw_total += prime_kw_i
        if pinfo.get('power_type') == 'Diesel':
            mode = pdata.get('sfc_mode', stn_data.get('sfc_mode', 'manual'))
            if mode == 'manual':
                sfc_val = pdata.get('sfc', stn_data.get('sfc', 0.0))
            elif mode == 'iso':
                sfc_val = _compute_iso_sfc(
                    pdata,
                    rpm_operating,
                    pump_bkw_i,
                    pdata.get('DOL', stn_data.get('dol', 0.0)),
                    stn_data.get('elev', 0.0),
                    ambient_temp,
                )
            else:
                sfc_val = 0.0
            fuel_per_kwh = (sfc_val * 1.34102) / fuel_density if sfc_val else 0.0
            cost_i = prime_kw_i * hours * fuel_per_kwh * price_hsd
        else:
            tariffs = stn_data.get('tariffs') or []
            rate_default = stn_data.get('rate', 0.0)
            remaining = hours
            cost_i = 0.0
            try:
                current = dt.datetime.strptime(start_time, "%H:%M")
            except Exception:
                current = dt.datetime(1900, 1, 1, 0, 0)
            while remaining > 0:
                applied = False
                for tr in tariffs:
                    try:
                        s = dt.datetime.strptime(tr.get('start'), "%H:%M")
                        e = dt.datetime.strptime(tr.get('end'), "%H:%M")
                    except Exception:
                        continue
                    if s <= current < e:
                        span = min((e - current).total_seconds() / 3600.0, remaining)
                        rate = float(tr.get('rate', rate_default))
                        cost_i += prime_kw_i * span * rate
                        current += dt.timedelta(hours=span)
                        remaining -= span
                        applied = True
                        break
                if not applied:
                    span = min(1.0, remaining)
                    cost_i += prime_kw_i * span * rate_default
                    current += dt.timedelta(hours=span)
                    remaining -= span
        cost_i = max(cost_i, 0.0)
        pinfo['pump_bkw'] = pump_bkw_i
        pinfo['prime_kw'] = prime_kw_i
        pinfo['power_cost'] = cost_i
        power_cost += cost_i

    cache.update(
        {
            'pump_details': pump_details,
            'tdh': tdh,
            'efficiency': efficiency,
            'pump_bkw': pump_bkw_total,
            'prime_kw': prime_kw_total,
            'power_cost': power_cost,
        }
    )
    return cache


# ---------------------------------------------------------------------------
# Downstream requirements
# ---------------------------------------------------------------------------

def _downstream_requirement(
    stations: list[dict],
    idx: int,
    terminal: dict,
    segment_flows: list[float] | None,
    KV_list: list[float],
    flow_override: float | list[float] | None = None,
    loop_usage_by_station: list[int] | None = None,
    pump_flow_overrides: dict[int, float] | None = None,
) -> int:
    """Return minimum residual head needed immediately after station ``idx``.

    The previous implementation only accumulated losses across consecutive
    non-pump stations.  When multiple pump stations appear in sequence (e.g. to
    represent different pump types at an origin), upstream pumps were unaware of
    the downstream pressure requirement and the solver could deem a feasible
    configuration infeasible.  This version performs a backward recursion over
    *all* downstream stations, subtracting the maximum head each pump can
    deliver and adding line/elevation losses for every segment.

    ``segment_flows`` may supply the flow rate after each station.  When
    ``flow_override`` is given it takes precedence and may be either a constant
    flow value or a full per-segment list.  ``loop_usage_by_station`` can be used
    to indicate whether each looped segment is active (0 means the loop is
    disabled).  ``pump_flow_overrides`` optionally maps station indices to flow
    rates used when computing pump head; this is useful when a downstream pump
    is bypassed and only the mainline flow enters the pumps.  The returned value
    is therefore the minimum residual needed after station ``idx`` so that the
    terminal residual head constraint can still be met.
    """

    from functools import lru_cache

    N = len(stations)
    if flow_override is not None:
        if isinstance(flow_override, list):
            flows = flow_override
        else:
            flows = [flow_override] * (N + 1)
    else:
        if segment_flows is None:
            raise ValueError("segment_flows or flow_override must be provided")
        flows = segment_flows

    @lru_cache(None)
    def req_entry(i: int) -> int:
        if i >= N:
            return int(terminal.get('min_residual', 0))
        stn = stations[i]
        kv = KV_list[i]
        # ``flows`` holds the flow rate *after* each station; use the downstream
        # value so losses reflect the correct segment flow between station ``i``
        # and ``i+1``.
        flow = flows[i + 1]
        L = stn.get('L', 0.0)
        t = stn.get('t', 0.007)
        # ``D`` may be present but ``None`` in pump-type expansions.  Treat
        # a ``None`` value as absent and fall back to the ``d`` key.  Without
        # this check subtraction would error.
        if stn.get('D') is not None:
            d_inner = stn['D'] - 2 * t
        else:
            d_inner = stn.get('d', 0.7) - 2 * t
        rough = stn.get('rough', 0.00004)
        dra_down = stn.get('max_dr', 0)

        head_loss, *_ = _segment_hydraulics(flow, L, d_inner, rough, kv, dra_down, None)
        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N - 1 else stations[i + 1].get('elev', 0.0)
        downstream = req_entry(i + 1)
        req = downstream + head_loss + (elev_next - elev_i)

        # Check intermediate peaks on both mainline and loopline.  Each peak
        # requires sufficient upstream pressure to maintain at least 25 m of
        # residual head at the peak itself.  Consider whichever peak demands the
        # highest pressure.
        # Helper to compute the residual head requirement at intermediate peaks.
        # ``flow_rate`` is the volumetric flow rate (m³/h) used to compute friction to the peak.
        def peak_requirement(flow_rate: float, peaks, d_pipe: float, rough_pipe: float, dra_perc: float) -> float:
            req_local = 0.0
            for peak in peaks or []:
                # Peak location can be stored under various keys
                dist = peak.get('loc') or peak.get('Location (km)') or peak.get('Location')
                elev_peak = peak.get('elev') or peak.get('Elevation (m)') or peak.get('Elevation')
                if dist is None or elev_peak is None:
                    continue
                head_peak, *_ = _segment_hydraulics(flow_rate, float(dist), d_pipe, rough_pipe, kv, dra_perc)
                req_p = head_peak + (float(elev_peak) - elev_i) + 25.0
                if req_p > req_local:
                    req_local = req_p
            return req_local

        # Compute peak requirement on the mainline using downstream flow ``flow``.
        peak_req_main = peak_requirement(flow, stn.get('peaks'), d_inner, rough, dra_down)
        peak_req = peak_req_main
        # Compute peak requirement on the loopline.  When the loop carries flow beyond this station
        # (e.g. under bypass), we conservatively use the upstream flow ``flows[i]`` to estimate
        # friction to the peak.  This avoids underestimating the head needed at peaks on the 18" line.
        loop = stn.get('loopline')
        usage = loop_usage_by_station[i] if loop_usage_by_station and i < len(loop_usage_by_station) else None
        loop_flow = flows[i] if usage != 0 else 0.0
        if loop and usage != 0:
            if loop.get('D') is not None:
                t_loop = loop.get('t', t)
                d_inner_loop = loop['D'] - 2 * t_loop
            else:
                d_inner_loop = loop.get('d', d_inner)
            rough_loop = loop.get('rough', rough)
            dra_loop = loop.get('max_dr', 0.0)
            # Use the upstream flow ``flows[i]`` for loop peaks to account for bypassed flow.
            peak_req_loop = peak_requirement(loop_flow, loop.get('peaks'), d_inner_loop, rough_loop, dra_loop)
            peak_req = max(peak_req_main, peak_req_loop)
        req = max(req, peak_req)

        if stn.get('is_pump', False):
            rpm_max_val = _station_max_rpm(stn)
            if rpm_max_val <= 0:
                rpm_max_val = _station_min_rpm(stn)
            rpm_max = int(max(rpm_max_val, 0))
            nop_max = stn.get('max_pumps', 0)
            flow_pump = pump_flow_overrides.get(i, flow) if pump_flow_overrides else flow
            if rpm_max and nop_max:
                rpm_map_src = stn.get('rpm_map')
                if isinstance(rpm_map_src, Mapping):
                    rpm_map_local = dict(rpm_map_src)
                else:
                    rpm_map_local = {}
                if not rpm_map_local:
                    combo_local = (
                        stn.get('active_combo')
                        or stn.get('combo')
                        or stn.get('pump_combo')
                    )
                    if isinstance(combo_local, dict):
                        for ptype, count in combo_local.items():
                            if isinstance(count, (int, float)) and count > 0:
                                rpm_map_local[ptype] = int(
                                    max(
                                        _station_max_rpm(stn, ptype=ptype, default=rpm_max)
                                        or rpm_max,
                                        0,
                                    )
                                )
                if not rpm_map_local:
                    pump_type = stn.get('pump_type')
                    if pump_type:
                        rpm_map_local[pump_type] = rpm_max
                if not rpm_map_local:
                    rpm_map_local = {'default': rpm_max}
                pump_info = _pump_head(stn, flow_pump, rpm_map_local, nop_max)
                tdh_max = sum(max(p['tdh'], 0.0) for p in pump_info)
            else:
                tdh_max = 0.0
            req -= tdh_max
        req = max(req, int(stn.get('min_residual', 0)))
        return int(round(req))
    return int(req_entry(idx + 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_pipeline(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill: list[dict] | dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
    start_time: str = "00:00",
    pump_shear_rate: float = 0.0,
    *,
    loop_usage_by_station: list[int] | None = None,
    enumerate_loops: bool = True,
    _internal_pass: bool = False,
    rpm_step: int = RPM_STEP,
    dra_step: int = DRA_STEP,
    narrow_ranges: dict[int, dict[str, tuple[int, int]]] | None = None,
    coarse_multiplier: float = COARSE_MULTIPLIER,
    state_top_k: int = STATE_TOP_K,
    state_cost_margin: float = STATE_COST_MARGIN,
) -> dict:
    """Enumerate feasible options across all stations to find the lowest-cost
    operating strategy.

    ``linefill`` describes the current batches in the pipeline as a sequence of
    dictionaries with at least ``volume`` (m³) and ``dra_ppm`` keys.  The
    leading batch's concentration is used as the upstream DRA level for the
    first station.  The function returns the updated linefill after pumping
    under the key ``"linefill"``.

    ``pump_shear_rate`` applies a uniform fractional attenuation to any DRA
    slug passing through an active pump.  The value is clamped to the
    interval ``[0, 1]`` and combines with per-station shear factors when
    present.

    The solver operates in two passes.  A coarse search first evaluates
    the pipeline using large step sizes for pump speed and drag-reduction
    percentage to identify a near‑optimal operating point.  A refinement
    pass then narrows the RPM and DRA ranges around that coarse solution
    and re-solves using the user-provided ``rpm_step`` and ``dra_step``.
    ``narrow_ranges`` is an internal helper used to restrict the values
    considered during the refinement stage.

    This function supports optional loop-use directives.  When
    ``enumerate_loops`` is ``True`` and no explicit
    ``loop_usage_by_station`` is provided the solver will automatically
    build a small set of loop-use patterns (e.g. Cases A–E) and run the
    optimisation for each.  The best result is returned.  When
    ``loop_usage_by_station`` is supplied the solver restricts which
    loop scenarios are considered at each station: 0=disabled, 1=parallel,
    2=bypass.  By default the function behaves like the original
    implementation with internal loop enumeration.

    Advanced callers can tune the search breadth using ``rpm_step`` and
    ``dra_step`` for the refinement pass and ``coarse_multiplier`` to scale
    the coarse pass step sizes.  Increasing ``state_top_k`` or
    ``state_cost_margin`` relaxes dynamic-programming pruning so more near-
    optimal states are retained for subsequent stations.  When these
    parameters are omitted the legacy defaults are used.
    """

    try:
        pump_shear_rate = float(pump_shear_rate)
    except (TypeError, ValueError):
        pump_shear_rate = 0.0
    pump_shear_rate = max(0.0, min(pump_shear_rate, 1.0))

    try:
        rpm_step = int(rpm_step)
    except (TypeError, ValueError):
        rpm_step = RPM_STEP
    if rpm_step <= 0:
        rpm_step = RPM_STEP

    try:
        dra_step = int(dra_step)
    except (TypeError, ValueError):
        dra_step = DRA_STEP
    if dra_step <= 0:
        dra_step = DRA_STEP

    try:
        coarse_multiplier = float(coarse_multiplier)
    except (TypeError, ValueError):
        coarse_multiplier = COARSE_MULTIPLIER
    if coarse_multiplier <= 0:
        coarse_multiplier = COARSE_MULTIPLIER

    try:
        state_top_k = int(state_top_k)
    except (TypeError, ValueError):
        state_top_k = STATE_TOP_K
    if state_top_k <= 0:
        state_top_k = STATE_TOP_K

    try:
        state_cost_margin = float(state_cost_margin)
    except (TypeError, ValueError):
        state_cost_margin = STATE_COST_MARGIN
    if state_cost_margin < 0:
        state_cost_margin = 0.0

    # When requested, perform an outer enumeration over loop usage patterns.
    # We only enter this branch when no explicit per-station loop usage is
    # specified.  Each candidate pattern is mapped onto the stations with
    # looplines, then the solver is invoked recursively with
    # ``enumerate_loops=False`` so that this block isn't re-entered.  The
    # best feasible result across all cases is returned.
    if enumerate_loops and loop_usage_by_station is None:
        # Identify the indices of stations with defined looplines
        loop_positions = [idx for idx, stn in enumerate(stations) if stn.get('loopline')]
        num_loops = len(loop_positions)
        # If there are no looped segments simply call solve_pipeline once
        if num_loops == 0:
            return solve_pipeline(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill,
                dra_reach_km,
                mop_kgcm2,
                hours,
                start_time,
                pump_shear_rate=pump_shear_rate,
                loop_usage_by_station=[],
                enumerate_loops=False,
                rpm_step=rpm_step,
                dra_step=dra_step,
                coarse_multiplier=coarse_multiplier,
                state_top_k=state_top_k,
                state_cost_margin=state_cost_margin,
            )
        # Determine per-loop diameter equality flags.  For each looped
        # segment compute whether the inner diameters of the mainline and
        # loopline match within a small tolerance.  This allows the
        # optimiser to apply Case‑1 logic on loops with equal pipes and
        # Case‑2 logic on those with differing pipes independently.
        default_t_local = 0.007
        flags: list[bool] = []
        for idx in loop_positions:
            stn = stations[idx]
            # Inner diameter of mainline
            if stn.get('D') is not None:
                d_main_outer = stn['D']
                t_main = stn.get('t', default_t_local)
                d_inner_main = d_main_outer - 2 * t_main
            else:
                # When only an inner diameter is given treat it as inner
                d_inner_main = stn.get('d', 0.0)
            loop = stn.get('loopline') or {}
            if loop:
                if loop.get('D') is not None:
                    d_loop_outer = loop['D']
                    t_loop = loop.get('t', stn.get('t', default_t_local))
                    d_inner_loop = d_loop_outer - 2 * t_loop
                else:
                    d_inner_loop = loop.get('d', d_inner_main)
            else:
                # Should not happen as only stations with loopline are in loop_positions
                d_inner_loop = d_inner_main
            flags.append(abs(d_inner_main - d_inner_loop) <= 1e-6)
        # Generate loop-usage patterns based on per-loop diameter equality
        cases = _generate_loop_cases_by_flags(flags)
        best_res: dict | None = None
        for case in cases:
            usage = [0] * len(stations)
            for pos, val in zip(loop_positions, case):
                usage[pos] = val
            res = solve_pipeline(
                stations,
                terminal,
                FLOW,
                KV_list,
                rho_list,
                RateDRA,
                Price_HSD,
                Fuel_density,
                Ambient_temp,
                linefill,
                dra_reach_km,
                mop_kgcm2,
                hours,
                start_time,
                pump_shear_rate=pump_shear_rate,
                loop_usage_by_station=usage,
                enumerate_loops=False,
                rpm_step=rpm_step,
                dra_step=dra_step,
                coarse_multiplier=coarse_multiplier,
                state_top_k=state_top_k,
                state_cost_margin=state_cost_margin,
            )
            if res.get('error'):
                continue
            if best_res is None or res.get('total_cost', float('inf')) < best_res.get('total_cost', float('inf')):
                # Track which loop usage produced the best result.  Store a
                # copy to avoid mutating the result of nested calls.  Users
                # can inspect this field to derive human‑friendly names.
                res_with_usage = res.copy()
                res_with_usage['loop_usage'] = usage.copy()
                best_res = res_with_usage
        return best_res or {
            'error': True,
            'message': 'No feasible pump combination found for stations.',
        }
    # Normalise linefill input into a list of batches each carrying volume and
    # DRA concentration.  Accepts either a list of dictionaries or a dict of
    # columns as produced by ``DataFrame.to_dict()``.  The linefill is copied
    # so callers are not mutated.
    linefill_state: list[dict] = []
    if linefill:
        if isinstance(linefill, dict):
            vols = linefill.get('volume') or linefill.get('Volume (m³)') or linefill.get('Volume')
            ppms = linefill.get('dra_ppm') or linefill.get('DRA ppm') or {}
            if vols is not None:
                items = vols.items() if isinstance(vols, dict) else enumerate(vols)
                for idx, v in items:
                    try:
                        vol = float(v)
                    except Exception:
                        continue
                    if vol <= 0:
                        continue
                    if isinstance(ppms, dict):
                        ppm_val = ppms.get(idx, 0.0)
                    elif isinstance(ppms, list):
                        ppm_val = ppms[idx] if idx < len(ppms) else 0.0
                    else:
                        ppm_val = 0.0
                    try:
                        ppm = int(float(ppm_val))
                    except Exception:
                        ppm = 0
                    linefill_state.append({'volume': vol, 'dra_ppm': ppm})
        elif isinstance(linefill, list):
            for ent in linefill:
                try:
                    vol = float(ent.get('volume') or ent.get('Volume (m³)') or ent.get('Volume') or 0.0)
                except Exception:
                    continue
                if vol <= 0:
                    continue
                try:
                    ppm = int(float(ent.get('dra_ppm') or ent.get('DRA ppm') or 0.0))
                except Exception:
                    ppm = 0
                linefill_state.append({'volume': vol, 'dra_ppm': ppm})
    linefill_state = copy.deepcopy(linefill_state)

    N = len(stations)

    # ------------------------------------------------------------------
    # Two-pass optimisation: first run a coarse search with enlarged
    # step sizes to find a near-optimal solution, then refine around that
    # solution using the user-provided steps.  The recursion is controlled
    # by the ``_internal_pass`` flag to avoid infinite loops.
    # ------------------------------------------------------------------
    if not _internal_pass:
        coarse_scale = coarse_multiplier
        coarse_rpm_step = int(round(rpm_step * coarse_scale)) if rpm_step > 0 else int(round(coarse_scale))
        if coarse_rpm_step <= 0:
            coarse_rpm_step = rpm_step if rpm_step > 0 else 1
        if coarse_scale >= 1.0 and rpm_step > 0:
            coarse_rpm_step = max(coarse_rpm_step, rpm_step)

        coarse_dra_step = int(round(dra_step * coarse_scale)) if dra_step > 0 else int(round(coarse_scale))
        if coarse_dra_step <= 0:
            coarse_dra_step = dra_step if dra_step > 0 else 1
        if coarse_scale >= 1.0 and dra_step > 0:
            coarse_dra_step = max(coarse_dra_step, dra_step)
        coarse_res = solve_pipeline(
            stations,
            terminal,
            FLOW,
            KV_list,
            rho_list,
            RateDRA,
            Price_HSD,
            Fuel_density,
            Ambient_temp,
            linefill,
            dra_reach_km,
            mop_kgcm2,
            hours,
            start_time,
            pump_shear_rate=pump_shear_rate,
            loop_usage_by_station=loop_usage_by_station,
            enumerate_loops=False,
            _internal_pass=True,
            rpm_step=coarse_rpm_step,
            dra_step=coarse_dra_step,
            coarse_multiplier=coarse_multiplier,
            state_top_k=state_top_k,
            state_cost_margin=state_cost_margin,
        )
        if coarse_res.get("error"):
            return coarse_res
        window = max(rpm_step, coarse_rpm_step)
        ranges: dict[int, dict[str, tuple[int, int]]] = {}
        for idx, stn in enumerate(stations):
            name = stn["name"].strip().lower().replace(" ", "_")
            if stn.get("is_pump", False):
                coarse_nop = int(coarse_res.get(f"num_pumps_{name}", 0))
                coarse_dr_main = int(coarse_res.get(f"drag_reduction_{name}", 0))
                st_rpm_min = int(_station_min_rpm(stn))
                st_rpm_max = int(_station_max_rpm(stn))
                if st_rpm_max <= 0 and st_rpm_min > 0:
                    st_rpm_max = st_rpm_min
                if coarse_nop == 0:
                    rmin = rmax = 0
                else:
                    coarse_rpm = int(coarse_res.get(f"speed_{name}", st_rpm_min))
                    rmin = max(st_rpm_min, coarse_rpm - window)
                    upper_bound = st_rpm_max if st_rpm_max > 0 else st_rpm_min
                    rmax = min(upper_bound, coarse_rpm + window)
                max_dr_main = int(stn.get("max_dr", 0))
                if coarse_dr_main <= 0 or coarse_dr_main >= max_dr_main:
                    dmin, dmax = 0, max_dr_main
                else:
                    dmin = max(0, coarse_dr_main - dra_step)
                    dmax = min(max_dr_main, coarse_dr_main + dra_step)
                entry: dict[str, tuple[int, int]] = {
                    "rpm": (rmin, rmax),
                    "dra_main": (dmin, dmax),
                }
                pump_types_rng = stn.get("pump_types") if isinstance(stn.get("pump_types"), Mapping) else None
                combo_rng = None
                if isinstance(stn.get("active_combo"), Mapping):
                    combo_rng = stn["active_combo"]  # type: ignore[index]
                elif isinstance(stn.get("pump_combo"), Mapping):
                    combo_rng = stn["pump_combo"]  # type: ignore[index]
                elif isinstance(stn.get("combo"), Mapping):
                    combo_rng = stn["combo"]  # type: ignore[index]
                if pump_types_rng and isinstance(combo_rng, Mapping):
                    for ptype, count in combo_rng.items():
                        if not isinstance(count, (int, float)) or count <= 0:
                            continue
                        pdata = pump_types_rng.get(ptype, {})
                        pmin_default = int(_station_min_rpm(stn, ptype=ptype, default=st_rpm_min))
                        pmax_default = int(_station_max_rpm(stn, ptype=ptype, default=st_rpm_max or st_rpm_min))
                        p_rmin = int(_extract_rpm(pdata.get("MinRPM"), default=pmin_default, prefer='min'))
                        p_rmax = int(_extract_rpm(pdata.get("DOL"), default=pmax_default, prefer='max'))
                        if p_rmax <= 0 and pmax_default > 0:
                            p_rmax = pmax_default
                        if p_rmax < p_rmin:
                            p_rmin, p_rmax = p_rmax, p_rmin
                        coarse_type_rpm: int | None = None
                        if coarse_nop > 0:
                            suffix = _normalise_speed_suffix(ptype)
                            coarse_key = f"speed_{name}_{suffix}"
                            coarse_val = coarse_res.get(coarse_key)
                            if coarse_val is not None:
                                coarse_type_rpm = int(round(_coerce_float(coarse_val, default=0.0)))
                            if coarse_type_rpm is None or coarse_type_rpm <= 0:
                                details_key = f"pump_details_{name}"
                                details_val = coarse_res.get(details_key)
                                if isinstance(details_val, list):
                                    for detail in details_val:
                                        if not isinstance(detail, Mapping):
                                            continue
                                        detail_ptype = detail.get('ptype')
                                        detail_str = str(detail_ptype) if detail_ptype is not None else ""
                                        target_str = str(ptype)
                                        detail_suffix = _normalise_speed_suffix(detail_str) if detail_str else ""
                                        if detail_str != target_str and detail_suffix != suffix:
                                            continue
                                        rpm_val = detail.get('rpm')
                                        coarse_candidate = int(round(_coerce_float(rpm_val, default=0.0)))
                                        if coarse_candidate > 0:
                                            coarse_type_rpm = coarse_candidate
                                            break
                        if coarse_type_rpm is not None and coarse_type_rpm > 0:
                            lower_bound = max(p_rmin, coarse_type_rpm - window)
                            upper_bound = min(p_rmax, coarse_type_rpm + window)
                            if upper_bound >= lower_bound:
                                p_rmin, p_rmax = lower_bound, upper_bound
                        entry[f"rpm_{ptype}"] = (p_rmin, p_rmax)
                loop = stn.get("loopline") or {}
                if loop:
                    coarse_dr_loop = int(coarse_res.get(f"drag_reduction_loop_{name}", 0))
                    lmin = max(0, coarse_dr_loop - dra_step)
                    lmax = min(int(loop.get("max_dr", 0)), coarse_dr_loop + dra_step)
                    entry["dra_loop"] = (lmin, lmax)
                ranges[idx] = entry
            else:
                coarse_dr_main = int(coarse_res.get(f"drag_reduction_{name}", 0))
                dmin = max(0, coarse_dr_main - dra_step)
                dmax = min(int(stn.get("max_dr", 0)), coarse_dr_main + dra_step)
                ranges[idx] = {"dra_main": (dmin, dmax)}
        return solve_pipeline(
            stations,
            terminal,
            FLOW,
            KV_list,
            rho_list,
            RateDRA,
            Price_HSD,
            Fuel_density,
            Ambient_temp,
            linefill,
            dra_reach_km,
            mop_kgcm2,
            hours,
            start_time,
            pump_shear_rate=pump_shear_rate,
            loop_usage_by_station=loop_usage_by_station,
            enumerate_loops=False,
            _internal_pass=True,
            rpm_step=rpm_step,
            dra_step=dra_step,
            narrow_ranges=ranges,
            coarse_multiplier=coarse_multiplier,
            state_top_k=state_top_k,
            state_cost_margin=state_cost_margin,
        )

    # -----------------------------------------------------------------------
    # Sanitize viscosity (KV_list) and density (rho_list) inputs
    #
    # In some scenarios the caller may provide ``KV_list`` or ``rho_list``
    # entries that are zero or ``None``.  A zero viscosity would result in a
    # division by zero when computing Reynolds numbers and friction factors, and
    # a zero density will preclude converting heads to pressure or computing
    # hydraulic power.  Such values frequently arise when the upstream UI has
    # no linefill information and defaults all entries to zero.  To ensure the
    # optimisation can progress we substitute conservative defaults when
    # encountering these values.  The defaults represent a moderately light
    # refined product at 25 °C: 1.0 cSt (~1×10⁻⁶ m²/s) for viscosity and
    # 850 kg/m³ for density.  Negative values are also treated as invalid.
    #
    KV_list = [float(kv) if (kv is not None and kv > 0) else 1.0 for kv in KV_list]
    rho_list = [float(rho) if (rho is not None and rho > 0) else 850.0 for rho in rho_list]
    segment_flows = [float(FLOW)]
    for stn in stations:
        delivery = float(stn.get('delivery', 0.0))
        supply = float(stn.get('supply', 0.0))
        prev_flow = segment_flows[-1]
        segment_flows.append(prev_flow - delivery + supply)

    default_t = 0.007
    default_e = 0.00004

    # Pre-compute static data for each station; head losses depend on DRA reach
    station_opts = []
    origin_enforced = False
    cum_dist = 0.0
    origin_diameter = 0.0
    for i, stn in enumerate(stations, start=1):
        name = stn['name'].strip().lower().replace(' ', '_')
        flow = segment_flows[i]
        kv = KV_list[i - 1]
        rho = rho_list[i - 1]

        L = stn.get('L', 0.0)
        # Determine pipe dimensions.  Some pump-type expansions may carry a
        # ``D`` key with value ``None``.  Treat a ``None`` diameter as
        # unspecified and fall back to using ``d`` instead.  Likewise,
        # thickness defaults when not provided.  Without this check,
        # ``stn['D']`` could be ``None`` and arithmetic would raise an error.
        if stn.get('D') is not None:
            thickness = stn.get('t', default_t)
            # ``outer_d`` may be ``None`` if ``D`` exists but is explicitly
            # null.  Guard against this by falling back to the internal
            # diameter ``d`` if provided, otherwise the default 0.7 m.
            outer_d = stn['D'] if stn['D'] is not None else stn.get('d', 0.7)
            d_inner = outer_d - 2 * thickness
        else:
            # When ``D`` is absent or ``None`` fall back to ``d``
            d_inner = stn.get('d', 0.7)
            outer_d = d_inner
            thickness = stn.get('t', default_t)
        if i == 1:
            try:
                origin_diameter = float(d_inner)
            except (TypeError, ValueError):
                origin_diameter = 0.0
            if origin_diameter < 0:
                origin_diameter = 0.0
        rough = stn.get('rough', default_e)

        # Use a default SMYS when the station provides ``None`` or omits the
        # parameter entirely.  A value of ``None`` would propagate and
        # break downstream multiplication.
        SMYS = stn.get('SMYS', 52000.0) or 52000.0
        design_factor = 0.72
        maop_psi = 2 * SMYS * design_factor * (thickness / outer_d) if outer_d > 0 else 0.0
        maop_kgcm2 = maop_psi * 0.0703069
        if mop_kgcm2 is not None:
            maop_kgcm2 = min(maop_kgcm2, float(mop_kgcm2))
        maop_head = maop_kgcm2 * 10000.0 / rho if rho > 0 else 0.0

        loop_info = stn.get('loopline')
        loop_dict = None
        if loop_info:
            L_loop = loop_info.get('L', L)
            if 'D' in loop_info:
                t_loop = loop_info.get('t', default_t)
                d_inner_loop = loop_info['D'] - 2 * t_loop
                outer_loop = loop_info['D']
            else:
                d_inner_loop = loop_info.get('d', d_inner)
                outer_loop = loop_info.get('d', outer_d)
                t_loop = loop_info.get('t', default_t)
            rough_loop = loop_info.get('rough', default_e)
            SMYS_loop = loop_info.get('SMYS', SMYS)
            maop_psi_loop = 2 * SMYS_loop * design_factor * (t_loop / outer_loop) if outer_loop > 0 else 0.0
            maop_kg_loop = maop_psi_loop * 0.0703069
            if mop_kgcm2 is not None:
                maop_kg_loop = min(maop_kg_loop, float(mop_kgcm2))
            maop_head_loop = maop_kg_loop * 10000.0 / rho if rho > 0 else 0.0
            loop_dict = {
                'name': loop_info.get('name', ''),
                'L': L_loop,
                'd_inner': d_inner_loop,
                'rough': rough_loop,
                'max_dr': loop_info.get('max_dr', 0.0),
                'maop_head': maop_head_loop,
                'maop_kgcm2': maop_kg_loop,
            }

        elev_i = stn.get('elev', 0.0)
        elev_next = terminal.get('elev', 0.0) if i == N else stations[i].get('elev', 0.0)
        elev_delta = elev_next - elev_i

        opts = []
        flow_m3s = flow / 3600.0
        area = np.pi * d_inner ** 2 / 4.0
        if stn.get('is_pump', False):
            min_p = stn.get('min_pumps', 0)
            if not origin_enforced:
                min_p = max(1, min_p)
                origin_enforced = True
            max_p = stn.get('max_pumps', 2)
            rng = narrow_ranges.get(i - 1) if narrow_ranges else None
            station_rpm_min = int(_station_min_rpm(stn))
            station_rpm_max = int(_station_max_rpm(stn))
            if station_rpm_max <= 0 and station_rpm_min > 0:
                station_rpm_max = station_rpm_min
            rpm_min = station_rpm_min
            rpm_max = station_rpm_max
            if rng and 'rpm' in rng:
                rpm_min = max(rpm_min, rng['rpm'][0])
                rpm_max = min(rpm_max, rng['rpm'][1])
            rpm_vals = _allowed_values(rpm_min, rpm_max, rpm_step)

            pump_types_data = stn.get('pump_types') if isinstance(stn.get('pump_types'), Mapping) else None
            combo_source: Mapping[str, float] | None = None
            if isinstance(stn.get('active_combo'), Mapping):
                combo_source = stn['active_combo']  # type: ignore[index]
            elif isinstance(stn.get('pump_combo'), Mapping):
                combo_source = stn['pump_combo']  # type: ignore[index]
            elif isinstance(stn.get('combo'), Mapping):
                combo_source = stn['combo']  # type: ignore[index]

            type_order: list[str] = []
            type_rpm_lists: dict[str, list[int]] = {}
            if pump_types_data and combo_source:
                for ptype in sorted(combo_source):
                    count = combo_source.get(ptype, 0)
                    if not isinstance(count, (int, float)) or count <= 0:
                        continue
                    pdata = pump_types_data.get(ptype, {})
                    p_rpm_min = int(
                        _extract_rpm(
                            pdata.get('MinRPM'),
                            default=_station_min_rpm(stn, ptype=ptype, default=rpm_min),
                            prefer='min',
                        )
                    )
                    p_rpm_max = int(
                        _extract_rpm(
                            pdata.get('DOL'),
                            default=_station_max_rpm(stn, ptype=ptype, default=rpm_max),
                            prefer='max',
                        )
                    )
                    if p_rpm_max <= 0 and rpm_max > 0:
                        p_rpm_max = rpm_max
                    if rng:
                        key = f'rpm_{ptype}'
                        if key in rng:
                            p_rpm_min = max(p_rpm_min, rng[key][0])
                            p_rpm_max = min(p_rpm_max, rng[key][1])
                        elif 'rpm' in rng:
                            p_rpm_min = max(p_rpm_min, rng['rpm'][0])
                            p_rpm_max = min(p_rpm_max, rng['rpm'][1])
                    type_order.append(ptype)
                    type_rpm_lists[ptype] = _allowed_values(p_rpm_min, p_rpm_max, rpm_step)

            fixed_dr = stn.get('fixed_dra_perc', None)
            max_dr_main = int(stn.get('max_dr', 0))
            if fixed_dr is not None:
                dra_main_vals = [int(round(fixed_dr))]
            else:
                dr_min, dr_max = 0, max_dr_main
                if rng and 'dra_main' in rng:
                    dr_min = max(0, rng['dra_main'][0])
                    dr_max = min(max_dr_main, rng['dra_main'][1])
                dra_main_vals = _allowed_values(dr_min, dr_max, dra_step)
            max_dr_loop = int(loop_dict.get('max_dr', 0)) if loop_dict else 0
            dr_loop_min, dr_loop_max = 0, max_dr_loop
            if rng and 'dra_loop' in rng:
                dr_loop_min = max(0, rng['dra_loop'][0])
                dr_loop_max = min(max_dr_loop, rng['dra_loop'][1])
            dra_loop_vals = _allowed_values(dr_loop_min, dr_loop_max, dra_step) if loop_dict else [0]
            for nop in range(min_p, max_p + 1):
                if nop == 0:
                    rpm_iter = [None]
                elif type_rpm_lists:
                    rpm_iter = product(*(type_rpm_lists[ptype] for ptype in type_order))
                else:
                    rpm_iter = [(rpm,) for rpm in rpm_vals]
                for rpm_choice in rpm_iter:
                    if nop == 0:
                        rpm = 0
                        rpm_map_choice: dict[str, int] = {}
                    elif type_rpm_lists:
                        if isinstance(rpm_choice, tuple):
                            rpm_map_choice = {
                                ptype: int(val)
                                for ptype, val in zip(type_order, rpm_choice)
                            }
                        else:
                            rpm_map_choice = {}
                        rpm = max(rpm_map_choice.values()) if rpm_map_choice else 0
                    else:
                        rpm = int(rpm_choice[0]) if isinstance(rpm_choice, tuple) else int(rpm_choice)
                        rpm_map_choice = {}
                    for dra_main in dra_main_vals:
                        for dra_loop in dra_loop_vals:
                            ppm_main = int(get_ppm_for_dr(kv, dra_main)) if dra_main > 0 else 0
                            ppm_loop = int(get_ppm_for_dr(kv, dra_loop)) if dra_loop > 0 else 0
                            opt_entry = {
                                'nop': nop,
                                'rpm': rpm,
                                'dra_main': dra_main,
                                'dra_loop': dra_loop,
                                'dra_ppm_main': ppm_main,
                                'dra_ppm_loop': ppm_loop,
                            }
                            if rpm_map_choice:
                                opt_entry['rpm_map'] = rpm_map_choice.copy()
                                for ptype, rpm_val in rpm_map_choice.items():
                                    opt_entry[f'rpm_{ptype}'] = rpm_val
                            opts.append(opt_entry)
            if not any(o['nop'] == 0 for o in opts):
                opts.insert(0, {
                    'nop': 0,
                    'rpm': 0,
                    'dra_main': 0,
                    'dra_loop': 0,
                    'dra_ppm_main': 0,
                    'dra_ppm_loop': 0,
                })
        else:
            # Non-pump stations can inject DRA independently whenever a
            # facility exists (max_dr > 0).  If no injection is available the
            # upstream PPM simply carries forward.
            non_pump_opts: list[dict] = []
            max_dr_main = int(stn.get('max_dr', 0))
            rng = narrow_ranges.get(i - 1) if narrow_ranges else None
            if max_dr_main > 0:
                dr_min, dr_max = 0, max_dr_main
                if rng and 'dra_main' in rng:
                    dr_min = max(0, rng['dra_main'][0])
                    dr_max = min(max_dr_main, rng['dra_main'][1])
                dra_vals = _allowed_values(dr_min, dr_max, dra_step)
                for dra_main in dra_vals:
                    ppm_main = int(get_ppm_for_dr(kv, dra_main)) if dra_main > 0 else 0
                    non_pump_opts.append({
                        'nop': 0,
                        'rpm': 0,
                        'dra_main': dra_main,
                        'dra_loop': 0,
                        'dra_ppm_main': ppm_main,
                        'dra_ppm_loop': 0,
                    })
            if not non_pump_opts:
                non_pump_opts.append({
                    'nop': 0,
                    'rpm': 0,
                    'dra_main': 0,
                    'dra_loop': 0,
                    'dra_ppm_main': 0,
                    'dra_ppm_loop': 0,
                })
            opts.extend(non_pump_opts)

        station_opts.append({
            'name': name,
            'orig_name': stn['name'],
            'idx': i - 1,
            'kv': kv,
            'rho': rho,
            'L': L,
            'd_inner': d_inner,
            'rough': rough,
            'cum_dist': cum_dist,
            'elev_delta': elev_delta,
            'maop_head': maop_head,
            'maop_kgcm2': maop_kgcm2,
            'loopline': loop_dict,
            'options': opts,
            'is_pump': stn.get('is_pump', False),
            'pump_combo': stn.get('pump_combo'),
            'pump_types': stn.get('pump_types'),
            'active_combo': stn.get('active_combo'),
            'dra_shear_factor': float(stn.get('dra_shear_factor', 0.0) or 0.0),
            'dra_injector_position': stn.get('dra_injector_position'),
            'shear_injection': bool(stn.get('shear_injection', False)),
            'coef_A': float(stn.get('A', 0.0)),
            'coef_B': float(stn.get('B', 0.0)),
            'coef_C': float(stn.get('C', 0.0)),
            'coef_P': float(stn.get('P', 0.0)),
            'coef_Q': float(stn.get('Q', 0.0)),
            'coef_R': float(stn.get('R', 0.0)),
            'coef_S': float(stn.get('S', 0.0)),
            'coef_T': float(stn.get('T', 0.0)),
            'min_rpm': station_rpm_min,
            'dol': station_rpm_max,
            'power_type': stn.get('power_type', 'Grid'),
            'rate': float(stn.get('rate', 0.0)),
            'tariffs': stn.get('tariffs'),
            'sfc': float(stn.get('sfc', 0.0)),
            'sfc_mode': stn.get('sfc_mode', 'manual'),
            'engine_params': stn.get('engine_params', {}),
            'elev': float(stn.get('elev', 0.0)),
        })
        cum_dist += L
    # Cache the baseline downstream head requirement for each station using the
    # unmodified segment flows.  Most scenarios reuse this value directly; only
    # bypass cases require recomputing the downstream flow profile.
    baseline_req = [
        _downstream_requirement(
            stations,
            idx,
            terminal,
            segment_flows,
            KV_list,
            loop_usage_by_station=loop_usage_by_station,
        )
        for idx in range(N)
    ]
    # -----------------------------------------------------------------------
    # Dynamic programming over stations

    init_residual = int(stations[0].get('min_residual', 50))
    # Initial dynamic‑programming state.  Each state carries the cumulative
    # operating cost, the residual head after the current station, the full
    # sequence of record dictionaries (one per station), the last MAOP
    # limits, the current flow into the next segment and, importantly, a
    # ``carry_loop_dra`` field.  ``carry_loop_dra`` represents the drag
    # reduction percentage that remains effective on the loopline due to
    # upstream injection when a bypass scenario occurs.  At the origin
    # there is no upstream DRA on the loopline so this value starts at zero.
    #
    # Represent the carried mainline DRA as a queue of length/ppm slices so the
    # slug can be advanced accurately from station to station.

    def _linefill_to_queue(entries: list[dict], diameter: float) -> list[tuple[float, float]]:
        queue_entries: list[tuple[float, float]] = []
        if not entries:
            return queue_entries
        for batch in entries:
            try:
                length_val = float(batch.get('length_km', 0.0) or 0.0)
            except Exception:
                length_val = 0.0
            if length_val <= 0:
                try:
                    vol_val = float(batch.get('volume', 0.0) or 0.0)
                except Exception:
                    vol_val = 0.0
                if vol_val > 0 and diameter > 0:
                    length_val = _km_from_volume(vol_val, diameter)
            if length_val <= 0:
                continue
            try:
                ppm_val = float(batch.get('dra_ppm', 0.0) or 0.0)
            except Exception:
                ppm_val = 0.0
            if queue_entries and abs(queue_entries[-1][1] - ppm_val) <= 1e-9:
                prev_len, prev_ppm = queue_entries[-1]
                queue_entries[-1] = (prev_len + length_val, prev_ppm)
            else:
                queue_entries.append((length_val, ppm_val))
        return queue_entries

    initial_queue_entries = _linefill_to_queue(linefill_state, origin_diameter)
    initial_reach = max(float(dra_reach_km), 0.0)
    if not initial_queue_entries and initial_reach > 0:
        initial_ppm = 0.0
        if linefill_state:
            for batch in linefill_state:
                try:
                    ppm_candidate = float(batch.get('dra_ppm', 0.0) or 0.0)
                except Exception:
                    ppm_candidate = 0.0
                if ppm_candidate > 0:
                    initial_ppm = ppm_candidate
                    break
            else:
                try:
                    initial_ppm = float(linefill_state[0].get('dra_ppm', 0.0) or 0.0)
                except Exception:
                    initial_ppm = 0.0
        initial_queue_entries.append((initial_reach, initial_ppm))

    initial_queue = tuple(
        (
            float(length),
            float(ppm_val),
        )
        for length, ppm_val in initial_queue_entries
        if float(length) > 0
    )

    states: dict[int, dict] = {
        init_residual: {
            'cost': 0.0,
            'residual': init_residual,
            'records': [],
            'last_maop': 0.0,
            'last_maop_kg': 0.0,
            'flow': segment_flows[0],
            'carry_loop_dra': 0,
            'dra_queue_full': initial_queue,
            'dra_queue_at_inlet': initial_queue,
            'inj_ppm_main': 0,
        }
    }

    for stn_data in station_opts:
        new_states: dict[int, dict] = {}
        best_cost_station = float('inf')
        for state in states.values():
            flow_total = state.get('flow', segment_flows[0])
            dra_queue_prev_full = state.get('dra_queue_full')
            if dra_queue_prev_full is None:
                legacy_queue = state.get('dra_queue', ())
                dra_queue_prev_full = tuple(legacy_queue) if legacy_queue else ()
            dra_queue_prev_inlet = state.get('dra_queue_at_inlet')
            if dra_queue_prev_inlet is None:
                dra_queue_prev_inlet = dra_queue_prev_full
            prefix_entries: tuple[tuple[float, float], ...] = ()
            total_prev_full = _queue_total_length(dra_queue_prev_full)
            total_prev_inlet = _queue_total_length(dra_queue_prev_inlet)
            upstream_length = max(total_prev_full - total_prev_inlet, 0.0)
            if upstream_length > 1e-9:
                prefix_entries = _take_queue_front(dra_queue_prev_full, upstream_length)
            d_inner_state = float(stn_data.get('d_inner') or stn_data.get('d') or 0.0)
            precomputed_queue = _prepare_dra_queue_consumption(
                dra_queue_prev_inlet,
                stn_data['L'],
                flow_total,
                hours,
                d_inner_state,
            )
            for opt in stn_data['options']:
                # -----------------------------------------------------------------
                # Enforce bypass rules on loopline injection:
                # if the previous station operated in bypass mode (Case‑G)
                # then no loopline DRA injection is permitted at this
                # station (dra_loop must be zero).  The upstream carry‑over
                # drag reduction is used instead.  We detect bypass using
                # ``loop_usage_by_station`` when provided.
                if stn_data['idx'] > 0 and loop_usage_by_station is not None:
                    usage_prev = loop_usage_by_station[stn_data['idx'] - 1]
                    if usage_prev == 2 and opt.get('dra_loop') not in (0, None):
                        continue
                pump_running = stn_data.get('is_pump', False) and opt.get('nop', 0) > 0
                dra_segments, queue_after_list, inj_ppm_main = _update_mainline_dra(
                    dra_queue_prev_inlet,
                    stn_data,
                    opt,
                    stn_data['L'],
                    flow_total,
                    hours,
                    pump_running=pump_running,
                    pump_shear_rate=pump_shear_rate,
                    dra_shear_factor=stn_data.get('dra_shear_factor', 0.0),
                    shear_injection=bool(stn_data.get('shear_injection', False)),
                    is_origin=stn_data['idx'] == 0,
                    precomputed=precomputed_queue,
                )
                queue_after_body = tuple(
                    (
                        float(entry.get('length_km', 0.0) or 0.0),
                        float(entry.get('dra_ppm', 0.0) or 0.0),
                    )
                    for entry in queue_after_list
                    if float(entry.get('length_km', 0.0) or 0.0) > 0
                )
                combined_full_entries = tuple(prefix_entries) + tuple(queue_after_body)
                merged_after_full = _merge_queue(combined_full_entries)
                queue_after_full = tuple(
                    (
                        float(length),
                        float(ppm),
                    )
                    for length, ppm in merged_after_full
                    if float(length or 0.0) > 0
                )
                seg_length_total = float(stn_data.get('L', 0.0) or 0.0)
                queue_after_inlet = _trim_queue_front(queue_after_full, seg_length_total)
                total_positive = sum(length for length, ppm in dra_segments if ppm > 0)
                if total_positive > 0:
                    weighted_dr = 0.0
                    for length, ppm in dra_segments:
                        if ppm <= 0:
                            continue
                        weighted_dr += float(get_dr_for_ppm(stn_data['kv'], ppm)) * length
                    eff_dra_main = weighted_dr / total_positive
                    dra_len_main = min(total_positive, stn_data['L'])
                else:
                    eff_dra_main = 0.0
                    dra_len_main = 0.0
                scenarios = []
                # Base scenario: flow through mainline only
                hl_single, v_single, Re_single, f_single = _segment_hydraulics(
                    flow_total,
                    stn_data['L'],
                    stn_data['d_inner'],
                    stn_data['rough'],
                    stn_data['kv'],
                    eff_dra_main,
                    dra_len_main,
                )
                scenarios.append({
                    'head_loss': hl_single,
                    'v': v_single,
                    'Re': Re_single,
                    'f': f_single,
                    'flow_main': flow_total,
                    'v_loop': 0.0,
                    'Re_loop': 0.0,
                    'f_loop': 0.0,
                    'flow_loop': 0.0,
                    'maop_loop': 0.0,
                    'maop_loop_kg': 0.0,
                    'bypass_next': False,
                })
                if stn_data.get('loopline'):
                    loop = stn_data['loopline']
                    # Drag reduction on loopline applies across the entire loop
                    eff_dra_loop = opt['dra_loop']
                    dra_len_loop = loop['L'] if eff_dra_loop > 0 else 0.0
                    # Parallel scenario (main + loop split by equal head)
                    hl_par, main_stats, loop_stats = _parallel_segment_hydraulics(
                        flow_total,
                        stn_data['L'],
                        stn_data['d_inner'],
                        stn_data['rough'],
                        eff_dra_main,
                        dra_len_main,
                        loop['L'],
                        loop['d_inner'],
                        loop['rough'],
                        eff_dra_loop,
                        dra_len_loop,
                        stn_data['kv'],
                    )
                    v_m, Re_m, f_m, q_main = main_stats
                    v_l, Re_l, f_l, q_loop = loop_stats
                    # Parallel scenario without bypass
                    scenarios.append({
                        'head_loss': hl_par,
                        'v': v_m,
                        'Re': Re_m,
                        'f': f_m,
                        'flow_main': q_main,
                        'v_loop': v_l,
                        'Re_loop': Re_l,
                        'f_loop': f_l,
                        'flow_loop': q_loop,
                        'maop_loop': loop['maop_head'],
                        'maop_loop_kg': loop['maop_kgcm2'],
                        'bypass_next': False,
                    })
                    # Bypass scenario: same flow split but bypass next pump on loop
                    scenarios.append({
                        'head_loss': hl_par,
                        'v': v_m,
                        'Re': Re_m,
                        'f': f_m,
                        'flow_main': q_main,
                        'v_loop': v_l,
                        'Re_loop': Re_l,
                        'f_loop': f_l,
                        'flow_loop': q_loop,
                        'maop_loop': loop['maop_head'],
                        'maop_loop_kg': loop['maop_kgcm2'],
                        'bypass_next': True,
                    })
                    # Loop‑only scenario: entire flow goes through loopline only.
                    # Only include when diameters differ; otherwise the parallel
                    # scenario already captures equal pipes.
                    if abs(stn_data['d_inner'] - loop['d_inner']) > 1e-6:
                        hl_loop, v_loop_only, Re_loop_only, f_loop_only = _segment_hydraulics(
                            flow_total,
                            loop['L'],
                            loop['d_inner'],
                            loop['rough'],
                            stn_data['kv'],
                            eff_dra_loop,
                            dra_len_loop,
                        )
                        scenarios.append({
                            'head_loss': hl_loop,
                            'v': 0.0,
                            'Re': 0.0,
                            'f': 0.0,
                            'flow_main': 0.0,
                            'v_loop': v_loop_only,
                            'Re_loop': Re_loop_only,
                            'f_loop': f_loop_only,
                            'flow_loop': flow_total,
                            'maop_loop': loop['maop_head'],
                            'maop_loop_kg': loop['maop_kgcm2'],
                            'bypass_next': False,
                        })

                pump_cache_bucket = opt.setdefault('_pump_cache', {})
                pump_cache_key = (
                    round(flow_total, 6),
                    round(hours, 4),
                    start_time,
                    round(Ambient_temp, 2),
                    round(Fuel_density, 3),
                    round(Price_HSD, 4),
                )
                cache = pump_cache_bucket.get(pump_cache_key)
                if cache is None:
                    cache = _build_pump_option_cache(
                        stn_data,
                        opt,
                        flow_total=flow_total,
                        hours=hours,
                        start_time=start_time,
                        ambient_temp=Ambient_temp,
                        fuel_density=Fuel_density,
                        price_hsd=Price_HSD,
                    )
                    pump_cache_bucket[pump_cache_key] = cache
                pump_details = cache['pump_details']
                tdh = cache['tdh']
                eff = cache['efficiency']
                pump_bkw_total = cache['pump_bkw']
                prime_kw_total = cache['prime_kw']
                power_cost = cache['power_cost']

                pump_bkw = pump_bkw_total / opt['nop'] if opt.get('nop', 0) > 0 else 0.0
                motor_kw = prime_kw_total / opt['nop'] if opt.get('nop', 0) > 0 else 0.0

                # Filter candidate scenarios based on explicit loop-usage directives.
                filtered_scenarios = []
                if loop_usage_by_station is not None and stn_data.get('loopline'):
                    usage = loop_usage_by_station[stn_data['idx']]
                    if usage == 0:
                        # Only the base (no-loop) scenario is allowed.  Pick the first
                        # scenario with zero loop flow.
                        for cand in scenarios:
                            if cand['flow_loop'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 1:
                        # Use only the parallel scenario: loop flow > 0 and not bypass.
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and not cand.get('bypass_next', False):
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 2:
                        # Use only the bypass scenario: loop flow > 0 and bypass flag set.
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and cand.get('bypass_next', False):
                                filtered_scenarios.append(cand)
                                break
                    elif usage == 3:
                        # Loop-only directive: select scenario where all flow goes through loopline
                        for cand in scenarios:
                            if cand['flow_loop'] > 0.0 and cand['flow_main'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                    else:
                        # Unrecognised directive: fall back to no-loop scenario.
                        for cand in scenarios:
                            if cand['flow_loop'] == 0.0:
                                filtered_scenarios.append(cand)
                                break
                else:
                    filtered_scenarios = scenarios
                for sc in filtered_scenarios:
                    # Skip scenarios with unacceptable velocities
                    if sc['flow_main'] > 0 and not (V_MIN <= sc['v'] <= V_MAX):
                        continue
                    if sc['flow_loop'] > 0 and not (V_MIN <= sc['v_loop'] <= V_MAX):
                        continue

                    # -----------------------------------------------------------------
                    # Special handling for bypass patterns across an entire pipeline.
                    #
                    # When there are exactly two stations and the first station's
                    # loopline bypasses the pumps at the next station, the loopline
                    # flow travels all the way from the origin to the terminal before
                    # rejoining the mainline.  In such cases the flow split between
                    # the mainline and loopline should be determined by equalising
                    # the total head loss (friction + elevation) from the origin to
                    # the terminal rather than on a per-segment basis.  The default
                    # implementation splits flow only over the current segment, which
                    # underestimates the required head for the loopline when the
                    # downstream segment contains peaks or substantial length.  The
                    # block below recomputes the flow split and corresponding head
                    # loss for the first segment based on the combined length of
                    # successive segments.  It then overwrites the candidate
                    # scenario's flow and velocity fields accordingly.  Only apply
                    # this correction when bypassing the next pump on the very
                    # first station in a two-station pipeline.
                    if (
                        sc.get('bypass_next')
                        and stn_data['idx'] == 0
                        and N == 2
                        and stn_data.get('loopline')
                    ):
                        # Identify the downstream station
                        next_stn = stations[1]
                        # Compute total mainline and loopline path lengths from
                        # the current station to the terminal
                        length_main_total = stn_data['L'] + next_stn['L']
                        # Loopline on the next station may not exist; use zero
                        length_loop_total = (
                            stn_data['loopline']['L'] + next_stn.get('loopline', {}).get('L', 0.0)
                        )
                        # If the downstream station does not define a loopline,
                        # treat the loopline length as only the current segment
                        if length_loop_total <= 0.0:
                            length_loop_total = stn_data['loopline']['L']
                        # Effective drag reduction for the entire path based on the
                        # segment-weighted mainline drag reduction
                        eff_dra_main_tot = eff_dra_main
                        # Carry-over drag reduction on the loop from the previous state
                        carry_prev = int(state.get('carry_loop_dra', 0))
                        # In bypass mode the loopline may still inject additional DRA.
                        # Combine any upstream carry-over with the current option so
                        # the full effect is considered when splitting flow over the
                        # total path.
                        eff_dra_loop_tot = (
                            carry_prev + opt['dra_loop']
                            if sc.get('bypass_next')
                            else opt['dra_loop']
                        )
                        # Compute flow split to equalise head loss over the entire path
                        hl_tot, main_stats_tot, loop_stats_tot = _parallel_segment_hydraulics(
                            flow_total,
                            length_main_total,
                            stn_data['d_inner'],
                            stn_data['rough'],
                            eff_dra_main_tot,
                            length_main_total if eff_dra_main_tot > 0 else 0.0,
                            length_loop_total,
                            stn_data['loopline']['d_inner'],
                            stn_data['loopline']['rough'],
                            eff_dra_loop_tot,
                            length_loop_total if eff_dra_loop_tot > 0 else 0.0,
                            stn_data['kv'],
                        )
                        v_main_tot, Re_main_tot, f_main_tot, q_main_tot = main_stats_tot
                        v_loop_tot, Re_loop_tot, f_loop_tot, q_loop_tot = loop_stats_tot
                        # Recompute head loss for the first segment using the split
                        # flow on this segment.  Apply the same drag reduction
                        hl_main_seg, v_main_seg, Re_main_seg, f_main_seg = _segment_hydraulics(
                            q_main_tot,
                            stn_data['L'],
                            stn_data['d_inner'],
                            stn_data['rough'],
                            stn_data['kv'],
                            eff_dra_main_tot,
                            stn_data['L'] if eff_dra_main_tot > 0 else 0.0,
                        )
                        # Recompute loopline velocity and friction factor for the
                        # first segment.  The loopline may have different length
                        # than the mainline on this segment.
                        hl_loop_seg, v_loop_seg, Re_loop_seg, f_loop_seg = _segment_hydraulics(
                            q_loop_tot,
                            stn_data['loopline']['L'],
                            stn_data['loopline']['d_inner'],
                            stn_data['loopline']['rough'],
                            stn_data['kv'],
                            eff_dra_loop_tot,
                            stn_data['loopline']['L'] if eff_dra_loop_tot > 0 else 0.0,
                        )
                        # Overwrite the candidate scenario with corrected values
                        sc = sc.copy()
                        sc['flow_main'] = q_main_tot
                        sc['flow_loop'] = q_loop_tot
                        sc['v'] = v_main_seg
                        sc['Re'] = Re_main_seg
                        sc['f'] = f_main_seg
                        sc['v_loop'] = v_loop_seg
                        sc['Re_loop'] = Re_loop_seg
                        sc['f_loop'] = f_loop_seg
                        sc['head_loss'] = hl_main_seg
                        sc['bypass_next'] = True

                    # Determine the effective drag reduction on the loopline.  In bypass
                    # mode (Condition‑G) the DRA injection at this station is not
                    # performed on the loopline; instead the drag reduction from the
                    # upstream station persists.  Otherwise use the station's
                    # prescribed DRA for the loopline.  When there is no loop flow
                    # the value is irrelevant but carried forward.
                    carry_prev = int(state.get('carry_loop_dra', 0))
                    if sc['flow_loop'] > 0:
                        if sc.get('bypass_next'):
                            eff_dra_loop = carry_prev + opt['dra_loop']
                            inj_loop_current = opt['dra_loop']
                            inj_ppm_loop = opt['dra_ppm_loop']
                        else:
                            eff_dra_loop = opt['dra_loop']
                            inj_loop_current = opt['dra_loop']
                            inj_ppm_loop = opt['dra_ppm_loop']
                    else:
                        eff_dra_loop = 0
                        inj_loop_current = 0
                        inj_ppm_loop = 0

                    # Determine next carry-over drag reduction value for the loop.
                    if sc['flow_loop'] > 0:
                        if sc.get('bypass_next'):
                            new_carry = carry_prev + opt['dra_loop']
                        else:
                            new_carry = opt['dra_loop']
                    else:
                        new_carry = carry_prev

                    # Compute the resulting superimposed discharge head after the pump and
                    # check MAOP constraints.  Use the head delivered by the pumps on
                    # this segment.
                    sdh = state['residual'] + tdh
                    if sdh > stn_data['maop_head'] or (
                        sc['flow_loop'] > 0 and sdh > stn_data['loopline']['maop_head']
                    ):
                        continue

                    # Compute downstream residual head after segment loss and elevation
                    residual_next = int(round(sdh - sc['head_loss'] - stn_data['elev_delta']))

                    # Compute minimum downstream requirement.  Use the cached baseline
                    # unless bypassing the next station, in which case recompute with
                    # updated flows so downstream pumps see the correct mainline demand.
                    min_req = baseline_req[stn_data['idx']]
                    if sc.get('bypass_next') and stn_data['idx'] + 1 < N:
                        # Recompute downstream flows; the mainline flow changes only
                        # when bypassing the next station.  ``seg_flows_tmp`` holds the
                        # flow after each station for this scenario.
                        seg_flows_tmp = segment_flows.copy()
                        next_flow = flow_total
                        seg_flows_tmp[stn_data['idx'] + 1] = next_flow
                        for j in range(stn_data['idx'] + 1, N):
                            delivery_j = float(stations[j].get('delivery', 0.0))
                            supply_j = float(stations[j].get('supply', 0.0))
                            seg_flows_tmp[j + 1] = seg_flows_tmp[j] - delivery_j + supply_j

                        pump_overrides: dict[int, float] = {}
                        next_index = stn_data['idx'] + 1
                        next_orig = stations[next_index].get('orig_name') or stations[next_index].get('name')
                        j = next_index
                        while j < N:
                            if stations[j].get('orig_name') == next_orig or (
                                next_orig is None and stations[j].get('orig_name') is None
                            ):
                                pump_overrides[j] = sc['flow_main']
                                j += 1
                            else:
                                break
                        min_req = _downstream_requirement(
                            stations,
                            stn_data['idx'],
                            terminal,
                            seg_flows_tmp,
                            KV_list,
                            loop_usage_by_station=loop_usage_by_station,
                            pump_flow_overrides=pump_overrides,
                        )
                    if residual_next < min_req:
                        continue

                    # Compute DRA costs.  Only charge for injections performed at this
                    # station.  Mainline and loopline injections are handled
                    # separately and loopline cost is incurred only when
                    # an injection is made.
                    dra_cost = 0.0
                    if inj_ppm_main > 0:
                        dra_cost += inj_ppm_main * (sc['flow_main'] * 1000.0 * hours / 1e6) * RateDRA
                    # Loopline injection uses ``inj_ppm_loop`` computed
                    # earlier.  Charge cost only when an actual injection is
                    # performed at this station.
                    if sc['flow_loop'] > 0 and inj_loop_current > 0:
                        dra_cost += inj_ppm_loop * (sc['flow_loop'] * 1000.0 * hours / 1e6) * RateDRA

                    total_cost = power_cost + dra_cost

                    # Build the record for this station.  Update loop velocity and MAOP
                    # information based on the scenario.  Use the effective drag
                    # reduction for loopline in display.  Note: drag_reduction_loop
                    # reflects the value used in this segment (carry over for bypass).
                    record = {
                        f"pipeline_flow_{stn_data['name']}": sc['flow_main'],
                        f"pipeline_flow_in_{stn_data['name']}": flow_total,
                        f"loopline_flow_{stn_data['name']}": sc['flow_loop'],
                        f"head_loss_{stn_data['name']}": sc['head_loss'],
                        f"head_loss_kgcm2_{stn_data['name']}": head_to_kgcm2(sc['head_loss'], stn_data['rho']),
                        f"residual_head_{stn_data['name']}": state['residual'],
                        f"rh_kgcm2_{stn_data['name']}": head_to_kgcm2(state['residual'], stn_data['rho']),
                        f"sdh_{stn_data['name']}": sdh if stn_data['is_pump'] else state['residual'],
                        f"sdh_kgcm2_{stn_data['name']}": head_to_kgcm2(
                            sdh if stn_data['is_pump'] else state['residual'], stn_data['rho']
                        ),
                        f"rho_{stn_data['name']}": stn_data['rho'],
                        f"maop_{stn_data['name']}": stn_data['maop_head'],
                        f"maop_kgcm2_{stn_data['name']}": stn_data['maop_kgcm2'],
                        f"velocity_{stn_data['name']}": sc['v'],
                        f"reynolds_{stn_data['name']}": sc['Re'],
                        f"friction_{stn_data['name']}": sc['f'],
                        f"coef_A_{stn_data['name']}": stn_data['coef_A'],
                        f"coef_B_{stn_data['name']}": stn_data['coef_B'],
                        f"coef_C_{stn_data['name']}": stn_data['coef_C'],
                        f"coef_P_{stn_data['name']}": stn_data['coef_P'],
                        f"coef_Q_{stn_data['name']}": stn_data['coef_Q'],
                        f"coef_R_{stn_data['name']}": stn_data['coef_R'],
                        f"coef_S_{stn_data['name']}": stn_data['coef_S'],
                        f"coef_T_{stn_data['name']}": stn_data['coef_T'],
                        f"min_rpm_{stn_data['name']}": stn_data['min_rpm'],
                        f"dol_{stn_data['name']}": stn_data['dol'],
                    }
                    if sc['flow_loop'] > 0:
                        record.update({
                            f"velocity_loop_{stn_data['name']}": sc['v_loop'],
                            f"reynolds_loop_{stn_data['name']}": sc['Re_loop'],
                            f"friction_loop_{stn_data['name']}": sc['f_loop'],
                            f"maop_loop_{stn_data['name']}": sc['maop_loop'],
                            f"maop_loop_kgcm2_{stn_data['name']}": sc['maop_loop_kg'],
                        })
                    else:
                        record.update({
                            f"velocity_loop_{stn_data['name']}": 0.0,
                            f"reynolds_loop_{stn_data['name']}": 0.0,
                            f"friction_loop_{stn_data['name']}": 0.0,
                            f"maop_loop_{stn_data['name']}": 0.0,
                            f"maop_loop_kgcm2_{stn_data['name']}": 0.0,
                        })
                    if stn_data['is_pump']:
                        speed_display = opt.get('rpm', 0)
                        if (not speed_display or speed_display <= 0) and isinstance(opt.get('rpm_map'), Mapping):
                            rpm_values = [
                                int(val)
                                for val in opt['rpm_map'].values()
                                if isinstance(val, (int, float))
                            ]
                            if rpm_values:
                                speed_display = max(rpm_values)
                        speed_fields: dict[str, float] = {}
                        for pinfo in pump_details:
                            suffix = _normalise_speed_suffix(pinfo.get('ptype', ''))
                            rpm_val = pinfo.get('rpm')
                            if isinstance(rpm_val, (int, float)):
                                speed_fields[f"speed_{stn_data['name']}_{suffix}"] = float(rpm_val)
                        record.update({
                            f"pump_flow_{stn_data['name']}": flow_total,
                            f"num_pumps_{stn_data['name']}": opt['nop'],
                            f"speed_{stn_data['name']}": speed_display,
                            f"efficiency_{stn_data['name']}": eff,
                            f"pump_bkw_{stn_data['name']}": pump_bkw,
                            f"motor_kw_{stn_data['name']}": motor_kw,
                            f"power_cost_{stn_data['name']}": power_cost,
                            f"dra_cost_{stn_data['name']}": dra_cost,
                            f"pump_details_{stn_data['name']}": [p.copy() for p in pump_details],
                            f"dra_ppm_{stn_data['name']}": inj_ppm_main,
                            f"dra_ppm_loop_{stn_data['name']}": inj_ppm_loop,
                            f"drag_reduction_{stn_data['name']}": eff_dra_main,
                            f"drag_reduction_loop_{stn_data['name']}": eff_dra_loop,
                        })
                        if speed_fields:
                            record.update(speed_fields)
                    else:
                        record.update({
                            f"pump_flow_{stn_data['name']}": 0.0,
                            f"num_pumps_{stn_data['name']}": 0,
                            f"speed_{stn_data['name']}": 0,
                            f"efficiency_{stn_data['name']}": 0.0,
                            f"pump_bkw_{stn_data['name']}": 0.0,
                            f"motor_kw_{stn_data['name']}": 0.0,
                            f"power_cost_{stn_data['name']}": 0.0,
                            f"dra_cost_{stn_data['name']}": dra_cost,
                            f"pump_details_{stn_data['name']}": [],
                            f"dra_ppm_{stn_data['name']}": inj_ppm_main,
                            f"dra_ppm_loop_{stn_data['name']}": inj_ppm_loop,
                            f"drag_reduction_{stn_data['name']}": eff_dra_main,
                            f"drag_reduction_loop_{stn_data['name']}": eff_dra_loop,
                        })
                    # Accumulate cost and update dynamic state.  When comparing states
                    # with the same residual bucket, prefer the one with lower cost
                    # or, when costs tie, the one with higher residual.  Carry
                    # forward the loop DRA carry value and the updated downstream queue.
                    new_cost = state['cost'] + total_cost
                    if new_cost < best_cost_station:
                        best_cost_station = new_cost
                    bucket = residual_next
                    record[f"bypass_next_{stn_data['name']}"] = 1 if sc.get('bypass_next', False) else 0
                    new_record_list = state['records'] + [record]
                    existing = new_states.get(bucket)
                    flow_next = flow_total
                    if (
                        existing is None
                        or new_cost < existing['cost']
                        or (
                            abs(new_cost - existing['cost']) < 1e-9
                            and residual_next > existing['residual']
                        )
                    ):
                        new_states[bucket] = {
                            'cost': new_cost,
                            'residual': residual_next,
                            'records': new_record_list,
                            'last_maop': stn_data['maop_head'],
                            'last_maop_kg': stn_data['maop_kgcm2'],
                            'flow': flow_next,
                            'carry_loop_dra': new_carry,
                            'dra_queue_full': queue_after_full,
                            'dra_queue_at_inlet': queue_after_inlet,
                            'inj_ppm_main': inj_ppm_main,
                        }

        if not new_states:
            return {"error": True, "message": f"No feasible operating point for {stn_data['orig_name']}"}
        # After evaluating all options for this station retain only the
        # lowest-cost state for each residual (already enforced by ``bucket``)
        # and globally prune to the top ``STATE_TOP_K`` states or those within
        # ``STATE_COST_MARGIN`` of the best.  This keeps the search space
        # manageable while preserving near-optimal candidates.
        items = sorted(new_states.items(), key=lambda kv: kv[1]['cost'])
        threshold = best_cost_station + state_cost_margin
        pruned: dict[int, dict] = {}
        for idx, (residual_key, data) in enumerate(items):
            if idx < state_top_k or data['cost'] <= threshold:
                pruned[residual_key] = data
        states = pruned

    # Pick lowest-cost end state and, among equal-cost candidates,
    # prefer the one whose terminal residual head is closest to the
    # user-specified minimum.  This avoids unnecessarily high
    # pressures at the terminal which would otherwise waste energy.
    term_req = int(terminal.get('min_residual', 0))
    best_state = min(
        states.values(),
        key=lambda x: (x['cost'], x['residual'] - term_req),
    )
    result: dict = {}
    for rec in best_state['records']:
        result.update(rec)

    residual = int(best_state['residual'])
    total_cost = best_state['cost']
    last_maop_head = best_state['last_maop']
    last_maop_kg = best_state['last_maop_kg']

    queue_source = best_state.get('dra_queue_full')
    if queue_source is None:
        queue_source = best_state.get('dra_queue', ())
    queue_final = [
        (
            float(length),
            float(ppm),
        )
        for length, ppm in queue_source
        if float(length) > 0
    ]

    def _queue_to_linefill_entries(
        queue_entries: list[tuple[float, float]],
        diameter: float,
    ) -> list[dict]:
        converted: list[dict] = []
        for length_val, ppm_val in queue_entries:
            length_km = float(length_val)
            if length_km <= 0:
                continue
            ppm_float = float(ppm_val)
            ppm_int = int(round(ppm_float)) if ppm_float > 0 else 0
            entry = {
                'length_km': length_km,
                'dra_ppm': ppm_int,
            }
            if diameter > 0:
                entry['volume'] = _volume_from_km(length_km, diameter)
            else:
                entry['volume'] = 0.0
            converted.append(entry)
        return converted

    dra_segments_result = [
        {'length_km': length, 'dra_ppm': int(round(ppm)) if ppm > 0 else 0}
        for length, ppm in queue_final
    ]
    result['dra_segments'] = dra_segments_result

    linefill_from_queue = _queue_to_linefill_entries(queue_final, origin_diameter)
    result['linefill'] = linefill_from_queue

    term_name = terminal.get('name', 'terminal').strip().lower().replace(' ', '_')
    result.update({
        f"pipeline_flow_{term_name}": segment_flows[-1],
        f"pipeline_flow_in_{term_name}": segment_flows[-2],
        f"pump_flow_{term_name}": 0.0,
        f"speed_{term_name}": 0,
        f"num_pumps_{term_name}": 0,
        f"efficiency_{term_name}": 0.0,
        f"pump_bkw_{term_name}": 0.0,
        f"motor_kw_{term_name}": 0.0,
        f"power_cost_{term_name}": 0.0,
        f"dra_cost_{term_name}": 0.0,
        f"dra_ppm_{term_name}": 0,
        f"dra_ppm_loop_{term_name}": 0,
        f"drag_reduction_{term_name}": 0,
        f"drag_reduction_loop_{term_name}": 0,
        f"head_loss_{term_name}": 0.0,
        f"velocity_{term_name}": 0.0,
        f"reynolds_{term_name}": 0.0,
        f"friction_{term_name}": 0.0,
        f"sdh_{term_name}": 0.0,
        f"residual_head_{term_name}": residual,
    })
    rho_term = rho_list[-1]
    result[f"rh_kgcm2_{term_name}"] = head_to_kgcm2(residual, rho_term)
    result[f"sdh_kgcm2_{term_name}"] = 0.0
    result[f"rho_{term_name}"] = rho_term
    result[f"maop_{term_name}"] = last_maop_head
    result[f"maop_kgcm2_{term_name}"] = last_maop_kg
    result['total_cost'] = total_cost
    result['dra_front_km'] = sum(length for length, ppm in queue_final if ppm > 0)
    result['error'] = False
    return result


def solve_pipeline_with_types(
    stations: list[dict],
    terminal: dict,
    FLOW: float,
    KV_list: list[float],
    rho_list: list[float],
    RateDRA: float,
    Price_HSD: float,
    Fuel_density: float,
    Ambient_temp: float,
    linefill: list[dict] | dict | None = None,
    dra_reach_km: float = 0.0,
    mop_kgcm2: float | None = None,
    hours: float = 24.0,
    start_time: str = "00:00",
    pump_shear_rate: float = 0.0,
    rpm_step: int = RPM_STEP,
    dra_step: int = DRA_STEP,
    coarse_multiplier: float = COARSE_MULTIPLIER,
    state_top_k: int = STATE_TOP_K,
    state_cost_margin: float = STATE_COST_MARGIN,
) -> dict:
    """Enumerate pump type combinations at all stations and call ``solve_pipeline``."""

    try:
        pump_shear_rate = float(pump_shear_rate)
    except (TypeError, ValueError):
        pump_shear_rate = 0.0
    pump_shear_rate = max(0.0, min(pump_shear_rate, 1.0))

    best_result = None
    best_cost = float('inf')
    best_stations = None
    N = len(stations)

    def expand_all(pos: int, stn_acc: list[dict], kv_acc: list[float], rho_acc: list[float]):
        nonlocal best_result, best_cost, best_stations
        if pos >= N:
            # When all stations have been expanded into individual pump units,
            # perform loop-case enumeration explicitly.  We determine the
            # positions of units with looplines (typically the last unit of each
            # physical station) and then build loop usage directives for each
            # representative case.  This avoids relying on the internal
            # loop-enumeration of ``solve_pipeline``, which can behave
            # unpredictably when stations are split into multiple units.
            loop_positions = [idx for idx, u in enumerate(stn_acc) if u.get('loopline')]
            # Always run at least once even if no loops exist
            if not loop_positions:
                cases = [[]]
            else:
                # Determine per-loop diameter equality flags for the expanded stations.
                default_t_local = 0.007
                flags_expanded: list[bool] = []
                for pidx in loop_positions:
                    stn_e = stn_acc[pidx]
                    # Inner diameter of the mainline segment
                    if stn_e.get('D') is not None:
                        d_main_outer = stn_e['D']
                        t_main = stn_e.get('t', default_t_local)
                        d_inner_main = d_main_outer - 2 * t_main
                    else:
                        d_inner_main = stn_e.get('d', 0.0)
                    lp = stn_e.get('loopline') or {}
                    if lp:
                        if lp.get('D') is not None:
                            d_loop_outer = lp['D']
                            t_loop = lp.get('t', stn_e.get('t', default_t_local))
                            d_inner_loop = d_loop_outer - 2 * t_loop
                        else:
                            d_inner_loop = lp.get('d', d_inner_main)
                    else:
                        d_inner_loop = d_inner_main
                    flags_expanded.append(abs(d_inner_main - d_inner_loop) <= 1e-6)
                # Generate loop-case combinations based on flags
                cases = _generate_loop_cases_by_flags(flags_expanded)
            for case in cases:
                usage = [0] * len(stn_acc)
                for pidx, val in zip(loop_positions, case):
                    usage[pidx] = val
                # Call solve_pipeline with explicit loop usage and disable
                # internal enumeration.  This ensures the provided directives
                # are respected even for split stations.
                result = solve_pipeline(
                    stn_acc,
                    terminal,
                    FLOW,
                    kv_acc,
                    rho_acc,
                    RateDRA,
                    Price_HSD,
                    Fuel_density,
                    Ambient_temp,
                    linefill,
                    dra_reach_km,
                    mop_kgcm2,
                    hours,
                    start_time,
                    pump_shear_rate=pump_shear_rate,
                    loop_usage_by_station=usage,
                    enumerate_loops=False,
                    rpm_step=rpm_step,
                    dra_step=dra_step,
                    coarse_multiplier=coarse_multiplier,
                    state_top_k=state_top_k,
                    state_cost_margin=state_cost_margin,
                )
                if result.get("error"):
                    continue
                cost = result.get("total_cost", float('inf'))
                if cost < best_cost:
                    # Preserve usage directive for later labelling
                    result_with_usage = result.copy()
                    result_with_usage['loop_usage'] = usage.copy()
                    best_cost = cost
                    best_result = result_with_usage
                    best_stations = stn_acc
            return

        stn = stations[pos]
        kv = KV_list[pos]
        rho = rho_list[pos]

        if stn.get('pump_types'):
            # Determine available counts for each type
            availA = stn['pump_types'].get('A', {}).get('available', 0)
            availB = stn['pump_types'].get('B', {}).get('available', 0)
            combos = generate_type_combinations(availA, availB)
            seen_active: set[tuple[int, int]] = set()
            max_station_limit: int | None
            raw_max = stn.get('max_pumps')
            if isinstance(raw_max, (int, float)) and raw_max > 0:
                max_station_limit = int(raw_max)
            else:
                max_station_limit = None
            raw_min = stn.get('min_pumps')
            min_station_required = int(raw_min) if isinstance(raw_min, (int, float)) and raw_min > 0 else 0
            for numA, numB in combos:
                total_units = numA + numB
                if total_units <= 0:
                    continue
                if max_station_limit is not None and total_units > max_station_limit:
                    continue
                pdataA = stn['pump_types'].get('A', {})
                pdataB = stn['pump_types'].get('B', {})
                for actA in range(numA + 1):
                    for actB in range(numB + 1):
                        if actA + actB <= 0:
                            continue
                        if max_station_limit is not None and (actA + actB) > max_station_limit:
                            continue
                        if (actA + actB) < min_station_required:
                            continue
                        active_key = (actA, actB)
                        if active_key in seen_active:
                            continue
                        seen_active.add(active_key)
                        unit = copy.deepcopy(stn)
                        unit['pump_combo'] = {'A': availA, 'B': availB}
                        unit['active_combo'] = {'A': actA, 'B': actB}
                        if actA > 0 and actB == 0:
                            pdata = pdataA
                        elif actB > 0 and actA == 0:
                            pdata = pdataB
                        else:
                            pdata = None
                        if pdata is not None:
                            for coef in ['A', 'B', 'C', 'P', 'Q', 'R', 'S', 'T']:
                                unit[coef] = pdata.get(coef, unit.get(coef, 0.0))
                            unit['MinRPM'] = pdata.get('MinRPM', unit.get('MinRPM', 0.0))
                            unit['DOL'] = pdata.get('DOL', unit.get('DOL', 0.0))
                            unit['power_type'] = pdata.get('power_type', unit.get('power_type', 'Grid'))
                            unit['rate'] = pdata.get('rate', unit.get('rate', 0.0))
                            unit['tariffs'] = pdata.get('tariffs', unit.get('tariffs'))
                            unit['sfc'] = pdata.get('sfc', unit.get('sfc', 0.0))
                            unit['sfc_mode'] = pdata.get('sfc_mode', unit.get('sfc_mode', 'manual'))
                            unit['engine_params'] = pdata.get('engine_params', unit.get('engine_params', {}))
                        else:
                            min_map: dict[str, float] = {}
                            dol_map: dict[str, float] = {}
                            if actA > 0:
                                min_map['A'] = _extract_rpm(
                                    pdataA.get('MinRPM'),
                                    default=_station_min_rpm(stn, ptype='A'),
                                    prefer='min',
                                )
                                dol_map['A'] = _extract_rpm(
                                    pdataA.get('DOL'),
                                    default=_station_max_rpm(stn, ptype='A'),
                                    prefer='max',
                                )
                            if actB > 0:
                                min_map['B'] = _extract_rpm(
                                    pdataB.get('MinRPM'),
                                    default=_station_min_rpm(stn, ptype='B'),
                                    prefer='min',
                                )
                                dol_map['B'] = _extract_rpm(
                                    pdataB.get('DOL'),
                                    default=_station_max_rpm(stn, ptype='B'),
                                    prefer='max',
                                )
                            unit['MinRPM'] = min_map
                            unit['DOL'] = dol_map
                            unit['power_type'] = pdataA.get('power_type', unit.get('power_type', 'Grid'))
                            unit['rate'] = pdataA.get('rate', unit.get('rate', 0.0))
                            unit['tariffs'] = pdataA.get('tariffs', unit.get('tariffs'))
                            unit['sfc'] = pdataA.get('sfc', unit.get('sfc', 0.0))
                            unit['sfc_mode'] = pdataA.get('sfc_mode', unit.get('sfc_mode', 'manual'))
                            unit['engine_params'] = pdataA.get('engine_params', unit.get('engine_params', {}))
                        unit['max_pumps'] = actA + actB
                        unit['min_pumps'] = actA + actB
                        expand_all(pos + 1, stn_acc + [unit], kv_acc + [kv], rho_acc + [rho])
        else:
            expand_all(pos + 1, stn_acc + [copy.deepcopy(stn)], kv_acc + [kv], rho_acc + [rho])

    expand_all(0, [], [], [])

    if best_result is None:
        return {
            "error": True,
            "message": "No feasible pump combination found for stations.",
        }

    best_result['stations_used'] = best_stations
    return best_result


_exported_names = [name for name in globals() if not name.startswith('_')]
_exported_names.extend(['_km_from_volume', '_volume_from_km'])
__all__ = list(dict.fromkeys(_exported_names))
