# Automatic baseline DRA computation: step-by-step outline with a worked example

This note summarizes how the automatic baseline DRA solver determines drag-reduction requirements segment by segment. It mirrors the UI flow used by the app and adds a concrete example so the intermediate numbers are easy to trace.

## Key inputs per station/segment
- **Pump curves and availability** (can include mixed pump types if allowed)
- **Max pumps at station** and **minimum suction head** at the originating station only; downstream stations take suction from the upstream residual head and any station-specific residual floors
- **Maximum operating pressure (MOP)** limit (converted to head)
- **Segment properties:** length, diameter, roughness, elevation/peaks, target laced flow/viscosity/density
- **Residual head target** at the downstream station or terminal (e.g., 50–60 m)
- **PPM cap** per segment for the baseline (15 ppm default)

## Algorithm (per segment, upstream to downstream)
1. **Pick the best pump combination at rated speed**
   - Enumerate all available pump combos (respecting max pumps and mixing rules).
   - For each combo, compute head at the target flow using the combo curve; pick the highest head.
2. **Apply suction and MOP limits**
   - Station discharge head (SDH) = min(best_combo_head + suction_head, MOP head limit).
3. **Compute required head for the segment**
   - Required = friction loss (from target flow/viscosity/density, length, diameter, roughness) + elevation/peaks + downstream residual head.
4. **Determine head shortfall and drag reduction**
   - If SDH ≥ required → no drag reduction needed; DRA ppm = 0 (or minimum enforced baseline ppm if configured).
   - If SDH < required → shortfall = required − SDH. Drag reduction % = shortfall / friction_loss.
   - Convert drag reduction % to DRA ppm using the viscosity-based correlation, capped at the per-segment PPM limit.
5. **Propagate downstream suction**
   - The residual head chosen for this segment becomes the suction head for the next station. If a downstream segment would exceed the PPM cap, increase the upstream residual head and recompute to minimize total PPM while respecting caps and residual floors.

## How downstream residual targets are chosen

For each segment, the downstream residual target comes from the **downstream station’s residual floor**, if any, or the **terminal residual** (whichever is higher as requirements propagate upstream). The UI’s "minimum suction" field only applies at the originating station; it is never reused for downstream targets. The solver now also records this value explicitly as `downstream_residual_target` in the debug trace so you can see the target separate from the originating station’s own minimum residual/suction head.

## Worked example (two segments)
Assumptions:
- Baseline cap = 15 ppm per segment.
- Terminal residual head = 60 m.
- Segment 1 (Paradip → Balasore): length 158 km; friction loss at target flow/viscosity = **650 m**; downstream residual head target = **60 m** (no explicit Balasore floor above the terminal requirement).
- Segment 2 (Balasore → Haldia): length 170 km; friction loss at target flow/viscosity = **520 m**; terminal residual head = **60 m**.
- Pump options at Paradip (max 2 pumps, mixing allowed):
  - 2×Type B at rated speed → head = **700 m** at target flow (best).
- Pump options at Balasore (max 1 pump):
  - 1×Type A at rated speed → head = **440 m** at target flow (best).
- MOP head limit = **600 m** at both stations.

### Segment 1 (Paradip → Balasore)
1) **Best pump head**: 2×Type B → 700 m.
2) **SDH after limits**: suction at origin = 180 m → raw head = 700 + 180 = 880 m; MOP cap = 600 m → **SDH = 600 m**.
3) **Required head**: friction 650 m + downstream residual 60 m (no peak elevation assumed) = **710 m**.
4) **Shortfall**: 710 − 600 = **110 m**.
   - Drag reduction % = 110 / 650 = **16.92%**.
   - Map 16.92% to ppm using correlation → suppose this yields **4 ppm** (within 15 ppm cap).
5) **Propagate residual**: Balasore suction = downstream target **60 m** (still respecting any Balasore floor if provided).

### Segment 2 (Balasore → Haldia)
1) **Best pump head**: 1×Type A → **440 m** at target flow.
2) **SDH after limits**: suction = 60 m → raw head = 500 m; MOP cap 600 m → **SDH = 500 m**.
3) **Required head**: friction 520 m + terminal residual 60 m = **580 m**.
4) **Shortfall**: 580 − 500 = **80 m**.
   - Drag reduction % = 80 / 520 = **15.38%** → correlation gives about **6–7 ppm**, under cap.
5) **Result**: Total baseline ≈ 4 ppm (seg1) + 7 ppm (seg2) = **~11 ppm**.

If Segment 2 had needed >15 ppm, the solver would raise Balasore’s residual head (and thus Segment 1’s target) until Segment 2 falls at or below 15 ppm, then recompute Segment 1’s drag reduction to minimize the summed ppm subject to caps and residual floors.
