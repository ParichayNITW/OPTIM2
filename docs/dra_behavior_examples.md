# Drag reducer dissipation and baseline floor enforcement examples

## 1. How the drag reducer dissipates

The hourly solver pumps the queue forward one sub-step at a time, reapplies the current DRA ppm profile to the shifted linefill, and then checks whether the upstream slug has vanished. If the head of linefill is untreated (zero ppm) or the tracked DRA front has collapsed to ~0 km, the optimiser backtracks to the previous hour and inserts a new origin slug.

| Hour | Sub-step | Pumped volume (m³) | Linefill head after shift | Tracked DRA front (km) | What happens next |
| --- | --- | --- | --- | --- | --- |
| 09:00 | 1/2 | 2,000 | 0 km untreated head; 25 km @ 4 ppm remaining | 25.0 | Solver accepts the step and carries the slug forward. |
| 09:30 | 2/2 | 2,000 | 0 km untreated head; 15 km @ 4 ppm remaining | 15.0 | Solver accepts the step; front shrinks as slug moves. |
| 10:00 | 1/2 | 2,000 | **10 km untreated head, slug fully consumed** | **0.0** | Hourly solve fails because friction rises with no DRA. Upstream check sees untreated head > 0 km **or** front ≤ tolerance, so it triggers backtrack. |
| 09:00 (backtracked) | — | — | Enforced queue now has 5 km @ 6 ppm | 5.0 | Retry starts with a fresh origin slug that prevents the zero-DRA failure. |

This mirrors the code path where each sub-step calls `shift_vol_linefill`, reapplies ppm via `apply_dra_ppm`, and, on failure, inspects the previous hour’s linefill to see whether the front disappeared before calling `_enforce_minimum_origin_dra`.【F:pipeline_optimization_app.py†L5135-L5235】

## 2. Baseline floor enforcement (step-by-step with numbers)

Assume:
- Pipeline length: 100 km
- Minimum fraction: 5% → baseline length floor = 5 km
- Baseline floor: 6 ppm, no segment-specific floors
- Treatable length (from volume/flow constraints): 4 km
- Plan volume: 1,000 m³ (enough to carve a slug)

### a. Build enforceable segments
Because there are no segment-specific floors, a single fallback segment is created at the origin with 5 km @ 6 ppm (5% of 100 km).【F:pipeline_optimization_app.py†L4604-L4664】

| Segment | Length requested (km) | PPM |
| --- | --- | --- |
| Origin fallback | 5.0 | 6 |

### b. Scale to treatable length
The treatable limit is 4 km, so the 5 km request is scaled down by 4 / 5 = 0.8. The enforced length becomes 4.0 km.【F:pipeline_optimization_app.py†L4748-L4789】

| Segment | Length after scaling (km) | Scaling note |
| --- | --- | --- |
| Origin fallback | 4.0 | 5.0 km × 0.8 = 4.0 km to fit the available volume |

### c. Compute slug volume
With 1,000 m³ available over 100 km, 10 m³ represents 1 km of pipe. For 4.0 km of treated distance, the target slug volume is 40 m³. No existing queue volume is counted, so the enforced slug totals **40 m³**.【F:pipeline_optimization_app.py†L4791-L4853】

### d. Allocate the volume to the segment
All 40 m³ are assigned to the single enforced segment, so the queue entry becomes 4.0 km @ 6 ppm, 40 m³.【F:pipeline_optimization_app.py†L4855-L4905】

| Segment | Final length (km) | PPM | Assigned volume (m³) |
| --- | --- | --- | --- |
| Origin fallback | 4.0 | 6 | 40 |

### e. Inject into the day plan
The plan has enough volume, so the first rows are split/relabeled to carve out 40 m³ @ 6 ppm. Remaining plan volume stays at its prior ppm. This creates the enforced origin slug and records the injection slices for transparency.【F:pipeline_optimization_app.py†L4907-L4996】

| Plan slice | Volume (m³) | Applied ppm | Note |
| --- | --- | --- | --- |
| Enforced slug | 40 | 6 | Carved from the head of the plan to meet the baseline floor |
| Remaining plan | 960 | existing ppm | Unchanged beyond the slug volume |

### f. Persist enforcement details
The tightened state records the enforced slug (ppm, length, volume), the plan injections, and the updated DRA reach so the retry knows a baseline slug exists at the origin.【F:pipeline_optimization_app.py†L4998-L5045】
