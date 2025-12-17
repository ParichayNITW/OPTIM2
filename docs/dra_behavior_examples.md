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

## 3. Walkthrough with your JSON (Paradip → Balasore)

The supplied JSON defines a 328 km line (158 km + 170 km). The inner diameter after subtracting wall thickness is 0.746 m, so one kilometre holds ~437 m³ (π·D²/4 · 1000). That converts every pumped cubic metre into a travel distance for the DRA front.

### 3.1 How the drag reducer dissipates in this case

Inputs used:

- Hourly flow: 1,000 m³/h → 1,000 m³ ÷ 437 m³/km ≈ **2.29 km** pushed each sub-step.
- Current linefill totals 143,538 m³ (≈ 328 km) with initial DRA 5.15–5.47 ppm throughout, but the day plan injects 0 ppm at the head.

We watch two half-hour sub-steps as the 0-ppm batch enters, eats into the DRA head, and eventually triggers backtracking when the upstream slug vanishes.【F:pipeline_optimization_app.py†L5135-L5234】

| Clock time | Pumped this sub-step (m³) | Head after shift | DRA front remaining (km) | Why the solver reacts |
| --- | --- | --- | --- | --- |
| 09:00 (sub-step 1) | 1,000 | First 2.29 km now 0 ppm (from day plan); rest still 5+ ppm | 325.7 | Solver is feasible: DRA still covers almost the whole line; no backtrack. |
| 09:30 (sub-step 2) | 1,000 | Head 4.58 km is 0 ppm; downstream still 5+ ppm | 323.4 | Still feasible; front shrinks as zero-DRA volume advances. |
| 10:00 (sub-step 1) | 1,000 | Head 6.87 km is 0 ppm; prior DRA slug effectively consumed at the origin | ~0.0 | Hydraulic solve fails on the untreated head. Upstream check sees untreated head > 0 km **or** front ≤ tolerance, so it backtracks and enforces a fresh origin slug. |
| 09:00 retry (after backtrack) | — | Queue now starts with enforced slug (Section 3.2) | ≥ enforced length | Retry proceeds with the enforced slug so the next solve has positive DRA at the head. |

### 3.2 Baseline floor enforcement with your numbers

Inputs used for enforcement call:

- Total length: 328 km.
- Baseline floor from JSON: 4 ppm across both segments (158 km + 170 km) → **328 km @ 4 ppm** requested.【F:pipeline_optimization_app.py†L4595-L4678】
- Minimum fraction: 5% of 328 km = **16.4 km** (already smaller than the 328 km baseline, so baseline governs).
- Hourly pumped volume: 1,000 m³ → treatable length = 1,000 m³ × (328 km ÷ 143,538 m³) ≈ **2.29 km** (limited by current linefill volume).【F:pipeline_optimization_app.py†L4580-L4592】
- Day plan volume: 67,200 m³ @ 0 ppm available to carve the slug.

Step-by-step enforcement with scaling and volumes:

| Step | Calculation | Result |
| --- | --- | --- |
| Build required segments | Two segment floors from JSON: 158 km @ 4 ppm, 170 km @ 4 ppm. | Requested 328 km @ 4 ppm.【F:pipeline_optimization_app.py†L4595-L4678】 |
| Scale to treatable length | Scale factor = 2.29 ÷ 328 ≈ 0.00699. | Segment A → 1.11 km; Segment B → 1.19 km; total **2.29 km**.【F:pipeline_optimization_app.py†L4748-L4789】 |
| Convert to volume | 437 m³/km × lengths. | Segment A ≈ 484 m³; Segment B ≈ 520 m³; total slug ≈ **1,004 m³** (capped to the 1,000 m³ available if needed).【F:pipeline_optimization_app.py†L4791-L4853】 |
| Assign to queue | Each scaled segment keeps 4 ppm. | Queue gets two entries: 1.11 km @ 4 ppm (484 m³) and 1.19 km @ 4 ppm (520 m³).【F:pipeline_optimization_app.py†L4855-L4905】 |
| Inject into plan | Carve ~1,004 m³ from head of the 0-ppm plan row. | Plan row is split: first ~1,004 m³ relabeled to 4 ppm; remaining ~66,196 m³ stays at 0 ppm. Records injection metadata for transparency.【F:pipeline_optimization_app.py†L4907-L4996】 |
| Persist enforcement | Update reach and note enforced slug. | `dra_reach_km` raised to ~2.29 km; detail records ppm, length, volume, and plan injections for the retry.【F:pipeline_optimization_app.py†L4998-L5054】 |

This enforced slug is what the backtracked 09:00 retry uses in Section 3.1 so that the head no longer starts at 0 ppm.
