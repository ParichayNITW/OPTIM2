# How the solver decides feasibility (step-by-step)

This note walks through the exact checks the optimizer performs before it declares a requested throughput infeasible, using the provided two-station JSON (Paradip → Balasore → Haldia at 2 ppm baseline) as the running example.

## 1) Build the option grid per station

1. **Pump combinations and RPM ranges** – For each station, the solver enumerates allowed pump-type combinations, the number of running pumps, and the full RPM band between the station’s `MinRPM` and `DOL` limits. Each candidate embeds the DRA choice for that station. 【F:pipeline_model.py†L5200-L5268】【F:pipeline_model.py†L5270-L5314】
2. **DRA ppm sweep starts at the baseline floor** – The DRA grid spans from the requested floor ppm (converted from %DR) up to the maximum drag-reduction cap, optionally downsampled only for refinement passes. PPM values that sit below the enforced floor are dropped. For the JSON, that means the non-pump options at each origin segment start at the 2 ppm baseline and climb toward the cap. 【F:pipeline_model.py†L5200-L5243】

## 2) Seed the downstream DRA queue from the baseline table

The manual baseline table creates an initial downstream DRA queue: 158 km at 2 ppm for Paradip→Balasore plus 170 km at 2 ppm for Balasore→Haldia, giving a 328 km treated column before optimization starts. This queue is carried into the first station’s dynamic-programming (DP) state so the solver knows how much treated fluid already exists. 【F:pipeline_model.py†L1780-L1854】【F:pipeline_model.py†L5545-L5559】

## 3) For each DP state, try every station option

During DP, each upstream state is expanded through every local option:

- The solver precomputes how far the current timestep pumps fluid (pumped length) so it can slice the DRA queue accurately for this segment. 【F:pipeline_model.py†L5554-L5566】
- It advances the queue through `_update_mainline_dra`, which applies shear, adds any new injection, and overlays required floors. If the floor absolutely cannot be met without more injection, the option is discarded **after** checking the full queue (not just the short pumped slice). 【F:pipeline_model.py†L1789-L1854】【F:pipeline_model.py†L5584-L5605】
- Hydraulics are then computed for the segment (mainline and possible loop), producing head loss, velocities, and friction factors for the specific pump/DRA combination. 【F:pipeline_model.py†L5626-L5643】【F:pipeline_model.py†L5644-L5668】
- The downstream head requirement is compared to the available pump head; only options that maintain the required residual head advance to the next station’s state list with their accumulated cost. 【F:pipeline_model.py†L3385-L3445】【F:pipeline_model.py†L5333-L5345】

## 4) Keep near-best states, prune dominated ones

At each station, the solver keeps only a bounded set of DP states: dominated head/cost pairs are removed, and near-ties are retained based on the configured absolute/percentage margins. This keeps the search tractable while preserving close contenders so feasible solutions aren’t lost to aggressive pruning. 【F:pipeline_model.py†L3445-L3472】【F:pipeline_model.py†L5470-L5519】

## 5) When is a flow marked infeasible?

The requested flow is deemed infeasible only if **no** DP state survives through the final station with enough head and a valid DRA queue. Before reporting infeasibility, the app now performs progressively wider retries: it relaxes pruning limits, re-expands the full DRA ppm range, and finally runs a brute-force pass to explore every combination. Only after all retries fail does the UI fall back to “max achievable flow.” 【F:pipeline_model.py†L3473-L3519】【F:pipeline_optimization_app.py†L5958-L6025】

## Why a low baseline can look infeasible today

With the 2 ppm baseline, the initial queue is long but thin. When a station pumps only a short slice (say ~50 km) in one timestep, that slice alone may not meet a per-segment floor even though the remaining 278 km downstream is already treated. The updated full-queue check prevents automatic rejection in this case, but any remaining infeasibility would come from hydraulic limits (insufficient pump head at the chosen RPMs) once every pump/DRA combination is actually evaluated. Increasing the baseline to 7 ppm boosts the effective drag reduction across the entire queue, reducing friction losses and making the head balance easier, so more combinations pass the hydraulic check and the solver finds a feasible plan.
