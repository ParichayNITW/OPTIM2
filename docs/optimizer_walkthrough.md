# Optimiser Execution Walkthrough (A–B–C example)

This note explains, step by step, how `pipeline_model.solve_pipeline` computes a
least-cost 24‑hour schedule while enforcing DRA lacing floors.  Each stage
references the exact implementation and then applies it to an illustrative
pipeline with three stations (origin **A**, intermediate **B**, terminal **C**).

* **Stations** A and B are pump stations created with the `_make_pump_station`
  helper used in the performance regression suite and allow up to 20% drag
  reduction (`max_dr=20`).
* **Terminal** C requires a minimum residual head of 50 m.
* **Initial linefill** comprises three products already dosed at 3 ppm; the
  day plan injects two new batches.  We assume a worst-case mainline flow of
  2 800 m³/h at 5 cSt when deriving lacing floors.
* **Suction heads** required at the two pump stations are 5 m (A) and 8 m (B).

The figures quoted below come from executing the model with these values, e.g.
via the Python snippet captured in `fced9f†L98-L113`.

## Step 1 – Canonical configuration assembly

The entry point normalises caller input (`solve_pipeline`,
`pipeline_model.py` L3440-L3526).  The solver accepts raw dictionaries and
produces a canonical `stations` list, viscosity/density vectors, per-station
flow targets, and user-tunable search parameters.  During this phase the code:

1. Coerces viscosities/densities to safe defaults so hydraulic calculations
   never divide by zero (`pipeline_model.py` L4330-L4344).
2. Builds `segment_flows`, the nominal throughput after each station, by
   subtracting deliveries and adding supplies (`pipeline_model.py` L4345-L4351).
3. Records any origin-wide DRA floor supplied by the UI so it can be folded into
   per-station limits once pipe geometry is known (`pipeline_model.py`
   L4352-L4414).

For our example, both stations inherit the same 2 800 m³/h design flow and the
origin diameter is 0.7 m, the default from `_make_pump_station`.

## Step 2 – Minimum lacing requirement per segment

`compute_minimum_lacing_requirement` is called from the Streamlit layer before
`solve_pipeline` to determine the minimum ppm that must be carried through each
segment.  The helper walks every station, comparing the superimposed discharge
head (SDH) at the worst-case flow against what the pumps can produce at DOL
speed; any deficit becomes a drag-reduction requirement (`pipeline_model.py`
L2063-L2138).  For each station:

1. Pipe hydraulics are evaluated with `_segment_hydraulics`, returning head loss
   for the full treated length (`pipeline_model.py` L2006-L2052).  With
   2 800 m³/h over a 5 km, 0.7 m pipe at 5 cSt we obtain a head loss of
   22.53 m, a velocity of 2.02 m/s, a Reynolds number of 2.83×10⁵, and a Darcy
   friction factor of 0.015 (`fced9f†L98-L101`).
2. `_pump_head` estimates the maximum head available from the installed pump at
   DOL.  Station A can raise roughly 200 m; when the 5 m suction requirement is
   subtracted the net head still exceeds the SDH, so the DRA gap is zero and no
   minimum ppm is imposed (`fced9f†L102-L109`).
3. The same logic is executed for Station B (8 m suction).  Again, no deficit is
   detected, so the required floor stays at 0 ppm (`fced9f†L110-L113`).

Had the SDH exceeded the available head, the solver would compute
`gap = max(SDH - (max_head - suction), 0)` and convert that gap to drag
reduction percentage capped by the global limit (30 %), finally deriving ppm via
`_dra_ppm_for_percent` (`pipeline_model.py` L2129-L2138, L2333-L2340).

The resulting `segments` list (our two entries with zero ppm floors) is cached
and attached to the station definitions so the runtime optimiser can enforce the
minimum concentrations per station.

## Step 3 – Carrying lacing floors into the hourly queue

When `solve_pipeline` starts building station options it first merges baseline
floors with any origin-wide requirement and translates percentage floors into
both integer DR limits and ppm thresholds (`pipeline_model.py` L4477-L4507).
The tolerance `floor_ppm_tol = max(floor_ppm_min×10⁻⁶, 10⁻⁹)` ensures floating-
point rounding cannot cause spurious floor violations (`pipeline_model.py`
L4508-L4510).  Because our example floor is 0 ppm, no additional restriction is
added.  If, for example, Station A had required 3 ppm, the floor would translate
into `floor_perc_min = ceil(_dra_percent_for_ppm(...))` and all candidate
combinations would be forced to keep `dra_main` at or above that integer
(`pipeline_model.py` L4491-L4507, L4612-L4638).

`_ensure_queue_floor` and `_overlay_queue_floor` guarantee that the in-flight
DRA queue (representing treated length downstream of each station) never drops
below the baseline requirement, even when no new product is injected.  The
adjusted queue becomes the inlet profile for the next station, so minimum ppm is
preserved across the entire mainline (`pipeline_model.py` L962-L1121, L4992-L5033).

## Step 4 – Enumerating station operating options

For each station the solver enumerates feasible pump settings:

1. RPM ranges are generated with `_rpm_candidates`, which either returns the
   full discrete grid or a coarse-to-fine selection depending on the refinement
   settings (`pipeline_model.py` L4532-L4589).  In our example the min and max
   RPM are both 1 000, so only one speed is considered.
2. Allowed drag-reduction percentages are derived from `max_dr` while respecting
   the floor computed in Step 3.  If a ppm floor exists, candidates that would
   under-dose after ppm conversion are filtered out (`pipeline_model.py`
   L4600-L4638).
3. Candidate tuples `(dra_main_use, ppm_main)` are de-duplicated using the ppm
   tolerance, ensuring the solver checks each effective dosing level at most
   once (`pipeline_model.py` L4671-L4704).
4. Loopline settings, pump shear, and injector placement modifiers are attached
   so downstream hydraulics can account for how much DRA survives the pumps
   (`pipeline_model.py` L4705-L4734).

Each candidate produces a structured record containing pump counts, RPMs, DRA
values, shear-adjusted effective ppm, and the resulting hydraulic state.
Options that would violate the floor are discarded early by the
`floor_requires_injection` guard (`pipeline_model.py` L5101-L5117).

## Step 5 – Updating the DRA queue and hydraulic profiles

`_update_mainline_dra` applies the selected dosing to the current queue of DRA
segments and returns the downstream profile (`pipeline_model.py` L1509-L1953).
This step:

1. Merges carried-over batches with newly injected doses while conserving
   treated length (`pipeline_model.py` L1551-L1624).
2. Applies floors to both the raw queue and the pumped portion so that any
   enforced minimum is satisfied exactly (`pipeline_model.py` L1778-L1839,
   L1823-L1825).
3. Exposes `floor_requires_injection`, a boolean telling the caller that the
   candidate failed to meet the minimum ppm and should be skipped (`pipeline_model.py`
   L1785-L1796, L1951-L1953).

## Step 6 – Evaluating hydraulic feasibility and cost

With the updated queue the solver computes hydraulic losses under different
loop usage scenarios using `_segment_hydraulics_composite` and, when loops are
present, `_parallel_segment_hydraulics` (`pipeline_model.py` L5158-L5200).
Each scenario yields head loss, velocity, Reynolds number, friction factor, and
loop bypass flags.  For each feasible scenario the optimiser assembles a state
record containing:

* Operating cost (power + fuel + DRA) for the station, derived from the tariff
  configuration (`pipeline_model.py` L5223-L5294, L5315-L5348).
* Updated hydraulic envelope, including residual head checks against peaks and
  the terminal requirement (`pipeline_model.py` L5303-L5338, L5463-L5520).
* The downstream DRA queue ready for the next station.

Candidates exceeding the per-hour cost cap or violating residual head/DRA caps
are pruned immediately (`pipeline_model.py` L5249-L5286, L5452-L5476).

## Step 7 – Dynamic programming across stations

The solver maintains a frontier of the lowest-cost states entering each station
using a dynamic-programming table limited by `state_top_k` and
`state_cost_margin` (`pipeline_model.py` L5360-L5447).  For every option that
passes hydraulic checks, the cumulative cost is compared to existing entries; if
it is cheaper (or within the allowed margin) it is retained, otherwise it is
pruned.  This yields, for each station index, a sorted list of viable upstream
configurations along with their DRA queues and residual head buffers.

## Step 8 – Selecting the least-cost end state

Once the terminal segment has been processed, the frontier holds every feasible
combination reaching Station C.  The solver selects the cheapest state, applies
any outstanding origin-wide cost cap, and returns a dictionary containing:

* Per-station pump RPM, number of pumps, mainline and loop DRA ppm, and the
  effective treated length (`pipeline_model.py` L5487-L5587, L5678-L5690).
* The merged downstream queue and DRA floor audit trail, so the UI can render
  the hourly ppm values and show that each station met or exceeded its minimum
  (`pipeline_model.py` L5584-L5635, L5993-L6023).
* The total cost in INR for the hour plus derived metrics such as max drag
  reduction used (`pipeline_model.py` L5629-L5635, L5670-L5689).

## Step 9 – 24‑hour schedule assembly

`solve_pipeline_with_types` and the Streamlit `Start task` button call the hour
solver repeatedly, optionally in parallel, and accumulate the 24 hourly records
before composing the daily table (`optimized_scheduler.py` L689-L750).  The app
also measures elapsed time for the daily run so the UI can display how many
seconds the solver took (`pipeline_optimization_app.py` L243-L281, L3909-L3964).

Because DRA floors are baked into every stage—from minimum requirement
computation, through queue manipulation, to candidate pruning—the final
24‑hour table always reports hourly dosing that is at least the enforced ppm
floor while still reflecting the globally least-cost configuration permitted by
the hydraulics and drag-reduction caps.
