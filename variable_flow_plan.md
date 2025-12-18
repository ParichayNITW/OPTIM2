# Variable-flow scheduling proposal

## Context
The current daily optimizer assumes a fixed flow rate for the 24h schedule: the requested daily volume is divided evenly across the day and each hour is solved independently at that constant flow. This causes two issues:

1. Early hours constrained by MOP may force a low flow across the entire day even though drag-reducer accumulation later would permit higher throughput.
2. The optimizer cannot trade off shorter high-flow periods against lower chemical usage to minimize cost while still meeting the total daily volume.

## Problem restatement
We need the optimizer to search over hour-by-hour flow rates so that the **sum of hourly volumes meets (or exceeds) the requested total volume at minimum cost**. Lower flow may be necessary in the first hours, but higher flow can be used later if hydraulic headroom increases due to DRA effects. The solver should therefore allow running the pipeline above the average rate for less than 24 hours when that yields the least cost plan.

## Proposed approach
1. **Discretize feasible hourly flow values**
   * Build a grid of candidate flows per hour between the minimum hydraulically feasible flow (considering MOP and pump availability) and an upper bound derived from station limits.
   * Use a tunable step (e.g., existing `flow_step` or a finer step when `coarse_step_multiplier=1`).

2. **Dynamic programming across hours**
   * For each hour `t`, keep states defined by `(cumulative_volume, linefill_state, dra_queue_state)` and store the minimum cost plan reaching that state.
   * Transition by choosing a flow from the hourly grid, invoking the existing hydraulic solver for that hour with the inherited linefill/DRA queues, and accumulate cost and volume.
   * Prune dominated states (higher cost and not more volume) to keep the search tractable.

3. **Stop condition**
   * After 24 transitions, select the least-cost state whose cumulative volume is >= the requested volume. If none exist, pick the maximum-volume state as the best-effort fallback (mirrors current behavior).

4. **Cost function alignment**
   * Reuse the per-hour cost components already computed (power, DRA usage, penalties). Ensure the DP accumulates these exactly as reported in the current per-hour results.

5. **Output reconstruction**
   * Trace back the chosen states to reconstruct per-hour flow, pump, and DRA settings for the final table. The UI can continue to render the existing 24-row summary, now with variable flow and total cost reflecting the optimized schedule.

## Implementation notes
* The DP layer can live alongside the current hourly solver: wrap the existing single-hour optimizer in a function that accepts prior linefill/DRA queues and returns the new state plus cost for a chosen flow.
* To avoid state explosion, bin cumulative volume (e.g., nearest 10–25 m³) and limit the number of stored states per hour (keep the top-N by cost for each volume bin).
* When `is_hourly` mode is explicitly requested, keep the existing single-hour behavior; the DP should run only in the 24h schedule path.
* The change is backward compatible: if the flow grid collapses to a single value (current constant-flow case), the DP reduces to the existing behavior.
