# Pipeline optimizer review (logic, speed, accuracy)

This note summarizes key risks in the current optimizer logic, explained in plain English with pointers to the code. Each point includes whether it can be addressed and how.

## 1) Zero head-loss when inputs are bad
* **What happens?** When roughness or viscosity are missing or invalid, the friction-term input to the Moody-style equation goes negative. The code then sets friction to zero, which makes head loss zero for that segment. That can make an impossible scenario look feasible.
* **Why it matters (plain English):** If the optimizer is fed a typo (e.g., a negative roughness) it may assume the pipe has no resistance at all and pick a plan that only works on paper.
* **Can we fix it?** Yes. Clamp roughness/viscosity to sane minimums and raise a warning instead of returning zero friction.
* **Where this lives:** `_segment_hydraulics` sets friction to 0 whenever the computed argument is non-positive. 【F:pipeline_model.py†L2053-L2086】

## 2) Drag-reduction effect is assumed flat over the treated length
* **What happens?** The solver applies the drag reduction as a simple percentage drop across a treated chunk. It does not model the chemical fading or shear degradation along the line, so a long treated segment is assumed to keep the same benefit end-to-end.
* **Why it matters (plain English):** In reality, the chemical wears off; treating 50 km at 20% may only feel like 20% at the start and less later. The current math may overpromise head savings and under-estimate pump needs on long runs.
* **Can we fix it?** Yes. Introduce a decay function (e.g., exponential with a user-provided half-life) so the effective drag reduction shrinks with distance or time.
* **Where this lives:** `_segment_hydraulics` applies a single percent reduction to the whole treated length when `dra_length` is provided. 【F:pipeline_model.py†L2062-L2085】

## 3) Search still scales combinatorially with pumps × RPMs × DRA steps
* **What happens?** Even with cached RPM/ppm grids, each station still iterates every pump count, every RPM combination, and every DRA grid value. The nested loops grow quickly as you widen the allowed steps or add pump types.
* **Why it matters (plain English):** Adding one extra RPM option for each pump type can double the number of trial setups. With multiple stations, the runtime can still spike despite the new caching.
* **Can we fix it?** Yes. Early prune obviously dominated options (e.g., higher RPM with higher cost but lower head), or use coarse-to-fine search that stops expanding when incremental head gain is tiny.
* **Where this lives:** The main enumeration loop builds a Cartesian product over per-type RPM lists and DRA grids for every pump-count choice. 【F:pipeline_model.py†L4740-L4815】

## 4) UI pruning limits may hide the true cheapest plan
* **What happens?** The Streamlit front-end caps the stored search states to the top 50 within a ₹5,000 margin by default. If many near-equal solutions exist, the true optimum might be dropped before later steps combine station choices.
* **Why it matters (plain English):** Throwing away near-ties too early can force the optimizer down a slightly more expensive path, especially on long pipelines with many stations.
* **Can we fix it?** Likely. Expose the limits to users, raise the cap for complex cases, or switch to a cost-relative cutoff (e.g., keep all states within 1% of best) instead of a hard count of 50.
* **Where this lives:** Default caps for `search_state_top_k` and `search_state_cost_margin` are set at the top of the Streamlit app. 【F:pipeline_optimization_app.py†L11-L54】
