# Feasible-solution search (plain language)

The optimizer always finishes a complete grid search before it labels a run infeasible. Here is the short version of how that search works and why higher DRA ppm values get checked even if you typed a lower baseline number:

1. **Build the chemical grid.** For each station the code converts the allowable %DR range into ppm and back so it can cover every 1‑ppm increment between the min and max bounds. That 1‑ppm grid is merged with the existing %DR step grid, so low baseline inputs cannot skip higher-ppm candidates.
2. **Enumerate pump setups.** Within that chemical grid, the solver loops over every allowed pump count, every pump-type combination (mixed types are allowed when the data says so), and every RPM from each type’s min to max in the configured step size.
3. **Evaluate each pair.** For each `(DRA ppm, pump setup)` pair, the hydraulic model checks SDH, pressure, suction, viscosity, and other constraints. It keeps the best feasible solution it finds anywhere in the grid.
4. **Decide the outcome.** Only after all DRA ppm values and pump setups have been tested does the solver decide what to return: if any feasible point exists it reports that solution; if none exist it reports the maximum achievable flow.

## Example

Suppose you type **5 ppm** as the baseline. The solver still tests 6 ppm, 7 ppm, and so on (in 1‑ppm steps) up to the station’s maximum. If SDH is satisfied at 7 ppm with some pump-speed mix, that feasible 7‑ppm result is returned instead of calling the run infeasible. The same exhaustive sweep happens if you had started at 7 ppm—the baseline never blocks the higher-ppm checks.
