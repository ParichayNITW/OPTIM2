# How the App Computes Maximum Achievable Flow

This note describes the same hydraulic logic the app uses when it says it is
"Computing max achievable flow". It follows the pipeline model and solver logic
that run inside the optimization workflow, not a standalone textbook example.

## What the solver checks
- **Pressure envelope:** Each station has a maximum allowable operating pressure
  (MAOP). When a global MOP is provided, the app converts that pressure limit to
  head and clamps it against each station's MAOP so the solver never accepts a
  profile that would over-stress the pipe wall.【F:pipeline_model.py†L2259-L2295】
- **Available head vs. required head:** For a trial flow, the solver builds the
  hydraulic profile along the full length of the pipe, accounting for diameter,
  elevation, friction, temperature, and any drag-reducing agent (DRA) that is
  scheduled. The profile must stay within the pressure envelope while still
  meeting delivery pressure at the terminal.【F:pipeline_model.py†L2245-L2310】【F:pipeline_model.py†L3616-L3905】
- **Pump capability:** The solver evaluates pump curves and combinations to see
  whether enough head can be produced at the requested flow without exceeding
  pump limits or minimum allowable speeds.【F:pipeline_model.py†L2482-L2559】

A flow is **feasible** only if all these checks succeed across every hour of the
schedule.

## How the max-flow search runs in the app
1. **Run the requested plan.** The time-series solver is run at the user’s
   requested flow. If it is feasible, no further action is needed.
2. **Broaden the search if necessary.** When the run fails, the app first widens
   the solver search depth (tighter RPM/DRA steps, broader state space) to avoid
   dropping flow prematurely.【F:pipeline_optimization_app.py†L5850-L5886】
3. **Step down the flow.** If the wider search still reports the plan as
   infeasible, the app starts a controlled search for the highest feasible rate:
   - It aligns the starting candidate to the configured decrement (50 m³/h by
     default) so it does not skip just-below-the-request rates.
   - It reduces the flow in fixed steps and reruns the full hydraulic solver for
     each candidate until the first feasible solution is found.
   - For day schedules, the requested total volume is trimmed to match the
     candidate rate so throughput numbers stay consistent.【F:pipeline_optimization_app.py†L5892-L6074】
4. **Report the limit.** The first candidate that passes all hydraulic checks is
   returned as the "maximum achievable flow". The app also reports how much
   throughput was reduced versus the original request.

## Example using the app’s logic
Suppose a 24-hour schedule requests **3,000 m³/h** (72,000 m³ total) but the
solver declares it infeasible even after widening the search depth. With the
50 m³/h decrement:
- The search starts at 2,950 m³/h (because the request was an exact multiple of
  50, the first step is one full decrement).
- If 2,950 m³/h still violates the pressure envelope, the solver tries 2,900,
  then 2,850 m³/h, and so on, each time running the full hydraulic profile and
  pump checks.
- Imagine 2,850 m³/h is the first feasible rate; the app reports that as the
  maximum achievable flow with a total of 68,400 m³ delivered, a reduction of
  3,600 m³ from the request.

This is the same iterative hydraulic search the app performs when you click the
"Compute max achievable flow" action—no shortcuts, and no simplified formulas.
