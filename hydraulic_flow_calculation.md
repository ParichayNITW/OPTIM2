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

## Feasibility logic (rechecked)
For any candidate flow, the solver treats it as **infeasible** if any of the
following fails during the hour-by-hour hydraulic run:

1) **Insufficient head vs. losses.** Pump head available at the candidate flow
   (after accounting for the number and type of pumps that can be started in the
   hour) must exceed total head demand: static lift, terminal residual, station
   residuals, and friction/minor losses along each segment.【F:pipeline_model.py†L2245-L2310】【F:pipeline_model.py†L3616-L3905】
2) **Pressure envelope exceeded.** The computed pressure at no point in the
   profile may cross MAOP or the user-provided MOP (converted to head). The
   solver clips the acceptable range to the tighter of these limits, so a single
   point above the envelope invalidates the flow.【F:pipeline_model.py†L2259-L2295】
3) **Pump operating limits.** The required head must be reachable using pumps
   that respect minimum speed, DOL limits, maximum pump count per station, and
   the configured pump selection rules. If no combination supplies the needed
   head at the candidate flow, the rate is rejected.【F:pipeline_model.py†L2482-L2559】

The max-flow search walks down through candidate rates and returns the first one
that satisfies **all** three at every time step.

## Worked check on the user JSON (why 2,708 m³/h fails)
Below is a strict hydraulic check using the supplied pipeline and pump data. The
candidate rate in question is **2,708 m³/h** (0.752 m³/s).

1) **Velocity and Reynolds number**
   - Pipe ID: 0.762 m → cross-sectional area A ≈ π·D²/4 ≈ 0.456 m².
   - Velocity v = Q/A ≈ 0.752 / 0.456 ≈ **1.65 m/s**.
   - With kinematic viscosity near 1×10⁻⁵ m²/s (4–16 cSt products), Reynolds
     Re ≈ v·D/ν ≈ 1.65·0.762 / 1e-5 ≈ **1.3×10⁵** → fully turbulent.

2) **Friction factor (Swamee–Jain)**
   - Relative roughness ε/D = 4×10⁻⁵ / 0.762 ≈ 5.25×10⁻⁵.
   - f ≈ 0.25 / [log₁₀(ε/(3.7D) + 5.74/Re⁰·⁹)]² ≈ **0.017** (turbulent steel pipe).

3) **Friction head**
   - Total length L = 328 km → L/D ≈ 328,000 / 0.762 ≈ 4.31×10⁵.
   - Velocity head v²/2g ≈ 1.65² / (2·9.81) ≈ **0.14 m**.
   - Darcy–Weisbach loss h_f = f·(L/D)·(v²/2g) ≈ 0.017·4.31×10⁵·0.14 ≈
     **1,000–1,020 m** of head.

4) **Available pump head at 2,708 m³/h**
   - Paradip station (2 pumps allowed). Best-case is two **MP4/MP5** type B
     pumps in series: head ≈ 351 m per pump at 2,500–2,700 m³/h → **≈ 700 m**
     combined.
   - Balasore station (1 pump max). Its **MP3** pump gives ≈ 400 m at
     2,500–2,700 m³/h.
   - Total available head in the hour is therefore **≈ 1,100 m** before any
     surge allowance.

5) **Other head requirements**
   - Elevation: about **2 m** lift to Haldia.
   - Residual pressures: minimum **125 m** at Paradip suction, **50 m** at
     Balasore discharge, and **75 m** at the terminal → roughly **250 m** of
     residual head that must remain after friction.

6) **Head balance**
   - Required head ≈ friction (1,000 m) + elevation (2 m) + residual (250 m)
     ≈ **1,252 m**.
   - Available head ≈ **1,100 m** (step 4).
   - **Gap:** available head is short by roughly **150 m**, so the solver marks
     2,708 m³/h as infeasible even before checking pressure envelope margins.

7) **Pressure envelope cross-check**
   - Because the total head is short, suction pressures would drop below the
     enforced residuals; boosting RPM to fix this would push discharge pressures
     above MAOP at the upstream station, also violating the envelope. Either
     way, the hydraulic run fails the feasibility test.

This numeric walk-through mirrors the app’s checks: at 2,708 m³/h the pipeline
demands more head than the pumps can safely supply while keeping residuals and
MAOP within limits, so the solver correctly labels the rate as **not feasible**.
