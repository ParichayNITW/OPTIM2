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
For any candidate flow, the solver treats it as **infeasible** only after it
exhausts all pump and DRA options and still cannot satisfy the hydraulic rules
below during the hour-by-hour run:

1) **Head balance including suction head.** Pump head available at the candidate
   flow (accounting for the number/type of pumps, their RPM band, and DRA
   settings allowed in that hour) must exceed total head demand: static lift,
   terminal residual, station discharge residuals, and friction/minor losses
   along each segment **minus any usable suction head at upstream stations**.
   Suction head is treated as head already in the system and therefore reduces
   the net head that pumps must supply.【F:pipeline_model.py†L2245-L2310】【F:pipeline_model.py†L3616-L3905】
2) **Pressure envelope respected.** The computed pressure at no point in the
   profile may cross MAOP or the user-provided MOP (converted to head). The
   solver clips the acceptable range to the tighter of these limits, so a single
   point above the envelope invalidates the flow.【F:pipeline_model.py†L2259-L2295】
3) **Pump operating limits searched.** The solver iterates over pump
   combinations, from minimum to maximum RPM (within MinRPM/DOL), across all
   available pump types and counts, and across permitted DRA doses for the hour.
   A candidate is rejected only if **no** combination inside those bounds can
   supply the needed head without violating speed, count, or envelope limits.
   【F:pipeline_model.py†L2482-L2559】【F:pipeline_optimization_app.py†L5850-L6074】

The max-flow search walks down through candidate rates and returns the first one
that satisfies **all** three at every time step. Among feasible candidates for a
requested flow, the objective is to pick the **least daily cost** schedule (pump
selection, RPM, and DRA) that still meets all constraints.

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
     **1,000–1,020 m** of head. (With the minimum enforced 4 ppm DRA, friction
     falls only marginally and does not change the conclusion.)

4) **Available pump head at 2,708 m³/h**
   - Paradip station (2 pumps allowed). Best-case is two **MP4/MP5** type B
     pumps in series: head ≈ 351 m per pump at 2,500–2,700 m³/h → **≈ 700 m**
     combined.
   - Balasore station (1 pump max). Its **MP3** pump gives ≈ 400 m at
     2,500–2,700 m³/h.
   - Total available head in the hour is therefore **≈ 1,100 m** before any
     surge allowance.

5) **Other head requirements and available suction**
   - Elevation: about **2 m** lift to Haldia.
   - Residual pressures: minimum **50 m** at Balasore discharge and **75 m** at
     the terminal.
   - Suction head at Paradip: **125 m** already present upstream, so it offsets
     part of the pump duty rather than adding to it.

6) **Head balance after suction credit**
   - Required head ≈ friction (1,000 m) + elevation (2 m) + discharge residuals
     (50 + 75 m) − suction head (125 m) ≈ **1,002 m**.
   - Available head ≈ **1,100 m** (step 4) from the best pump pairing in the
     hour at maximum allowable RPM.
   - **Margin:** only about **+100 m** of head remains. The solver then checks
     whether any speed/DRA combination within limits can keep every point inside
     the pressure envelope. At 2,708 m³/h, even with minimum friction and maximum
     allowed RPM, the upstream discharge pressure clips the MAOP envelope before
     that margin can be used, so the hydraulic run still fails.

7) **Pressure envelope cross-check after exploring all settings**
   - The solver tries all permitted combinations of pump type/count, RPM, and
     DRA (including lacing baselines) to trade head against friction. Every
     combination that meets suction and discharge residuals at this flow still
     breaches the MAOP/MOP envelope at the upstream station. That is why the
     flow is labeled **not feasible** despite iterating through all pump-speed
     and DRA options.

This walk-through matches the app’s logic: suction head is credited, all pump
speed and DRA combinations are explored, and a flow is marked infeasible only
after none of those combinations can satisfy head balance **and** stay within
the pressure envelope. Once a flow is feasible, the optimizer seeks the least
daily-cost combination of pumps, RPM, and DRA for that rate.
