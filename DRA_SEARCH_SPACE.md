# DRA search space in plain language

This note explains how the optimizer picks drag-reducing agent (DRA) doses alongside pump choices, using an easy-to-picture "ABC" pipeline segment example.

## What the optimizer varies
- **DRA dose choices:** The optimizer builds a grid of allowed DRA percentages for each station. Those percentages are always converted to ppm and capped at the 12 ppm ceiling. User-entered baselines and floors also sit inside that 0–12 ppm window.
- **Pump selections:** At pump stations, it tries different combinations of pump types (e.g., mainline vs. booster) and RPM settings inside each type’s min/max limits.

## Simple ABC pipeline example
Imagine three stations in order: **A → B → C**.

1. **Station A (pump station):**
   - Pump types: two mains and one booster. The optimizer enumerates permitted RPMs for each type (e.g., 1500–1800 RPM in steps), then considers every valid blend of those RPMs.
   - DRA grid: it builds percentage steps that convert to between the user baseline and the 12 ppm ceiling (respecting viscosity limits). Each pump/RPM combo is paired with every DRA grid point that stays within the cap.

2. **Station B (pump station):**
   - Same process as A: list allowable RPMs per pump type, form combinations, and pair each with each DRA grid value within 0–12 ppm.

3. **Station C (non-pump station):**
   - No pumps to vary, so only DRA grid values within 0–12 ppm are considered.

Across the route, the optimizer evaluates every station’s options in combination, but **DRA is never allowed to exceed the 12 ppm ceiling**, and pump RPMs stay inside their own limits.
