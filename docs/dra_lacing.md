# Drag-reducing agent (DRA) lacing overview

The optimisation UI requests a *target laced flow*, *target laced viscosity*, and *minimum suction pressure* because the solver needs a worst-case operating point when it decides how much drag reduction must persist along the entire trunk line. These numbers feed directly into `compute_minimum_lacing_requirement`, which walks the station list, evaluates superimposed discharge head (SDH), and returns the minimum concentration/length pair that keeps the terminal and intermediate suction limits satisfied at that design point.【F:pipeline_model.py†L1786-L1905】 The returned requirement drives the baseline queue that every schedule starts with, ensuring the dispatcher never schedules a path that would strand the terminal below its suction target even if pumps cycle off.

## How the code enforces lacing

Once the optimisation begins enumerating station options, `_update_mainline_dra` advances the downstream queue one station at a time.【F:pipeline_model.py†L1233-L1630】 Each call:

1. Converts the current flow/timestep into a pumped length so it knows how many kilometres of inventory leave the station during the hour.
2. Applies shear only when a station is running and the inherited slug crosses its pumps. Because injectors are modelled downstream of the pumps by default (`apply_injection_shear` only toggles when an injector is explicitly flagged as `"upstream"`), the fresh slug leaves the station at its requested ppm while the pumped portion inherits the sheared concentration for the next station to see.【F:pipeline_model.py†L1322-L1608】
3. Reconstructs the downstream queue by prepending the fresh slug (if any), appending the sheared portion of the inherited queue, and trimming the tail so the stored distribution still matches the physical linefill.【F:pipeline_model.py†L1616-L1630】 Floors from `compute_minimum_lacing_requirement` or terminal policies are overlaid slice-by-slice when needed so the residual drag reduction never drops below the baseline.【F:pipeline_model.py†L1537-L1596】

Together, `compute_minimum_lacing_requirement` and `_update_mainline_dra` implement “lacing”: the former computes how much DRA must be woven through the line, and the latter physically tracks that slug as it moves past each station and pump.

## Worked example: A → B → C pipeline

The helper script `scripts/dra_lacing_walkthrough.py` reproduces the three-station scenario discussed in chat. It assumes:

- Station A (origin) injects 8 ppm downstream of its pumps across a 40 km segment.
- Station B sits 40 km downstream, injects 4 ppm over a 60 km segment, and applies 20 % shear to any upstream slug that it pushes through its own pumps.
- Flow is 1,000 m³/h through a 0.33 m inner diameter pipe, so each station advances roughly 11.7 km of fluid per hour.
- The line starts full of untreated product, matching a “no lacing yet” baseline.

Running the script produces the hourly table below, which shows how the treated lengths accumulate through the first six hours of operation. The `A→B` and `B→C` columns expose the treated slices covering each segment during that hour, while the final column lists the first few entries of the downstream queue after Station B—this is the evolving linefill that Station C will eventually receive.

| Hour | A injection (ppm) | B injection (ppm) | A→B treated profile | B→C treated profile | Downstream queue after B |
| ---: | ---: | ---: | --- | --- | --- |
|    1 |    8.00 |    4.00 | 11.7 km @  8.0 ppm<br>28.3 km @  0.0 ppm | 11.7 km @  4.0 ppm<br>48.3 km @  0.0 ppm | 11.7 km @  8.0 ppm<br>28.3 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>48.3 km @  0.0 ppm |
|    2 |    8.00 |    4.00 | 23.4 km @  8.0 ppm<br>16.6 km @  0.0 ppm | 11.7 km @  4.0 ppm<br>11.7 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>24.9 km @  0.0 ppm | 23.4 km @  8.0 ppm<br>16.6 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  0.0 ppm<br>… (+2 more) |
|    3 |    8.00 |    4.00 | 35.1 km @  8.0 ppm<br> 4.9 km @  0.0 ppm | 11.7 km @  4.0 ppm<br>11.7 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  0.0 ppm<br>… (+2 more) | 35.1 km @  8.0 ppm<br> 4.9 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  0.0 ppm<br>… (+4 more) |
|    4 |    8.00 |    4.00 | 40.0 km @  8.0 ppm | 11.7 km @  4.0 ppm<br> 6.8 km @  5.0 ppm<br> 4.9 km @  0.0 ppm<br>11.7 km @  4.0 ppm<br>… (+3 more) | 40.0 km @  8.0 ppm<br>11.7 km @  4.0 ppm<br> 6.8 km @  5.0 ppm<br> 4.9 km @  0.0 ppm<br>… (+4 more) |
|    5 |    8.00 |    4.00 | 40.0 km @  8.0 ppm | 11.7 km @  4.0 ppm<br>11.7 km @  5.0 ppm<br>11.7 km @  4.0 ppm<br> 6.8 km @  5.0 ppm<br>… (+3 more) | 40.0 km @  8.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  5.0 ppm<br>11.7 km @  4.0 ppm<br>… (+4 more) |
|    6 |    8.00 |    4.00 | 40.0 km @  8.0 ppm | 11.7 km @  4.0 ppm<br>11.7 km @  5.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  5.0 ppm<br>… (+2 more) | 40.0 km @  8.0 ppm<br>11.7 km @  4.0 ppm<br>11.7 km @  5.0 ppm<br>11.7 km @  4.0 ppm<br>… (+3 more) |

### Reading the table

- **Hour 1**: Station A lays down an 11.7 km slug at the full 8 ppm while Station B, still pumping untreated inventory, injects 4 ppm over the first 11.7 km of its segment.
- **Hour 3**: The head of A’s slug arrives at B. The pumped portion is sheared down to ~5 ppm before B adds its 4 ppm slug downstream, so the B→C segment now contains alternating slices of B’s fresh injection and the degraded remnant from A.
- **Hours 4–6**: The A→B segment is fully treated at 8 ppm. B continues to layer 4 ppm slugs ahead of the 5 ppm sheared remnant from A, giving the terminal a repeating pattern of treated slices even before any regulatory floor is applied.

You can rerun the script with different injections, shear assumptions, or flows to explore how the lacing queue adapts. Because it uses the same private helpers that drive the optimiser, the walkthrough mirrors production behaviour and provides a precise, hour-by-hour view of the linefill dynamics.
