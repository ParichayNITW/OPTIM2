# Worked DRA profile example (3 stations, pumps toggled)

This note explains how to reproduce the exact ppm-versus-kilometre output that
the optimiser emits for a three-station scenario when every DRA and pump switch
is enumerated. Instead of hand-copying a giant table into the docs, the example
is now driven by a runnable script so you can tweak the inputs and immediately
inspect the optimiser's raw output.

## Scenario that feeds the optimiser

| Item | Value |
| --- | --- |
| Total line length | 30 km (three 10 km segments) |
| Volume | 18,000 m³ (sets the inner diameter used to convert flow to km) |
| Flow rate | 10 km/h (each station segment is fully refreshed every hour) |
| Stations | Alpha, Bravo, Charlie — all have pumps and DRA injectors |
| Pump constraints | Alpha's pump is forced on (origin station); Bravo/Charlie pumps are optional |
| DRA rates when "On" | Alpha = 5 ppm, Bravo = 8 ppm, Charlie = 10 ppm |
| Baseline queue | 10 km @ 0 ppm, 10 km @ 15 ppm, 10 km @ 5 ppm, 10 km @ 0 ppm |
| Global pump shear | 1.0 (passed straight to `_update_mainline_dra`) |
| Simulation window | Initial snapshot (07:00) plus two hourly updates (08:00, 09:00) |

Because the queue advances 10 km per hour, each station sees exactly one slice
per hour in this toy scenario. The optimiser still tracks the queue slice by
slice, so the same setup can be used to explore more complex lacing.

## Running the worked example yourself

1. Execute the helper script:

   ```bash
   python scripts/three_station_profiles.py
   ```

2. The script instantiates the `Scenario` above (with `pump_shear_rate=1.0`) and
   feeds it to `generate_combination_profiles(hours=2)` so all 32 combinations
   (three DRA toggles × two downstream pumps) are enumerated.
3. Each block in the output lists the switch states (DRA + pump) followed by a
   three-row table that shows the optimiser's ppm-vs-km slices for Alpha, Bravo,
   and Charlie at 07:00, 08:00, and 09:00.

### Example excerpt

```
Alpha DRA: Off | Alpha Pump: On | Bravo DRA: Off | Bravo Pump: Off | Charlie DRA: Off | Charlie Pump: Off
Time | Alpha | Bravo | Charlie
-----|-------|-------|--------
07:00 | 10.0 km @ 0.0 ppm | 10.0 km @ 15.0 ppm | 10.0 km @ 5.0 ppm
08:00 | 10.0 km @ 0.0 ppm | 10.0 km @ 0.0 ppm | 10.0 km @ 15.0 ppm
09:00 | 10.0 km @ 0.0 ppm | 10.0 km @ 0.0 ppm | 10.0 km @ 0.0 ppm
```

All other combinations follow the same format, including the cases where Bravo
or Charlie's pump is running while their injectors are off. Because the dump is
produced straight from the optimiser, you can trust that it reflects the current
logic inside `dra_profile_combinations.generate_combination_profiles`.

## ABC pipeline variant (100 km + 50 km)

To match the latest ABC pipeline request (A→B = 100 km, B→C = 50 km, entire
line initially at 0 ppm, baseline/fallback requirement of 2 ppm on both
segments, 5 km/h flow, optional DRA shots of 3 ppm at A and 3 ppm at B, pumps
with global shear 1.0), run:

```bash
python scripts/abc_pipeline_profiles.py
```

This helper considers every combination of:

* A's DRA switch (On injects 3 ppm; its pump is forced on because it is the
  origin station), and
* B's DRA and pump switches (On injects 3 ppm, pump shear respected).

Both stations feed `_update_mainline_dra` with a `fallback_dra_ppm=2.0`, so even
when the injectors are off the optimiser enforces the 2 ppm baseline that the
user requested for each segment.

The output is formatted the same way as the three-station example: each block
lists the switch states followed by a two-column table showing the optimiser's
slice-by-slice ppm-vs-km view of the A→B and B→C segments at 07:00, 08:00, and
09:00. Because the script simply wraps `generate_combination_profiles`, the
numbers always reflect the optimiser's current logic.
