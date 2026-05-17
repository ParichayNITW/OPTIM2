# Manual baseline DRA walkthrough for provided JSON

This note reconstructs the baseline head balance using the JSON inputs (Paradip → Balasore → Haldia) and the user-entered lacing targets:

- Target laced flow: **2,500 m³/h**
- Target laced viscosity: **7 cSt**
- Lacing fluid density: **850 kg/m³**
- Origin suction head: **120 m** (user input; applies only at Paradip)
- MOP: **58 kg/cm²** (≈682.12 m head at 850 kg/m³)

Inner pipe diameter (0.762 m OD, 7.9248 mm wall): **0.7461504 m**【ed539f†L31-L35】

## Segment 1: Paradip → Balasore (158 km)
1. Friction loss at 2,500 m³/h, 7 cSt: **448.71 m**【b9c6f4†L1-L20】
2. Maximum pump head at Paradip (combinations at DOL and target flow):
   - 1×A: **182.0 m**
   - 2×A: **364.0 m**
   - 1×B: **360.0 m**
   - 2×B: **720.0 m** (best within the 2-pump limit)
   - 1A + 1B: **542.0 m**
   Pump head chosen = **720.0 m** from the 2×B combination. Pump head + suction = 720.0 + 120 = **840.0 m**, capped by MOP head **644.14 m**【279921†L1-L18】
3. Required head = friction 448.71 + downstream residual target 50 + elevation gain ≈ 2 → **≈500.71 m**.【279921†L1-L7】
4. Available head (capped by MOP) 644.14 m exceeds the 500.71 m requirement ⇒ **0% drag reduction** for this segment.

## Segment 2: Balasore → Haldia (170 km)
1. Friction loss at 2,500 m³/h, 7 cSt: **482.79 m**【b9c6f4†L1-L20】
2. Maximum pump head at Balasore (only 1×A available): pump head **400.0 m**; adding the downstream target suction 50.0 m yields **450.0 m** available before any MOP cap.【279921†L1-L18】
3. Required head = friction 482.79 + terminal residual 60 = **542.79 m**.【279921†L1-L18】
4. Shortfall = 542.79 − 450.0 = **92.79 m** ⇒ **19.22% drag reduction** (shortfall ÷ friction).【279921†L1-L18】

This reconstruction shows that with the provided inputs (flow 2,500 m³/h, 7 cSt, suction 120 m at origin, 60 m terminal residual), the expected DR is **0%** for Paradip–Balasore and **≈19.22%** for Balasore–Haldia, matching the solver output when only the downstream residual floors are used for suction.【279921†L1-L18】
