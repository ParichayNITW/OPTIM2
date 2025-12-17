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
2. Maximum pump head at Paradip (combinations, DOL @ target flow):
   - A: 357.41 m (single) → 714.82 m (2A)
   - B: 357.41 m (from JSON B-type) → 714.82 m (2B)
   - 1A+1B: 752.32 m
   Max permitted head = **714.82 m** (2B)【67eac2†L1-L12】
3. Station discharge head (SDH) = min(MOP head 682.12 m, pump head + suction 714.82 + 120) = **682.12 m**【f60129†L1-L6】
4. Required head = friction 448.71 + downstream residual target 50 + elevation gain ≈ 2 → **≈500.71 m**.
5. Available head exceeds requirement ⇒ **0% drag reduction** for this segment; residual head margin remains for the next segment.

## Segment 2: Balasore → Haldia (170 km)
1. Friction loss at 2,500 m³/h, 7 cSt: **482.79 m**【b9c6f4†L1-L20】
2. Maximum pump head at Balasore (1×A available): **394.91 m** at target flow【67eac2†L6-L12】
3. If only the downstream target (60 m) is used as suction, SDH = 394.91 + 60 = **454.91 m** (well below MOP cap). Required head = 482.79 + 60 = **542.79 m**, so the shortfall is **87.88 m**, implying **≈18% drag reduction**.
4. To achieve only **≈1.4% drag reduction** (as in the screenshot), the inlet/suction head would need to be raised to ~141 m so that available head becomes **≈536 m**, leaving a ~6.8 m shortfall (≈1.4% of 482.79 m). This elevated suction is not part of the user inputs and suggests the solver is inflating the Balasore inlet to stay within a PPM cap rather than respecting the origin-only suction rule.

## Where the software likely diverges from the manual calculation
- The solver appears to lift the downstream inlet head (to ~141 m) instead of using the propagated residual/target suction, which yields a much lower DR% than the physical head balance with suction = 60 m.
- If the solver is using a higher flow (e.g., the 2,990 m³/h cap) or a different viscosity, friction losses rise to 622 m / 669 m【ed539f†L31-L35】【0e6ce0†L1-L19】, which would further increase the required DR unless suction is inflated.

This reconstruction shows that with the provided inputs (flow 2,500 m³/h, 7 cSt, suction 120 m at origin, 60 m terminal residual), the expected DR is **0%** for Paradip–Balasore and **≈18%** for Balasore–Haldia unless the inlet head at Balasore is artificially raised beyond the specified residual target.
