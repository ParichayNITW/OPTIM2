# Calculating Maximum Achievable Flow in a Pipeline

This note walks through how to estimate the maximum steady-state flow a pipeline can sustain given a pressure limit at the inlet and outlet. It uses only hydraulic principles and hand-calculable formulas.

## Key Relationships
- **Energy balance**: Available pressure drop drives the flow. The allowable pressure drop is the difference between inlet and outlet pressures after accounting for elevation changes.
- **Friction losses**: The Darcy–Weisbach equation relates friction head loss to flow rate: 
  
  \[ h_f = f \cdot \frac{L}{D} \cdot \frac{V^2}{2g} \]
  
  where:
  - \(h_f\): friction head loss (m)
  - \(f\): friction factor (dimensionless, from Moody chart or Colebrook-White)
  - \(L\): pipe length (m)
  - \(D\): internal diameter (m)
  - \(V\): fluid velocity (m/s), related to flow rate by \(V = Q/A\)
  - \(g\): gravitational acceleration (9.81 m/s²)
- **Minor losses**: Valves, bends, and fittings add extra loss: \(h_m = K \cdot V^2/(2g)\), where \(K\) is the sum of loss coefficients.
- **Total head loss**: \(h_{total} = h_f + h_m\) must not exceed the available head drop \(\Delta H\) from pressure limits and elevation.

## Solving for Maximum Flow
1. **Compute available head drop** (\(\Delta H\)): 
   \[ \Delta H = \frac{P_{in} - P_{out}}{\rho g} + (z_{in} - z_{out}) \]
   where \(P\) is pressure (Pa), \(\rho\) is fluid density (kg/m³), and \(z\) is elevation (m).
2. **Assume a flow rate** and find velocity \(V = Q/A\) using cross-sectional area \(A = \pi D^2/4\).
3. **Estimate friction factor** \(f\) using Reynolds number \(Re = VD/\nu\) and pipe roughness. Iterate if needed because \(f\) depends on \(V\).
4. **Calculate head losses**:
   - Friction: \(h_f = f (L/D) (V^2/2g)\).
   - Minor: \(h_m = K (V^2/2g)\).
   - Sum to get \(h_{total}\).
5. **Compare with available head**: Increase or decrease the trial flow until \(h_{total}\) matches \(\Delta H\). The corresponding \(Q\) is the maximum achievable flow without exceeding the pressure limits.

## Worked Example
Consider water at 20°C flowing through a 1 km steel pipeline with 0.3 m internal diameter. The inlet pressure is 700 kPa, outlet pressure is 200 kPa, and elevations are equal. Assume an absolute roughness of 0.000045 m and minor loss coefficient sum \(K = 5\).

1. **Available head drop**:
   \[ \Delta H = \frac{700{,}000 - 200{,}000}{1000 \times 9.81} \approx 51.0 \text{ m} \]
2. **First flow guess**: try \(Q = 0.20\) m³/s.
   - Area: \(A = \pi (0.3)^2 / 4 \approx 0.0707\) m².
   - Velocity: \(V = Q/A \approx 2.83\) m/s.
   - Reynolds number: \(Re = V D / \nu \approx 2.83 \times 0.3 / (1.0\times10^{-6}) \approx 8.5\times10^{5}\) (turbulent).
   - Relative roughness: \(\epsilon/D = 0.000045 / 0.3 \approx 1.5\times10^{-4}\).
   - Friction factor from Moody/Colebrook: \(f \approx 0.018\).
3. **Head losses at this flow**:
   - Friction: \(h_f = 0.018 \times (1000/0.3) \times (2.83^2 / (2 \times 9.81)) \approx 24.4 \text{ m}\).
   - Minor: \(h_m = 5 \times (2.83^2 / (2 \times 9.81)) \approx 2.0 \text{ m}\).
   - Total: \(h_{total} \approx 26.4 \text{ m}\), which is below the available 51 m. Flow can increase.
4. **Adjusted flow guess**: try \(Q = 0.28\) m³/s (40% higher).
   - Velocity: \(V \approx 3.96\) m/s, \(Re \approx 1.2\times10^{6}\), updated \(f \approx 0.017\).
   - Friction: \(h_f \approx 44.0 \text{ m}\).
   - Minor: \(h_m \approx 4.0 \text{ m}\).
   - Total: \(h_{total} \approx 48.0 \text{ m}\), just under \(\Delta H = 51\) m.
5. **Result**: The maximum achievable flow is about 0.28 m³/s. Pushing higher would increase losses beyond the available head, violating the pressure limits.

## Practical Notes
- Include safety margin (e.g., 5–10%) below the theoretical limit to account for temperature changes, fouling, or roughness uncertainty.
- For systems with pumps, ensure pump head curves intersect the system curve near the operating point; otherwise, adjust pump speed or configuration.
- If elevation changes are significant, compute \(\Delta H\) segment by segment and sum the head gains/losses.
- Compressible fluids require accounting for density changes; the same head-loss logic applies but with iterative density updates along the line.
