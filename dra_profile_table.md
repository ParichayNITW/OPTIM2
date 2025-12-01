# DRA profile walk-through (Paradip → Balasore → Haldia)

Assumptions used to mirror the provided scenario:
- Pipe ID = 0.746 m (cross-sectional area ≈ 0.4370866443 m²).
- Movement = 5.88 km per hour with pumps always on; shear factor = 1 for all pumps.
- Paradip injects 6 ppm each hour into the hourly pumped length (5.88 km added at the upstream end); Balasore injects 0 ppm.
- Baseline floor does not alter queue ppm values; only actual injections change ppm.
- Linefill at 07:00 (ordered from Paradip toward Haldia):
  - m1: 68,220 m³ → 156.0789 km at 6.26 ppm
  - m2: 31,484 m³ → 72.0315 km at 0 ppm
  - m3: 39,877 m³ → 91.2336 km at 0 ppm
  - m4: 3,957 m³ → 9.0531 km at 5 ppm
- Segment lengths: Paradip→Balasore = 158 km; Balasore→Haldia = 170 km.
- At each hour: remove 5.88 km from the downstream tail (delivered), then prepend a 5.88 km slug at 6 ppm from Paradip.

## Segment profiles by hour
### 07:00 (initial linefill)
- **Paradip→Balasore (158 km window):**
  - 156.079 km @ 6.26 ppm (m1)
  - 1.921 km @ 0 ppm (front of m2)
- **Balasore→Haldia (170 km window):**
  - 70.110 km @ 0 ppm (rest of m2)
  - 91.234 km @ 0 ppm (m3)
  - 9.053 km @ 5 ppm (m4)

### 08:00 (after 5.88 km pumped, new 5.88 km @ 6 ppm added upstream)
- **Paradip→Balasore (158 km window):**
  - 5.880 km @ 6 ppm (new injection at Paradip)
  - 152.120 km @ 6.26 ppm (remaining m1)
- **Balasore→Haldia (170 km window):**
  - 3.959 km @ 0 ppm (tail of m1 stripped to 0 ppm by Balasore pumps; shear factor = 1)
  - 72.031 km @ 0 ppm (m2)
  - 91.234 km @ 0 ppm (m3)
  - 3.173 km @ 5 ppm (m4 after 5.88 km delivery)

### 09:00 (after another 5.88 km pumped, second 5.88 km @ 6 ppm added upstream)
- **Paradip→Balasore (158 km window):**
  - 5.880 km @ 6 ppm (09:00 injection at Paradip)
  - 5.880 km @ 6 ppm (08:00 injection now partway down the segment)
  - 146.240 km @ 6.26 ppm (remaining m1)
- **Balasore→Haldia (170 km window):**
  - 9.839 km @ 0 ppm (tail of m1 stripped to 0 ppm by Balasore pumps; shear factor = 1)
  - 72.031 km @ 0 ppm (m2)
  - 88.527 km @ 0 ppm (m3 after 2.707 km delivered)
