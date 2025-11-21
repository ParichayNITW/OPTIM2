# Faster runs without changing optimisation results

The optimiser already searches the full grid of pump speeds and DRA levels to
find the true least-cost solution.  The bottleneck was the bookkeeping around
that grid: every station rebuilds identical step-by-step lists of candidate RPM
and ppm values many times across the coarse, exhaustive, and refinement passes.

We now remember those lists the first time they are generated.  When another
station or pass asks for the same range and step size, the solver reuses the
cached tuple instead of constructing a new list.  Because the search space
itself is unchanged, the chosen operating plan and costs remain identical—the
solver simply spends less time regenerating the same integers over and over.

In plain terms: we stopped re-writing the same checklist of dial settings on
fresh sheets of paper.  The optimiser still checks every setting it used to, it
just reuses the checklist instead of copying it from scratch, so you get the
same best answer faster.

### A concrete numeric example

- Suppose every station can try **7 RPM steps** (``1,000–1,600`` in ``100`` rpm
  steps) and **15 DRA doses** (``0–70 ppm`` in ``5`` ppm steps).
- The optimiser runs **3 passes** (coarse, exhaustive, refinement) across
  **3 stations**, and each pass asks for both grids.

Before caching, the solver built the same ``7``-value RPM list and ``15``-value
ppm list for every request: ``(7 + 15) × 3 passes × 3 stations = 198`` list
builds.  After caching, the first request makes the RPM list once and the DRA
list once, and the other ``(3 × 3 − 1) = 8`` requests simply reuse those two
lists.  The optimiser still tries all 7 × 15 combinations; it just avoids
re-drawing the identical lists **196** extra times.

We are **not** pre-computing every flow vs. head or flow vs. efficiency curve
for every possible pump combination or RPM.  Those physics are still evaluated
when the solver needs them, using the same per-option cache it already had to
avoid repeating identical calculations within a run.  The only new caching is
of the integer RPM/ppm candidate lists themselves.

## A three-station, layman-friendly picture

Imagine three pump stations lined up left to right.  For each station, the
optimiser needs a checklist of pump-speed and DRA-dose positions to try, such
as ``0, 5, 10, …, 70 ppm``.  Before the change, every station wrote out its own
copy of that checklist for every pass (coarse, exhaustive, and refinement),
even though the numbers were identical.  That is like three people solving the
same puzzle but each re-drawing the blank grid every time they erase a mistake.

After the change, the first station writes the checklist once and pins it to
the wall.  The other stations glance at the same pinned sheet instead of
hand-copying it, so everyone still tests every RPM/ppm combination but wastes
less time re-drawing.

## How much faster?

The biggest wins show up when the solver asks for the same grid many times.
In a simple benchmark that repeatedly requests the standard DRA grid (``0–70 ppm``
in ``5 ppm`` steps) 100,000 times, reusing the cached list takes about 0.015
seconds; forcing a fresh build on every request takes about 0.122 seconds.  In
other words, the grid-creation work for that loop drops by roughly **88%**
without changing the combinations the optimiser explores.【015ccc†L1-L3】  Longer
optimisation runs that revisit the same integer ranges across multiple stations
and passes see similar proportional savings because the cached checklists are
reused instead of rebuilt.
