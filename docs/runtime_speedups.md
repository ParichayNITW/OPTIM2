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

We are **not** pre-computing every flow vs. head or flow vs. efficiency curve
for every possible pump combination or RPM.  Those physics are still evaluated
when the solver needs them, using the same per-option cache it already had to
avoid repeating identical calculations within a run.  The only new caching is
of the integer RPM/ppm candidate lists themselves.
