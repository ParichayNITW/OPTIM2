# Three-station DRA handling example (plain English)

This walks through a simple pipeline with three pump stations (A, B, C) to
illustrate how drag-reducing agent (DRA) queues and downstream head (SDH)
reporting behaved before and after the latest fixes.

## Scenario setup
- The pipeline flows from station A → station B → station C.
- A DRA slug is injected at station A and travels down the line.
- Each station needs to report a *residual head* (pressure) that does **not**
  artificially increase while the slug slowly fades.

## What happened before
1. **Zero-ppm slices were put in front of treated slices.**
   - When the optimizer tried to keep the queue length consistent, it added an
     untreated (0 ppm) slice *ahead* of already-treated slices.
   - In the A → B → C example, that meant the front of the queue briefly showed
     no DRA even though the slug was actually passing the stations. SDH reports
     could jump upward and then down again as the slug “reappeared.”
2. **SDH could bump up as the slug decayed.**
   - Each solve reused no history, so if the slug weakened slightly between
     solves, station B or C might report a higher SDH than the previous solve,
     which felt wrong to operators watching a steady decay.
3. **Initial reach could exceed the pipe.**
   - The starting reach length for a fixed DRA could be longer than the total
     pipeline. In the three-station line, that made the solver assume the slug
     already spanned everything, masking the real progression.

## What changed
1. **Untreated slices now get appended, not prepended.**
   - The optimizer keeps treated portions at the front of the queue and tacks
     any necessary zero-ppm slice to the *back*. For A → B → C, the slug stays
     visible to each station in order, so SDH no longer spikes when bookkeeping
     adjusts queue lengths.
2. **SDH displays reuse prior values when DRA is present.**
   - Each pump station keeps the lowest SDH it saw while the slug is in play.
     As the slug decays past B and C, their reported SDH cannot jump upward; it
     can only hold steady or decline, matching operator expectations.
3. **Initial DRA reach is clamped to pipeline length.**
   - The solver caps the starting reach at the total A → C distance. The slug’s
     progress across the three stations is now modeled over the real length
     instead of an over-long estimate.

## Practical effect for the three stations
- Station A injects DRA and immediately sees treated flow; the queue keeps that
  treated slice at the front.
- Station B receives the slug next. Its SDH history ensures the reported head
  cannot rise above the prior solve while the slug is present, even as it fades.
- Station C experiences the same smooth decay. With the reach capped to A → C,
  the slug’s travel timing is realistic and SDH stays monotonic.

Together, these changes keep the optimizer from “losing” the slug at the front
of the queue, stop SDH from bouncing up as the slug decays, and keep the DRA
reach aligned with the actual three-station pipeline length.
