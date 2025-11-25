# DP pruning settings explained in plain English

This note breaks down the three knobs that control how many dynamic-programming (DP) states the optimizer keeps as it marches station by station. It uses a tiny pipeline example with concrete numbers so you can see what stays and what gets dropped.

## The three knobs
- **DP state (what is it?)** – A single "state" is a snapshot of one possible plan after a station: which pumps ran, how much chemical was injected, the running cost so far, the head still available, and the DRA queue carried forward.
- **Max DP states (``state_top_k``)** – The solver always keeps at least this many of the cheapest states after each station. Think of it as the size of the shortlist it refuses to shrink past.
- **DP cost margin (absolute)** – Keeps any state whose cost is within this many currency units of the best one found at the station. Example: best = ₹100,000, absolute margin = ₹5,000 → anything ≤ ₹105,000 survives.
- **DP cost margin (% of best)** – A relative safety band. It keeps states that are within this percentage of the best cost. Example: best = ₹100,000, margin = 1% → anything ≤ ₹101,000 survives. On expensive runs this percentage can be larger than the absolute margin.

## Which rule wins when they disagree?
The filters run together in a fixed order:
1) **Pareto check first.** Any state that costs more *and* gives equal-or-less head than another is dropped as dominated.
2) **Sort by cost.** Cheapest first.
3) **Keep protected entries.** Rare edge-case states tagged as protected are always kept.
4) **Keep the cheapest ``Max DP states`` no matter what.** The first ``state_top_k`` entries in cost order survive even if they sit outside the margins.
5) **Apply the bigger of the two margins.** The solver computes a threshold: ``best cost + max(absolute margin, % margin of best)``. Any remaining state under that threshold also survives, even if it is beyond the top-K list.
6) **Everything else is dropped.** This stops the search from exploding but keeps near-ties.

## Walk-through with a tiny 3-station pipeline
Assume defaults: ``Max DP states = 3``, ``DP cost margin = ₹5,000``, ``DP cost margin (% of best) = 1%``.

### Station A (best cost = ₹100,000)
- Options after Pareto pruning (cost, residual head):
  1. ₹100,000, 12m (best)
  2. ₹104,000, 11m
  3. ₹106,000, 13m
  4. ₹108,000, 10m
- Threshold = best + max(₹5,000, 1% of ₹100,000 = ₹1,000) = **₹105,000**.
- What survives?
  - "Max DP states" keeps the first three because they are the cheapest (₹100k, ₹104k, ₹106k) even though ₹106k is outside the ₹105k threshold.
  - The ₹108k option is beyond the top 3 *and* beyond the threshold, so it is dropped.
- **States carried to Station B: 3.**

### Station B (costs accumulate; best cost = ₹250,000)
- New options combine Station A carry-over with Station B choices. Suppose we get (cost, residual head):
  1. ₹250,000, 9m (best)
  2. ₹253,000, 9.5m
  3. ₹255,500, 11m
  4. ₹258,000, 12m
- Threshold = best + max(₹5,000, 1% of ₹250,000 = ₹2,500) = **₹255,000**.
- What survives?
  - The first three options are again kept by "Max DP states." Even though ₹255,500 exceeds the threshold, it stays because it sits inside the top 3.
  - The ₹258,000 option is beyond the top 3 and above the threshold, so it is dropped.
- **States carried to Station C: 3.**

### Station C (showing the percentage margin kicking in)
- Suppose this station is expensive: best cost balloons to ₹1,200,000. Options:
  1. ₹1,200,000, 8m (best)
  2. ₹1,208,000, 8.5m
  3. ₹1,213,000, 10m
  4. ₹1,218,000, 12m
- Threshold = best + max(₹5,000, 1% of ₹1,200,000 = ₹12,000) = **₹1,212,000**.
- What survives?
  - "Max DP states" keeps the first three cheapest by default.
  - The 1% margin is now larger than the absolute ₹5,000 margin, so the threshold uses **₹12,000**. That means anything up to ₹1,212,000 survives even if it is not in the top 3.
  - The fourth option (₹1,218,000) is beyond both top-3 and threshold, so it is dropped.
- **States reaching the finish: 3**, but with a wider tolerance that avoided throwing away near-ties when costs spiked.

## How they work together in practice
- The **Pareto and margin checks** prevent wasteful states (more cost for equal-or-less head) from clogging the search.
- **Max DP states** guarantees a minimum breadth: the solver never shrinks below that many cheapest candidates.
- The **absolute and % margins** add a safety halo that can keep promising near-ties outside the top-K list, and the larger of the two margins always wins so expensive runs keep a sensible buffer.
