# Station option trimming explained with an A→B→C pipeline

This note retells the current trimming logic from `pipeline_model.py` using a
made-up but concrete pipeline:

```
Origin  A ----- 220 km ----- B ----- 210 km ----- C  Terminal
         ^ pumps here      ^ pumps here
```

Both stations (A and B) can run up to **two pumps** and can inject drag reducer
(DRA) at **five levels**: 0, 20, 40, 60, or 80 ppm.  They can also pick one of
**four motor speeds**: 1400, 1500, 1600, or 1700 RPM.  These small round numbers
make it easy to count combinations.

The optimiser follows the same seven steps for every station.  Below we walk
through those steps in plain language for Station A, then repeat the trimming
for Station B, and finally show what the dynamic-programming (DP) solver does
with the reduced lists.

## Step-by-step for one station

### Step 1 – List every raw option

*Pump counts × DRA levels × RPM choices*

For Station A the raw list has:

* 3 pump-count buckets: 0 pumps, 1 pump, 2 pumps.
* 5 DRA choices: 0, 20, 40, 60, 80 ppm.
* 4 RPM choices: 1400, 1500, 1600, 1700 (only meaningful when at least one pump runs).

For the **0-pump bucket** the only thing that changes is the chemical dose, so we
get `5 × 1 = 5` combinations (each RPM entry is the same “motor off” setting).
For the **1-pump** and **2-pump buckets** the RPM settings matter, giving
`5 × 4 = 20` combinations apiece.  Adding the buckets together produces
`5 + 20 + 20 = 45` raw options before trimming.

We draw the same 45 options for Station B.

### Step 2 – Split the list by pump count

The optimiser now keeps three separate lists for Station A:

1. **Bucket 0 pumps** – 5 entries (DRA changes, RPM stays at the single “off” value).
2. **Bucket 1 pump** – 20 entries.
3. **Bucket 2 pumps** – 20 entries.

This separation guarantees that heavy-pump entries can never shove lighter
setups out of the list when we trim.

### Step 3 – Sort each bucket by “less chemical, slower motor”

Inside a bucket, entries are ordered by four keys in this exact order:

1. Lower DRA first.
2. If DRA matches, the option that sits closest to the slowest RPM comes first.
3. If still tied, lower RPM wins.
4. Finally, lower pump count wins (this keeps a stable order between buckets).

For the 1-pump bucket at Station A, the very first entries look like this after
sorting:

| Row | Pumps on | DRA (ppm) | RPM |
|-----|-----------|-----------|-----|
| 0   | 1         | 0         | 1400|
| 1   | 1         | 0         | 1500|
| 2   | 1         | 0         | 1600|
| 3   | 1         | 0         | 1700|
| 4   | 1         | 20        | 1400|
| …   | …         | …         | …   |

High-DRA, high-RPM options end up near the bottom of each bucket, but they are
still present in the list.

### Step 4 – Take up to 24 evenly spaced entries per bucket

The hard cap is `MAX_OPTIONS_PER_PUMP_COUNT = 24`.  Our buckets hold 5, 20, and 20
entries respectively, so **nothing is dropped** at Station A.  If a bucket had,
say, 200 entries, the sampler would keep 24 of them using even spacing: rows 0,
8, 16, 24, … up to the last row.  That guarantees we keep low, mid, and high
DRA/RPM values instead of only the lowest few.

Station B keeps the same 5/20/20 split because it has the same layout.

### Step 5 – Merge the buckets back together

After the per-bucket trim, Station A now has `5 + 20 + 20 = 45` entries.
Station B also has 45 entries.  These counts are below the station-wide limit of
`MAX_OPTIONS_PER_STATION = 72`, so we do not need any extra trimming.

If a station ever exceeded 72 entries even after the per-bucket trim, the code
would sort the merged list by the same “less chemical, slower motor” rule and
again pick evenly spaced rows (0, floor(n/71), 2×floor(n/71), …, last) so the
remaining 72 still cover the full range.

### Step 6 – Pass the reduced list to the DP solver

Each station now hands **45** options to the DP stage instead of all 45 raw
entries (no change in this simple example).  In bigger real-world JSON files the
same steps routinely shrink 400–800 raw combinations to at most 72 per station.

## Putting both stations together

With the trimmed lists we now understand the size of the search space:

* Station A → 45 options.
* Station B → 45 options.
* For each of the 24 hours the DP solver picks exactly one option at Station A
  and one option at Station B.  Because the DP keeps a top-K list of states for
  each residual-pressure bucket, it still considers every combination of these
  options that could matter.

If we had not trimmed the lists in a more complicated case (say 240 options per
station), the solver would face `240 × 240 = 57,600` pairs to inspect each hour.
Capping both stations at 72 options reduces that to `72 × 72 = 5,184` pairs per
hour, which is **11× smaller**.  When the raw lists are even larger, the cut is
often 10× or 20×.  That is why the run time drops from “spinning for more than
10 minutes” to under a couple of minutes.

## Four-station example with 10 DRA steps and 2200–2900 RPM

Now let us walk through the exact scenario you asked for.  The pipeline has
four pump stations in a row.  Station 1 can run 0–3 pumps of the same type, and
Stations 2–4 can run 0–2 pumps of the same type.  Every pumping bucket (other
than the 0-pump bucket) may choose any RPM from 2200 to 2900 in 100 RPM steps,
giving eight possible speeds.  Every station also has 10 DRA dosage steps to
choose from.

### 1. Raw combinations before trimming

We count the possibilities bucket by bucket.  For the 0-pump bucket only the
DRA level changes, so RPM does not create extra rows.

* **Station 1**
  * 0 pumps → `10 × 1 = 10` combinations (ten DRA steps, motor off).
  * 1 pump → `10 × 8 = 80` combinations (ten DRA levels at eight RPM settings).
  * 2 pumps → another `10 × 8 = 80` combinations.
  * 3 pumps → another `10 × 8 = 80` combinations.
  * **Total** → `10 + 80 + 80 + 80 = 250` raw options.
* **Stations 2, 3, and 4** (all identical)
  * 0 pumps → `10 × 1 = 10` combinations.
  * 1 pump → `10 × 8 = 80` combinations.
  * 2 pumps → `10 × 8 = 80` combinations.
  * **Total** → `10 + 80 + 80 = 170` raw options at each station.

If we multiplied the station totals together we would get
`250 × 170 × 170 × 170 = 1,228,250,000` different station-option tuples to
consider every hour.  That is the huge number we must tame.

### 2. Combinations after the trimming logic

The code trims in two passes:

1. **Cap each pump-count bucket at 24 entries.**  The sampler sorts the bucket by
   “less DRA, closer to minimum RPM” and then keeps 24 evenly spaced rows.  In
   this example the 0-pump buckets stay untouched (10 entries < 24).  The 1-,
   2-, and 3-pump buckets shrink from 80 down to 24 entries apiece.
2. **Cap the whole station at 72 entries.**  After step 1, Station 1 holds
   `10 + 24 + 24 + 24 = 82` entries, so it needs an extra trim.  The sampler
   sorts the merged list and keeps 72 evenly spaced rows.  Because the zero-pump
   entries sit at the front of the ranking, nearly all of them remain, together
   with a well-spaced mix of the three pumping buckets.  The station therefore
   hands about **72 options** to the solver.

   Each of Stations 2–4 ends Step 1 with `10 + 24 + 24 = 58` entries, so they
   already sit below the station-wide cap.  No second trim is required and each
   of those stations provides **58 options**.

After trimming, Station 1 offers **72** options while Stations 2–4 offer **58**
each.  The cartesian product now has
`72 × 58 × 58 × 58 = 14,048,064` combinations—still large, but already almost
**90× smaller** than the raw list.

### 3. What happens after the K-state pruning

The dynamic-programming (DP) layer carries a short “candidate list” of partial
plans from one station to the next.  Two guardrails keep this list small:

* `STATE_TOP_K = 50` → within each residual-pressure bucket (think of a bucket
  as “still have 5 bar of slack” or “only 2 bar of slack”), only the best 50
  candidates survive.
* Global cap → even if several buckets are full, the union of all buckets never
  holds more than `max(50 × 2, 100) = 100` candidates.  In practice that means
  “never carry more than 100 partial plans into the next station.”

#### One slow pass through a single hour

1. **Before the hour starts** we have at most 100 candidates left over from the
   previous hour.  You can picture them as 100 index cards, each card showing
   “pressure left + running cost so far.”
2. **Station 1** tries all of its 72 options on every card.  Worst case,
   `100 cards × 72 options = 7,200` fresh cards are generated.  Right away the
   DP sorts those 7,200 cards by “same bucket, cheaper first” and keeps only 50
   per bucket, never more than 100 overall.  The rest go straight into the bin.
3. **Station 2** receives those ≤100 survivors, tries its 58 options, produces
   up to another 5,800 raw cards, then trims back to ≤50 per bucket and ≤100
   total again.
4. **Station 3** repeats the exact same dance with its 58 options.
5. **Station 4** does it once more with its 58 options.

After all four stations, the hour ends with ≤100 cards.  Those are the only
candidates we carry into the next hour, and the cycle repeats.

#### Tiny worked example (only 3 cards survive)

Suppose the previous hour hands the new hour just **3** candidates: one in the
“high pressure” bucket and two in the “low pressure” bucket.

* Station 1 multiplies 3 cards by 72 options → 216 trial cards.  It keeps the
  cheapest 1×50 in the high-pressure bucket and 2×50 in the low-pressure bucket,
  so still only 3 cards continue.
* If later stations invent slightly better variations, the bucket rule lets up
  to 50 cards stay in each bucket, but the global cap still prevents the list
  from crossing 100 cards.

The key idea is that **every station option is tried**; we simply refuse to keep
the obviously costly versions once we know better cards exist for the same
bucket.

#### Counting the combinations the solver really touches

With the trims in place, the worst case inside an hour is
`7,200 + (3 × 5,800) = 24,600` evaluations.  The untrimmed alternative (all
station options × all carried states) would have been hundreds of millions of
tests.  The K-state pruning therefore removes >99% of the work while still
keeping the best candidates in every bucket.

### 4. Final count of options explored for the pipeline

Across the four stations the solver touches at most 24,600 option/state pairs in
an hour after trimming and K-state pruning.  Over a 24-hour optimisation that is
`24,600 × 24 = 590,400` evaluations, which the current implementation handles
comfortably.  Every surviving option is still combined with every surviving
state, so the DP continues to search the full space defined by the trimmed
station lists and therefore can still find the global minimum within that space.

## Why the cheapest 24 h cost is still found

1. **No greedy shortcuts after trimming.**  The DP stage still expands all
   surviving options for every hour.  Trimming only shrinks the list it receives;
   it does not skip combinations within that list.
2. **Low-, mid-, and high-DRA settings remain.**  Because we sort by low DRA and
   then sample evenly, the kept rows cover the entire chemical range.  Any plan
   that truly needs more DRA or higher RPM keeps the relevant options.
3. **Protected fallback states stay.**  The DP logic separately keeps baseline
   (zero-DRA) states and maintains a per-bucket “top K” to avoid dropping nearly
   optimal paths.  With only 24–72 entries per station the DP can track all of
   them comfortably.

In short: we still examine every combination that could plausibly deliver the
lowest cost, but we avoid drowning the solver in thousands of near-duplicate
options that only slow the calculation down.

## Quick recap in plain words

*Step 1* build the full list.  *Step 2* split it by pump count.  *Step 3* sort by
“less chemical, slower motor”.  *Step 4* take up to 24 evenly spaced entries in
each bucket.  *Step 5* merge everything and, if needed, trim again to 72 using
the same even spacing.  *Step 6* give those options to the DP, which then
examines all 24-hour sequences from the reduced, evenly spread menu.

That is the complete process for the new trimming logic.
