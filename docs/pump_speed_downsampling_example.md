# Pump-speed downsampling example

The optimiser keeps the number of pump-speed combinations manageable by trimming
per-type RPM lists until their cartesian product falls beneath the
`REFINED_RETRY_COMBO_CAP` (256 by default).【F:pipeline_model.py†L371-L444】 The
example below walks through one concrete scenario using the exact formula from
`_cap_type_rpm_lists` and `_downsample_evenly` in layman's terms.

## Starting point

Imagine a station that can use one pump from each of three types (A, B, and C).
The hydraulic model generates a list of possible motor speeds for each type
after applying the user-selected coarse/refined RPM step sizes:

| Pump type | Available RPM values |
|-----------|----------------------|
| A         | 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950 |
| B         | 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750 |
| C         | 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650 |

With no pruning the solver would need to evaluate every combination of these
speeds:

```
10 choices for A × 10 choices for B × 10 choices for C = 1,000 combinations.
```

The `REFINED_RETRY_COMBO_CAP` allows only 256 combinations, so the optimiser
must discard some RPM points before running the dynamic-programming search.

## Step 1 – compute the target size for type A

The lists are processed in descending order of length. All three lists have the
same length (10), so type A happens to be considered first. The formula in the
code computes a new target length like this:

```
length = 10              (number of entries in type A's list)
cap = 256                (the configured cartesian-product cap)
total = 1,000            (current A×B×C combinations)

length * cap / total = 10 × 256 / 1,000 = 2.56
floor(2.56) = 2
max(2, 2) = 2 → target length for type A
```

Because the target (2) is smaller than the original length (10), `_downsample_evenly`
keeps two evenly spaced entries—always the first and last item when the target is
2. Type A is therefore reduced to `[1500, 1950]`.

The total number of combinations now becomes:

```
(2 options for A) × 10 × 10 = 200 combinations.
```

Since 200 is already below the 256 cap, types B and C are left untouched and the
loop exits early (`total <= cap` ends the while-loop in the code).【F:pipeline_model.py†L420-L444】

## Step 2 – what the solver actually tests

During the subsequent hydraulic evaluation, the optimiser tries every remaining
combination. With the downsampled list above, that means it examines the 200 RPM
triples shown schematically below:

```
A ∈ {1500, 1950}
B ∈ {2300, 2350, …, 2750}
C ∈ {1200, 1250, …, 1650}
```

The extreme points (1500 and 1950 RPM) are still included, ensuring the search
covers the lowest and highest speeds even though intermediate values such as
1650 RPM were pruned from type A's list.【F:pipeline_model.py†L371-L399】

## Why this matters

By trimming each list just enough to stay within the combination cap, the solver
avoids the cost of scoring 1,000 scenarios while still sampling the RPM range
in a structured way. The trade-off is that a discarded combination—for example,
A at 1700 RPM paired with B at 2450 RPM and C at 1350 RPM—is never evaluated.
If that specific triple would have delivered a slightly lower total cost than
any of the 200 retained combinations, the optimiser will miss it. The cap keeps
runtimes predictable, but it also means the search is approximate rather than
mathematically exhaustive.
