# Design notes

Things that are not obvious from the code, and the reasons behind them.

## Why a planner hook

The first version did the obvious thing: create an empty copy of the schema, write statistics into it with
`pg_restore_relation_stats()` and `pg_restore_attribute_stats()`, and EXPLAIN.
Column statistics worked. Table sizes did not. `estimate_rel_size()` calls
`table_block_relation_estimate_size()`, which reads the current block count
from the storage manager and treats `reltuples / relpages` from pg_class as a
density to multiply it by. Zero blocks means zero tuples, and a 40M-row
table plans as `rows=1`. Indexes go through the same logic. The btree height
comes from the index metapage. These functions are built for pg_upgrade and
`pg_dump --statistics-only`, where the data files exist.

Options considered:

1. Fill the shadow tables with enough junk rows to get the right block count.
   40M rows is gigabytes per step. No.
2. Extend the relation files as sparse files, so the block count is right
   and nothing is written. It works, but only with write access to the
   shadow's data directory, and editing relation files under a running
   server is the kind of thing that breaks in confusing ways.
3. A loadable module on `get_relation_info_hook`, which runs after the
   planner has filled in `RelOptInfo` and is documented as the place to
   "editorialize" on relation sizes. It overwrites `rel->pages`,
   `rel->tuples` and `rel->allvisfrac`, and `pages`, `tuples` and
   `tree_height` for each index, from the syscache copy of pg_class.

Option 3 won. It is under 170 lines of C, needs nothing beyond `LOAD`
(superuser, or the library placed in `$libdir/plugins`), and can be switched
off per session (`flipcast_sizes.enabled`), which the
integration tests use to show the problem it fixes.

Btree height has no catalog value to copy, so the module estimates it as
`ceil(log(leaf pages) / log(fanout))`. Leaf density overstates inner fanout:
leaves are deduplicated and filled to 90%, inner pages are filled to 70% by
CREATE INDEX and keep a heap TID in each pivot when keys repeat. A 300k-row
int4 index, checked with `bt_metap()` and `bt_page_stats()`, held about 155
entries per inner page; with 8-byte keys a 70%-full inner page tops out near
285. A cap of 200 got the height right on every int4, int8 and timestamptz
index tried, from 300k to 1M rows, and with it the shadow's costs match the
real database exactly in the integration tests. One
level of error would cost `50 * cpu_operator_cost` per index descent.

## Ground truth for backtests

A snapshot records the plan each tracked query got on the source. The
backtest still uses plans replayed through the shadow as its truth. The two
agree in every case measured so far, but they answer different questions.
Replayed truth isolates the statistics forecast: shadow, settings and
planner are identical on both sides, so any disagreement is a forecasting
error. The observed plans then measure the shadow on its own ("fidelity").
Mixing the two would blame the forecast for shadow differences, or the other
way around.

## Picking models

Every statistic has a short list of candidate curves, and the choice is made
per table and per column from the history:

- Error is measured one step ahead from every origin, weighted by
  `0.5 ^ (age / 3)`, so the last few origins dominate. Early origins train on
  two or three points and say little about the curve's behavior with twelve.
- A more complex candidate has to beat every simpler one by 10%.
- With four or more points, any trend needs a slope of at least two standard
  errors over the full series or over its newer half. This came from a unit
  test: on pure noise, the weighted error alone picked a line often enough to
  matter, and twelve months of extrapolated noise is a fake switch.
- Every fitted curve is anchored on the latest observation. Without that, a
  line that misses today's value by 3% shows up as a jump between step 0 and
  step 1, and the plan "switches" in month 1.

The "accelerating" row model is `dR/dt = exp(a + b t)`, fitted on the insert
rates between snapshots and integrated from the latest row count. A plain
exponential overshoots on tables like this, because early on the base is
small and the ratio between snapshots is large. A plain line falls behind.
On the demo history the 6-month error went from about -45% (linear, recent)
to between -3% and +9%.

## MCV shares

Let `c_v = f_v * R` be the number of rows with value `v`. The marginal share
`m_v = dc_v / dR` is the share among newly added rows, and

```
f_v(T) = (c_v(now) + integral of m_v(t) dR(t) from now to T) / R(T)
```

The integral runs in 12 substeps along the projected row curve. `m_v` is
either fixed at its recent value (slope of `c` against `R` over the newer half
of the history) or trended in logit space with a damped trend: after `h`
months the trend has added `b * (phi + ... + phi^h)` with `phi = 0.85`, so it
can never add more than about 5.7 months' worth. Undamped, the demo's
`'export'` share went from 12% to 42% in a year, while the generator caps it
at 20% of new rows.

After projection the list goes through ANALYZE's rules: at most
`statistics_target` entries, sorted by frequency, and a value stays only if
it is at least 1.25x as common as an average non-MCV value (the cutoff in
`analyze_mcv_list()`, simplified). For a column whose MCV list covers every
row (no histogram), the shares are renormalized to `1 - null_frac`.

## Append histograms

For a column whose top bound rises snapshot after snapshot and that is
correlated with physical order, the projected histogram is built from a CDF
on a grid:

- old values: `P_now * F_now(v)` rows at or below `v`, where `F_now` is the
  piecewise-linear CDF of the current equi-depth buckets and `P_now` the
  current non-null, non-MCV population. If the bottom bound is moving up
  (retention), mass below the new bottom is dropped.
- new values on `(top_now, top_then]`: the share of new rows at or below `v`
  is the share of the growth that happened by the time `v` was the top. For
  timestamps that time follows the calendar, so with accelerating growth more
  rows land near the end. For ids it follows the row count, which makes the
  new segment uniform.

The new bounds are the CDF inverted at `i / buckets`. The bucket count stays
the same as the latest snapshot's.

Under the faster and slower growth scenarios, row-driven quantities run on a
warped clock (`t_eff = now + (t - now) * rate`), but timestamp columns keep
the real calendar. A `created_at` histogram can't extend into the future just
because the business grew faster.

## Query parameters

pg_stat_statements text arrives with constants as `$n`. Those statements are
planned with `EXPLAIN (GENERIC_PLAN)`, which is also what a prepared
statement runs after five executions. `interval $3`, which
pg_stat_statements produces from `interval '30 days'`, is not valid SQL, so
`type $n` is rewritten to `$n::type` first.

Config queries with parameters go through `PREPARE` and
`EXPLAIN EXECUTE` under `plan_cache_mode = force_custom_plan`, so the server
infers parameter types and the planner sees the values. A config query and a
pg_stat_statements entry with the same fingerprint (constants and parameters
folded to `?`, like pg_stat_statements does) are merged, and when the config
entry has its own parameters the generic plan is kept alongside as
`name (generic)`.

`now()`, `current_timestamp`, `current_date` and friends in query text are
rewritten to literals for the step being planned. The planner folds stable
functions into selectivity estimates at plan time, so without this a "last 7
days" query planned for month 9 would ask about the real last 7 days, which
is outside every projected histogram.

## Pricing a switch

At the switch step, the old plan's cost comes from forcing it: switch off,
via `enable_*`, the node types the new plan uses and the old one doesn't, and
check that the shape that comes back is exactly the old shape. If it isn't,
try again with every node type the old plan doesn't use switched off. Since
PostgreSQL 18, disabled nodes are counted separately instead of getting a
`1e10` cost penalty, so the forced plan's cost is its real cost. The same
trick prices the new plan with today's statistics ("if it flipped today").

## Things tried and dropped

- A logistic in time for MCV shares: overshoots on append-only tables (see
  above).
- A single power law for n_distinct against rows: for a foreign key, distinct
  values follow the parent table, and the exponent against the child keeps
  falling. By the end of the demo history it was 29% high one month ahead. A line in time
  and a power law over recent snapshots are both candidates now.
- Comparing plans by cost: plans drift in cost every step, and what breaks
  production is a change of shape. Costs are reported but never used to
  decide that a switch happened.
