# flipcast

Forecasts when your Postgres query plans will switch as the tables grow.

Postgres picks a plan from table statistics. Those statistics drift as data
piles up, and one day a query crosses over to a different plan. Usually the
new plan is fine. Sometimes it is much slower and you find out in production.
On February 19, 2026, Clerk was down for about 90 minutes after an automatic
ANALYZE left the planner thinking a column held almost no non-null values; the
query returned over 17,000 rows ([postmortem](https://clerk.com/blog/2026-02-19-system-outage-postmortem)).

flipcast snapshots the planner's statistics on a schedule, fits a model of
how each one changes, writes the projected statistics for each future month
into an empty copy of the schema, and asks the planner what it would do then.
It also checks which plans could flip today from ANALYZE sampling noise alone,
which is the Clerk case. Output from the demo database:

```
  month  9 (9-10)  events_of_kind (generic)
    select id, account_id, created_at from events where kind = $1
    switches from a bitmap scan to a sequential scan on events around month 9, at roughly 8.5M rows in events (2027-07-29)
      events: bitmap scan (events_kind_idx) -> seq scan
    rows: events 1.4M -> 8.5M
    estimated cost at the switch: old plan 254,927, new plan 255,077
    if it flipped today (an unlucky ANALYZE sample can do that): 1.4x the cost, 30,662 -> 41,448
    latency, scaled by cost from pg_stat_statements: ~40.04 ms now, ~333.1 ms at the switch
    what to do:
      ALTER SYSTEM SET random_page_cost = 1.1;  -- SSD/NVMe storage only
        random_page_cost is still the spinning-disk default of 4. on SSDs random reads cost about the same as sequential ones
        checked in the shadow: switch gone through month 12; cost at month 9: 255,077 -> 193,773
      CREATE INDEX CONCURRENTLY ON public.events (kind) INCLUDE (id, account_id, created_at);
        about 20.0% of events matches by then, past the point where fetching heap pages one by one beats reading the table. a covering index allows an index-only scan, which skips the heap for all-visible pages
        checked in the shadow: does not help, the switch still happens at month 9
```

"9-10" is the same forecast run with growth 10% faster and 10% slower. Every
suggested index or setting gets tested in the shadow before it is printed,
which is why the covering index above says it doesn't help.

## Quick start

You need Docker and Python 3.11+.

```bash
make dev
```

```bash
make demo
```

`make demo` starts PostgreSQL 18 with the flipcast module on localhost:5433,
grows a demo database through 14 simulated months (one snapshot per month),
then runs a backtest and a forecast. Once the image is built that takes
about half a minute.

The same steps by hand:

```bash
docker compose up -d --wait
```

```bash
flipcast demo --dsn postgresql://postgres@localhost:5433/postgres --pg-dump "docker compose exec -T postgres pg_dump"
```

```bash
flipcast -c flipcast-demo/flipcast.toml backtest
```

```bash
flipcast -c flipcast-demo/flipcast.toml forecast
```

## On a real database

1. Run the shadow server somewhere that is not your production database:
   `docker compose up -d` gives you one, or build `docker/Dockerfile`. It is
   stock PostgreSQL 18 plus a small planner hook (see below). It only ever
   holds empty tables.
2. Copy `examples/flipcast.toml`, set the two DSNs, and list the queries you
   care about with realistic parameters. The top statements from
   pg_stat_statements are picked up on their own.
3. Run `flipcast check`. It tests both connections, pg_dump, the module, and
   a full schema and stats load.
4. Run `flipcast collect` once a day from cron. Snapshots take about 0.7 KB
   per column gzipped (the demo's 10 columns come to 7 KB), so around 1 MB a
   day for 100 tables of 15 columns.
5. After a few weeks of snapshots, run `flipcast backtest` to see how well
   the forecast would have done on your own history, then `flipcast forecast`.

`flipcast forecast --fail-within 3` exits with status 3 if any switch is
predicted within three months, so it can run as a scheduled CI job. There is
also `--format markdown` and `--format json`.

The source can run PostgreSQL 14 or later. The shadow has to be 18, because
it writes statistics with `pg_restore_relation_stats()` and
`pg_restore_attribute_stats()`, which are new in 18. If the source runs an
older major version, the forecast shows what the 18 planner would do, and the
backtest prints how often that matched what the source actually chose.

The collector only reads. pg_stats hides columns the role can't `SELECT`, so
give it `pg_read_all_data` (or SELECT on the tables), plus `pg_read_all_stats`
to see other roles' statements in pg_stat_statements. It runs plan-only
`EXPLAIN` on the tracked queries with a 2 second lock timeout.

The design notes in [docs/design.md](docs/design.md) cover the decisions
behind the shadow, the models and the backtest in more detail.

## How it works

### Snapshots

`flipcast collect` stores, per snapshot: `relpages`, `reltuples` and
`relallvisible` for every table and index; every `pg_stats` row (null
fraction, n_distinct, most common values and their frequencies, histogram
bounds, correlation, plus the array and range statistics); the planner
settings; the schema as `pg_dump --schema-only` prints it; the top
statements from pg_stat_statements; and the plan each tracked query gets at
that moment. That last part is the ground truth for backtesting.

### Growth models

Each statistic gets a few candidate models, fitted on the snapshot history.
The candidate is picked by rolling-origin error: fit on the first k
snapshots, predict snapshot k+1, repeat, with recent origins weighted more.
In-sample fit would always pick the most flexible curve, which is the wrong
choice for extrapolation. A trend also has to be statistically there (a
slope of at least two standard errors) before it is used, so noise around a
flat value stays flat.

| statistic | candidates |
|---|---|
| row count | constant, linear, exponential, the last two on recent snapshots only, and "accelerating": the insert rate grows exponentially and the table is its running sum |
| pages | linear in rows |
| null fraction, correlation | constant, linear, linear over recent snapshots |
| n_distinct | constant, power law in rows (all or recent snapshots), linear in time |
| MCV frequencies | constant, steady mix, shifting mix (see below) |
| histogram | append, drift, static |

Two of these need more explanation.

Most common values are modeled through counts, not shares. If new rows are
18% `'export'` and old ones were 3%, the overall share climbs but can never
pass 18%, however long you wait. A trend on the share itself doesn't know
that. flipcast fits the share among new rows (the marginal share), lets it
trend with Holt-style damping, and integrates it over the projected growth.
The rebuilt MCV list then goes through the same cutoff ANALYZE uses, so
values that stop being common drop off the list, the way they would for
real.

Histograms on columns whose top bound keeps rising (ids, `created_at`) use
the append model. The old buckets keep their shape, and the new rows are
spread over the extended range following the row growth curve. With
accelerating growth that puts more of the histogram on recent values,
which is what makes "last 7 days" queries get more expensive month after
month. Timestamp columns extend with the calendar; id columns extend with
the row count.

Parameters move with the steps too. `now() - interval '7 days'` in a tracked
query is rewritten to the projected date, and a parameter like
`"mcv:orders.customer_id:10"` is looked up in the projected statistics at
each step.

### The shadow, and why it needs a C module

The obvious plan is: empty copy of the schema, autovacuum off, inject the
projected stats with the new PostgreSQL 18 functions, run EXPLAIN. That
does not work on its own. The planner does not take `relpages` from
`pg_class` as the table size. It asks the storage manager how many blocks
the table has right now and only uses `reltuples / relpages` as a density
(`table_block_relation_estimate_size()` in `tableam.c`). An empty table has
zero blocks, so it plans as zero rows whatever the catalog says:

```
-- after pg_restore_relation_stats(... 'relpages', 500000, 'reltuples', 4e7)
explain select * from ev where user_id = 42;
 Seq Scan on ev  (cost=0.00..0.00 rows=1 width=52)
```

`ext/flipcast_sizes` is under 170 lines of C that hook `get_relation_info`
and replace the size estimates with the catalog values, for tables and their
indexes. It also guesses btree heights, because the real height lives in the
index metapage, which is empty here. The shadow sessions `LOAD` it. With it,
the same query plans as 40M rows:

```
 Bitmap Heap Scan on ev  (cost=18.64..3132.64 rows=800 width=52)
```

Replaying real statistics through the shadow gives the same plan shapes and
the same costs as the real database, to two decimal places, in the integration tests
(`tests/integration/test_shadow.py`). In the demo backtest, 94 of 94 replayed
plans matched what the source chose.

Autovacuum is off on every shadow table (`autovacuum_enabled = off`), so
nothing re-analyzes the empty tables and wipes the injected stats.

### Switches and fixes

For each step, the shadow gets that step's statistics and every query is
planned. A switch is a change of plan shape: node types, relations, indexes
and tree structure, ignoring costs and row counts.

For each switch flipcast also reports:

- the cost of both plans at the switch, by forcing the old one with `enable_*`
  settings. Near a crossover they are close, which is the point: the planner
  is choosing between two plans it thinks are equal.
- what it would cost if the planner flipped today. That is the number to
  worry about, because an unlucky ANALYZE sample can move a query across the
  line early.
- a latency estimate, scaling the current mean time from pg_stat_statements
  by the cost ratio. Rough, and labeled as such.
- fixes. Rules read off the two plans suggest a composite index for the
  LIMIT + ORDER BY trap, a covering index, a higher statistics target when
  the MCV list is full, `random_page_cost`, or `plan_cache_mode`. Index and
  setting fixes are applied in the shadow and the sweep is re-run, so each one
  comes back with "switch gone through month 12", "moves out to month N" or
  "does not help".

### Sampling noise

ANALYZE reads 300 x `default_statistics_target` rows, so every fraction it
stores has sampling error. For each query, flipcast nudges the stats of the
columns its plan filters on by about two standard errors of a sample that
size (null fraction, MCV shares, correlation), halves and doubles n_distinct
(whose estimator is much less stable), and re-plans. It also tries the case
where the sample saw almost no non-null values. Queries whose plan changes
are listed, with what the flipped plan costs under the real statistics.
`tests/integration/test_fragility.py` builds a mostly-NULL column where a
hash join turns into a nested loop that way.

## Backtesting

`flipcast backtest` does the forecast again from every point in the
history: fit on snapshots 0..k, project to the dates of the later snapshots,
and compare with the plans those snapshots actually got. A switch counts as
caught when the predicted date is within 45 days and the new plan is the same
one. The same run is done for a baseline that only scales row and page counts
and leaves every column statistic alone.

On the demo database (14 monthly snapshots, 10 origins, three runs; the
data is seeded, ANALYZE's sampling is not):

| | flipcast | rows-only baseline |
|---|---|---|
| plan shape right, all queries and steps | 97.2-97.5% | 85.4% |
| real switches caught | 13-14 of 14 | 0 of 14 |
| predicted switches that were real | 93-100% | 0 of 4 |
| timing error on caught switches | 21-24 days | |
| row count error, 1 / 3 / 6 months ahead | 1.3% / 4.5% / 13% | same |

Take these with salt. The demo generator grows its events table with a
geometric insert rate, which is exactly the shape of one of the row-count
candidates, so this is a friendly case. The misses that remain are a generic
plan sitting on a near-tie between an index scan and a bitmap scan, decided
by a drift in the correlation estimate about as large as its sampling noise.
The noise check flags that query as one ANALYZE away from flipping. Your own
history, after a few weeks of `flipcast collect`, is the test that counts.

## Known problems

- Extended statistics (`CREATE STATISTICS`) can't be injected until
  PostgreSQL 19, so the shadow plans as if they didn't exist. Queries that
  depend on them (correlated filter columns, mostly) will plan differently
  in the shadow than in production. The backtest's fidelity line shows it.
- When a constant falls past the end of a histogram, the planner probes the
  real index for the actual min or max (`get_actual_variable_range`). The
  shadow's indexes are empty, so it uses the histogram bound. Plans for such
  queries match, but costs can be slightly off.
- Btree heights are guessed from size, with inner pages assumed to hold 200
  entries. One level off moves an index probe's cost by
  `50 * cpu_operator_cost`.
- The schema is today's. New partitions, new indexes and dropped columns in
  the future are not modeled. Snapshots before a schema change are kept out
  of backtest comparisons that span it.
- Growth models assume the past continues. A launch, a backfill or a data
  retention job breaks that. The month ranges cover uncertainty in the
  growth rate, not a wrong model.
- Statements from pg_stat_statements arrive with their constants replaced
  by `$n`, so they get generic plans. List the important ones in the config
  with parameters to see custom plans too.
- Costs are planner cost units, not time. The latency line is a scaling of
  measured time by cost and ignores caching.

## Layout

```
ext/flipcast_sizes/   planner hook: catalog sizes for tables and indexes
src/flipcast/
  collect.py          snapshots from the source
  growth.py           per-table and per-column growth models
  share.py            marginal-share model for MCV frequencies
  fit.py              least squares, candidate curves, rolling-origin selection
  shadow.py           schema copy, autovacuum off, stats restore
  sweep.py            plan every query at every step, find switches
  fragility.py        sampling-noise check
  advise.py           fix suggestions
  forecast.py         ties it together, prices switches, checks fixes
  backtest.py         replay history and score
  report.py           text, markdown, json
  demo.py             demo database with monthly snapshot history
tests/unit/           no database needed
tests/integration/    needs PostgreSQL 18 with flipcast_sizes
```

## Development

```bash
make dev
```

```bash
make lint
```

```bash
make test
```

```bash
make up
```

```bash
make itest
```

`make itest` runs against the compose server. To use another one, set
`FLIPCAST_TEST_DSN` and `FLIPCAST_TEST_PG_DUMP` (a PostgreSQL 18 pg_dump
command). CI builds the Docker image, runs both suites, and runs the demo,
backtest and forecast; the reports are uploaded as build artifacts.

To build the module outside Docker you need the server headers
(`postgresql-server-dev-18` on Debian and Ubuntu):

```bash
make -C ext/flipcast_sizes PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config install
```
