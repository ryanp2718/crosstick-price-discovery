# crosstick-price-discovery

[![reproduce](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml)
[![security](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml)

**Does one crypto exchange lead another, and by how much?** This repo answers that for
BTC-USD on Coinbase and Kraken over 20 days of self-captured level-2 data, and it commits
the out-of-sample predictions and the prices so the answer can be checked rather than taken
on trust.

[**Read the essay**](notebooks/price-discovery.ipynb), and the [appendix](notebooks/appendix.ipynb)
for the checks behind it (both render on GitHub, no setup).

Three measurements agree that Coinbase leads, in the sense that its price moves first and
Kraken's follows:

| measure | value | what it says |
| --- | --- | --- |
| Kraken's move in the 10s after a Coinbase jump of 2bp or more | **91%** of Coinbase's (69% to 96% by day) | Kraken catches up to Coinbase's jumps |
| out-of-sample R2, Coinbase to Kraken, 5s horizon | **0.1035** [0.0908, 0.1120] | Coinbase's order book forecasts Kraken's next 5 seconds |
| the same experiment reversed, Kraken to Coinbase | **0.0108** [0.0081, 0.0144] | paired on the 12 shared dates, dR2 **0.0927** [0.0785, 0.1026] |
| Hasbrouck information share, Coinbase (1s bars) | bounds **[0.754, 0.997]** | most of the lasting price change starts at Coinbase |

Why Coinbase moves first, whether news reaches its traders first or Kraken is slower to
update its quotes, is a separate question that prices alone cannot answer. And the forecast
is of the mid-price, before fees, the spread or fills, so it is not a trading result.

The forecast sees none of Kraken's own book or flow, so it shares no term with what it
predicts. Two controls test it. With the target displaced by an hour, the same pipeline
returns R2 `-0.0001` [-0.0003, 0.0001], and CI asserts it still straddles zero. Requiring
both books to be current within **200ms** rather than five seconds keeps 54% of the bars
and moves the forward result from `0.1035` to `0.1028`, with the asymmetry intact at every
cut in between. The [appendix](notebooks/appendix.ipynb) has both in full, with the rest of
the checks.

Everything above regenerates from committed Parquet files. No database, no object store,
no Docker, no 1.5 GB feature cache.

## Run it

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --locked                                              # fetches Python 3.12 too
uv run pytest -q                                              # the published-run gates
uv run jupyter lab notebooks/                                 # read and re-run both
```

`uv sync` reads `pyproject.toml` for what is required and `uv.lock` for the one exact
resolution of it, down to transitive packages that carry no pin of their own. `--locked`
fails rather than re-resolving, so you either get that environment or a clear error.
There is nothing else to install: `.python-version` pins the interpreter and uv fetches
it, so no system Python is involved.

Both notebooks assert the numbers their text quotes as they go, and the appendix recomputes
every number in the run records, so a number that drifts fails the notebook rather than
quietly rendering. CI executes both top to bottom on a clean runner with no credentials and no
platform running, which is what the reproduce badge tracks. It writes to throwaway copies,
since re-executing regenerates every execution count and every PNG and the committed notebooks
are the record:

```bash
uv run jupyter nbconvert --execute --to notebook --output-dir "$(mktemp -d)" \
  notebooks/price-discovery.ipynb notebooks/appendix.ipynb
```

## What is in `data/`

| file | rows | what it holds |
| --- | --- | --- |
| `oos_real.parquet` | 180,195 | pooled out-of-sample rows: `date`, `y`, `threshold`, one `pred_*` per model |
| `oos_reverse.parquet` | 181,289 | the same, predicting Coinbase from Kraken |
| `oos_placebo.parquet` | 166,040 | the same, from the displaced-target run |
| `record_*.json` | | each run apart from its rows: spec, provenance, fold metrics, intervals, information shares |
| `freshness/record_*.json` | | six more runs varying only the staleness cut, records only |
| `prices.parquet` | 1,529,334 | both venues' mid at every one-second bar the real run kept: `date`, `ts_ns`, `coinbase_mid`, `kraken_mid` |
| `prices_retimed.parquet` | 547,482 | the same bars on other clocks, each venue at the exchange's own stamp and Kraken moved 20 to 200ms earlier, kept only where a series differs from `prices.parquet`: `date`, `ts_ns`, `series`, `mid` |
| `clocks.json` | | each venue's delay from its own stamp to arrival, by day, and the test of what that stamp marks |

The `oos_*` files hold one row per non-overlapping test bar, from the walk-forward's test
folds only. That is enough to recompute every headline number: pooled R2, hit rate and
gross edge with date-block bootstrap intervals, dead-zone precision, and the full
coverage-matched selectivity curve. Training rows are not published and are not needed.

`prices.parquet` is what the run's information shares were fitted on. It was exported from
crosstick's local feature cache by `scripts/export_prices.py`, picking each date's cache entry
by the source and builder fingerprints in the real run's record, and it carries those
fingerprints in its own metadata. Refitting with crosstick's estimator returns every estimate
in the record at every stride, which the tests assert.

That estimator fits straight across holes in the recording, as if the prices on either side
were one second apart. The essay and the appendix skip those steps instead, which moves the
one-second bounds from the record's [0.720, 0.997] to [0.754, 0.997], and the appendix shows
both.

`prices_retimed.parquet` and `clocks.json` were read from crosstick's lake by
`scripts/extract_clocks.py` and written by `scripts/export_clocks.py`, which refuses to write
unless the arrival clock, rebuilt from the same tables, equals `prices.parquet` bit for bit.

## How the numbers were produced

The capture platform, feature build and modelling code live in
[ryanp2718/crosstick](https://github.com/ryanp2718/crosstick). The analysis half of that
package is installed here from an exact commit, so the code that reads these files is the
code that wrote them:

```
crosstick[research] @ git+https://github.com/ryanp2718/crosstick@fb7a22d
```

All three runs were recorded from a clean tree at that commit and carry the same feature
builder (`8f96b676`), the same feature schema (`9c0eae5c`) and the same 20 silver source
fingerprints. `tests/test_published_runs.py` asserts all of it, using crosstick's own
definition of what makes a placebo and a reverse run comparable rather than a restatement
of it.

**Method, briefly.** One-second bars, 20 trading days, 1.53M usable rows after a 5000 ms
staleness cut. 27 features: order-flow imbalance, depth, spread and trailing returns, built
point-in-time from the predicting venue only. Target is the predicted venue's mid return 5
bars ahead. Walk forward with an 8-day minimum train and 3-day test blocks, 4 folds, no
shuffling, an embargo at every boundary and early stopping on its own embargoed slice.
Intervals are date-block bootstraps over the 12 test dates, because adjacent seconds are
not independent draws. The information share is fitted on each day's one-second prices
separately, using only the seconds whose ten seconds before have no hole in the recording.
