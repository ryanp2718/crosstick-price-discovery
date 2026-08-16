# crosstick-price-discovery

[![reproduce](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml)
[![security](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml)

**Does one crypto exchange lead another, and by how much?** This repo answers that for
BTC-USD on Coinbase and Kraken over 20 days of self-captured level-2 data, and it commits
the out-of-sample predictions so the answer can be checked rather than taken on trust.

[**Read the notebook**](notebooks/price-discovery.ipynb) (renders on GitHub, no setup).

Two measures, one predictive and one structural, agree that Coinbase leads:

| measure | value | what it says |
| --- | --- | --- |
| out-of-sample R2, Coinbase to Kraken, 5s horizon | **0.1035** [0.0908, 0.1120] | Coinbase order flow forecasts Kraken's next 5 seconds |
| the same experiment reversed, Kraken to Coinbase | **0.0108** [0.0081, 0.0144] | the reverse direction is a tenth as strong, intervals disjoint |
| Hasbrouck information share, Coinbase (1s bars) | **0.859** +/- 0.071 | Coinbase carries the permanent price innovation |

The forecast is cross-venue by construction: the model is given Kraken's forward return as
its target and none of Kraken's own book or flow as input, so it shares no term with what
it is predicting. The reverse run is the same 20 days, same bars, same feature builder and
same folds, with only the direction changed, which is what rules out "Kraken is simply the
slower book" as the explanation.

Run the same pipeline against a **target displaced by one hour** and it returns R2
`-0.0001` [-0.0003, 0.0001]. That null is committed alongside the real result and CI
asserts it still straddles zero, because a leakage-free pipeline is the only thing that
makes the first number mean anything.

Everything above regenerates from committed Parquet files. No database, no object store,
no Docker, no 1.5 GB feature cache.

## Run it

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --locked                                              # fetches Python 3.12 too
uv run pytest -q                                              # the published-run gates
uv run jupyter lab notebooks/price-discovery.ipynb            # read and re-run it
```

`uv sync` reads `pyproject.toml` for what is required and `uv.lock` for the one exact
resolution of it, down to transitive packages that carry no pin of their own. `--locked`
fails rather than re-resolving, so you either get that environment or a clear error.
There is nothing else to install: `.python-version` pins the interpreter and uv fetches
it, so no system Python is involved.

The notebook asserts its own headline numbers against the run records as it goes, so a
number that drifts fails the notebook rather than quietly rendering. CI executes it top to
bottom on a clean runner with no credentials and no platform running, which is what the
reproduce badge tracks. It writes to a throwaway copy, since re-executing regenerates
every execution count and every PNG and the committed notebook is the record:

```bash
uv run jupyter nbconvert --execute --to notebook \
  --output-dir "$(mktemp -d)" --output executed.ipynb \
  notebooks/price-discovery.ipynb
```

## What is in `data/`

| file | rows | what it holds |
| --- | --- | --- |
| `oos_real.parquet` | 180,195 | pooled out-of-sample rows: `date`, `y`, `threshold`, one `pred_*` per model |
| `oos_reverse.parquet` | 181,289 | the same, predicting Coinbase from Kraken |
| `oos_placebo.parquet` | 166,040 | the same, from the displaced-target run |
| `record_*.json` | | each run apart from its rows: spec, provenance, fold metrics, intervals, information shares |

One row per non-overlapping test bar, from the walk-forward's test folds only. That is
enough to recompute every headline number: pooled R2, hit rate and gross edge with
date-block bootstrap intervals, dead-zone precision, and the full coverage-matched
selectivity curve. Training rows are not published and are not needed.

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
not independent draws.

## What this does not claim

The R2 is a forecast of mid-price return, not a trading result. `gross_bps_per_trade` is
gross by name: no fees, no slippage, no queue position, and no allowance for the fact that
a 5-second edge measured on a captured tape is not a 5-second edge measured through a real
order gateway. Selectivity is reported at forced coverage to show where the signal
concentrates, not to propose a strategy.

Gonzalo-Granger component shares are computed and deliberately not quoted. Under the
conditioning of this VECM they leave the unit interval on 6 of 20 dates, which is a known
failure mode of the measure; Hasbrouck bounds are unaffected and are the number cited
above.

The window is 20 days of one instrument on two venues. It is a measurement, not a law.
