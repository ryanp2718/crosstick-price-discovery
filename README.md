# crosstick-price-discovery

[![reproduce](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/reproduce.yml)
[![security](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml/badge.svg)](https://github.com/ryanp2718/crosstick-price-discovery/actions/workflows/security.yml)

**Does one crypto exchange lead another, and by how much?** This repo answers that for
BTC-USD on Coinbase and Kraken over 20 days of self-captured level-2 data, and it commits
the out-of-sample predictions so the answer can be checked rather than taken on trust.

Two independent measures agree that Coinbase leads:

| measure | value | what it says |
| --- | --- | --- |
| out-of-sample R2, Coinbase features to Kraken 5s return | **0.1035** [0.0908, 0.1120] | Coinbase order flow forecasts Kraken's next 5 seconds |
| permutation importance, Coinbase block | **100%** of total | dropping Kraken's own features costs nothing |
| Hasbrouck information share, Coinbase (1s) | **0.859** +/- 0.071 | Coinbase carries the permanent price innovation |

The same pipeline run against a **displaced target** returns R2 `-0.0001`
[-0.0003, 0.0001]. That null is committed alongside the real result, and CI asserts it
still straddles zero, because a leakage-free pipeline is the only thing that makes the
first number mean anything.

Everything above regenerates from two committed Parquet files. No database, no object
store, no Docker, no 1.5 GB feature cache.

## Run it

```bash
pip install -r requirements-dev.txt
pytest -q
```

That recomputes the headline numbers from the committed rows, checks the two runs are the
same experiment, and asserts the placebo still finds nothing. It is what the reproduce
badge tracks, and it needs no credentials and no running platform.

The walkthrough notebook lands in `notebooks/` next, and will be executed by the same
workflow so a reader knows its numbers were reproduced rather than pasted.

## What is in `data/`

| file | rows | what it holds |
| --- | --- | --- |
| `oos_real.parquet` | 180,195 | pooled out-of-sample rows: `date`, `y`, `threshold`, one `pred_*` per model |
| `oos_placebo.parquet` | 166,040 | the same, from the displaced-target run |
| `record_real.json` | | the whole run except the rows: spec, provenance, fold metrics, intervals, information shares |
| `record_placebo.json` | | the same, for the null |

One row per non-overlapping test bar, from the walk-forward's test folds only. That is
enough to recompute every headline number: pooled R2 and hit rate with date-block
bootstrap intervals, dead-zone precision, and the full coverage-matched selectivity curve.
Training rows are not published and are not needed.

## How the numbers were produced

The capture platform, feature build and modelling code live in
[ryanp2718/crosstick](https://github.com/ryanp2718/crosstick). The analysis half of that
package is installed here from an exact commit, so the code that reads these files is the
code that wrote them:

```
crosstick[research] @ git+https://github.com/ryanp2718/crosstick@fb7a22d
```

Both records were written from a clean tree at that commit and carry the same feature
builder (`8f96b676`), the same feature schema (`9c0eae5c`) and the same 20 silver source
fingerprints. `tests/test_published_pair.py` asserts all of it, using crosstick's own
definition of what makes a placebo comparable rather than a restatement of it.

**Method, briefly.** One-second bars, 20 trading days, 1.53M usable rows after a 5000 ms
staleness cut. 27 features: order-flow imbalance, depth, spread and trailing returns,
built point-in-time from both venues. Target is Kraken's mid return 5 bars ahead. Walk
forward with an 8-day minimum train and 3-day test blocks, 4 folds, no shuffling. Intervals
are date-block bootstraps over the 12 test dates, because adjacent seconds are not
independent draws.

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
