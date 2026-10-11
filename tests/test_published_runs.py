"""The published runs, checked against themselves.

Three distinct claims are asserted here, and they fail for different reasons.

The first is that the files are what they say they are: written from a clean tree, over
the same days, by the same feature builder. Without that, the placebo nulls out a
different question than the one the real run asked, and the reverse direction measures a
different window rather than a different direction. Once the files sit side by side in
`data/` nothing downstream can tell. `research.export` already defines which fields make
two runs the same experiment, so that definition is imported rather than restated.

The second is the leakage gate. The pipeline that produced R2 0.1035 on the real target
produces R2 -0.0001 on a target displaced by an hour, and the interval straddles zero.

The third is the finding itself: the two directions do not overlap. Both are recomputed
here from the committed rows rather than read from the records, because reading a record
back would only prove the record is a file.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from research.export import EXPERIMENT
from research.infoshare import information_shares
from research.runs import read_oos
from research.validation import confidence_intervals

DATA = Path(__file__).resolve().parents[1] / "data"
TAGS = ("real", "reverse", "placebo")

# The recorded baselines the notebook and the README quote. The bootstrap is seeded, so
# these are equality checks with room for float drift across platforms, not tolerances
# chosen to let a moving number pass.
REAL_R2 = (0.10349, 0.09075, 0.11202)
REAL_HIT = (0.75719, 0.73511, 0.78052)
REVERSE_R2 = (0.01076, 0.00814, 0.01437)
PLACEBO_R2 = (-0.00010, -0.00028, 0.00011)
TOLERANCE = 5e-5


def record(tag: str) -> dict:
    return json.loads((DATA / f"record_{tag}.json").read_text(encoding="utf-8"))


def oos(tag: str, model: str = "gbt"):
    return read_oos(pl.read_parquet(DATA / f"oos_{tag}.parquet"))[model]


@pytest.mark.parametrize("tag", TAGS)
def test_record_has_provenance(tag: str) -> None:
    provenance = record(tag)["provenance"]
    assert not provenance["git_dirty"], "recorded from a dirty tree, so its sha is not its code"
    assert provenance["git_sha"]
    assert len(provenance["source_fingerprints"]) == len(record(tag)["spec"]["dates"])


@pytest.mark.parametrize("tag", TAGS)
def test_every_run_shares_one_data_vintage(tag: str) -> None:
    """A difference between two runs has to be the thing under test, not the silver
    underneath them moving between them."""
    assert record(tag)["provenance"]["source_fingerprints"] == (
        record("real")["provenance"]["source_fingerprints"]
    )


@pytest.mark.parametrize("tag", ("reverse", "placebo"))
def test_every_run_shares_a_feature_builder(tag: str) -> None:
    real, other = record("real")["provenance"], record(tag)["provenance"]
    assert real["builder_fingerprint"] == other["builder_fingerprint"]
    assert real["feature_schema_digest"] == other["feature_schema_digest"]


def test_the_placebo_is_the_same_experiment() -> None:
    real, placebo = record("real")["spec"], record("placebo")["spec"]
    assert not real["placebo"] and placebo["placebo"]
    assert {field: real[field] for field in EXPERIMENT} == {
        field: placebo[field] for field in EXPERIMENT
    }


def test_the_reverse_run_differs_only_in_direction() -> None:
    """Same window, same bars, same folds. Only the venue being predicted moves, and the
    target and spread column move with it."""
    real, reverse = record("real")["spec"], record("reverse")["spec"]
    assert not real["placebo"] and not reverse["placebo"]
    differing = {field for field in EXPERIMENT if real[field] != reverse[field]}
    assert differing == {"predict_venue", "target", "spread_col"}


def test_the_placebo_finds_no_signal() -> None:
    """The gate: displace the target and the whole pipeline has to go quiet."""
    point, low, high = confidence_intervals(oos("placebo"))["r2_vs_zero"]
    assert low < 0 < high, f"placebo R2 interval misses zero: [{low:.5f}, {high:.5f}]"
    assert (point, low, high) == pytest.approx(PLACEBO_R2, abs=TOLERANCE)


def test_the_two_directions_do_not_overlap() -> None:
    """The finding. A symmetric mechanical relationship cannot produce a gap this wide."""
    forward = confidence_intervals(oos("real"))["r2_vs_zero"]
    reverse = confidence_intervals(oos("reverse"))["r2_vs_zero"]
    assert forward == pytest.approx(REAL_R2, abs=TOLERANCE)
    assert reverse == pytest.approx(REVERSE_R2, abs=TOLERANCE)
    assert reverse[2] < forward[1], f"intervals overlap: {reverse[2]:.5f} >= {forward[1]:.5f}"


def test_the_real_headline_recomputes() -> None:
    pooled = confidence_intervals(oos("real"))
    assert pooled["r2_vs_zero"] == pytest.approx(REAL_R2, abs=TOLERANCE)
    assert pooled["hit_rate"] == pytest.approx(REAL_HIT, abs=TOLERANCE)


@pytest.mark.parametrize("tag", TAGS)
def test_rows_match_the_record(tag: str) -> None:
    """The record counts rows it no longer carries, so the two can disagree."""
    assert len(oos(tag).y) == record(tag)["n_oos"]


# ---- the freshness sweep ----------------------------------------------------------
# The forecast is only evidence about Coinbase's flow if it is not instead reading a
# Kraken book that had gone stale. Tightening the cut is the direct test of that, and it
# is published as records rather than rows: six more prediction files would roughly
# double this repo to support one robustness point.
FRESHNESS = DATA / "freshness"
CUTS = (200, 500, 1000)


def sweep(venue: str, cut: int) -> dict:
    return json.loads((FRESHNESS / f"record_{venue}_{cut}.json").read_text(encoding="utf-8"))


def sweep_r2(venue: str, cut: int) -> list[float]:
    return sweep(venue, cut)["pooled"]["gbt"]["r2_vs_zero"]


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("venue", ("kraken", "coinbase"))
def test_the_sweep_is_the_same_experiment(venue: str, cut: int) -> None:
    """A different builder or a different vintage would make the sweep a different
    experiment at each cut, which is exactly what it is meant to hold fixed."""
    p, real = sweep(venue, cut)["provenance"], record("real")["provenance"]
    assert not p["git_dirty"]
    assert p["git_sha"] == real["git_sha"]
    assert p["builder_fingerprint"] == real["builder_fingerprint"]
    assert p["feature_schema_digest"] == real["feature_schema_digest"]
    assert p["source_fingerprints"] == real["source_fingerprints"]


@pytest.mark.parametrize("cut", CUTS)
def test_the_sweep_varies_only_the_staleness_cut(cut: int) -> None:
    forward, headline = sweep("kraken", cut)["spec"], record("real")["spec"]
    differing = {f for f in EXPERIMENT if forward[f] != headline[f]}
    assert differing == {"max_age_ms"}


@pytest.mark.parametrize("cut", CUTS)
def test_the_asymmetry_survives_every_cut(cut: int) -> None:
    """At 200ms both books must be current within a fifth of a second, which costs 46%
    of the bars. If the result were stale data read back, it would not survive that."""
    forward, reverse = sweep_r2("kraken", cut), sweep_r2("coinbase", cut)
    assert reverse[2] < forward[1], f"intervals overlap at {cut}ms"
    assert forward[0] > 5 * reverse[0], f"asymmetry collapses at {cut}ms"


def test_the_forward_result_is_flat_across_cuts() -> None:
    """A 25x change in how fresh a book must be moves the headline by under a point of
    R2. That is the claim the sweep exists to make."""
    points = [sweep_r2("kraken", cut)[0] for cut in CUTS] + [REAL_R2[0]]
    assert max(points) - min(points) < 0.005, f"forward R2 moves across cuts: {points}"


# ---- the price panel --------------------------------------------------------------
# Both venues' mids at every one-second bar the real run's VECM was fitted on, exported
# from the feature cache by `scripts/export_prices.py`. A price file could be any prices,
# so it is held to the run twice: by the provenance it carries, and by refitting the
# information shares from it and getting the published estimates back.
PRICES = DATA / "prices.parquet"


def test_the_prices_are_the_real_runs_rows() -> None:
    carried = json.loads(pq.read_schema(PRICES).metadata[b"crosstick"])
    real = record("real")
    assert carried["source_record"] == "record_real.json"
    for field in ("git_sha", "builder_fingerprint", "source_fingerprints"):
        assert carried[field] == real["provenance"][field], field
    assert carried["target"] == real["spec"]["target"]
    assert carried["max_age_ms"] == real["spec"]["max_age_ms"]
    assert pl.read_parquet(PRICES).height == real["n_rows"]


def test_the_prices_reproduce_the_published_information_shares() -> None:
    """Every date at every stride, refitted with crosstick's own estimator."""
    real = record("real")
    venues = tuple(real["spec"]["infoshare_venues"])
    prices = pl.read_parquet(PRICES)
    for block in real["infoshare"]:
        for want in block["estimates"]:
            day = prices.filter(pl.col("date") == want["date"]).sort("ts_ns")
            panel = np.log(day.select([f"{v}_mid" for v in venues]).to_numpy())
            got = information_shares(panel[:: block["stride"]], venues, block["lags"])
            where = f"{want['date']} at {block['stride']}s"
            assert got.n == want["n"], where
            assert got.alpha == pytest.approx(want["alpha"], rel=1e-9), where
            assert got.bounds.ravel() == pytest.approx(np.ravel(want["bounds"]), abs=1e-9), where


# ---- the other clocks -------------------------------------------------------------
# The same bars timed otherwise, read off crosstick's lake by `scripts/extract_clocks.py`.
# The export already held the rebuilt arrival clock to `prices.parquet` bit for bit, which
# needs the lake. What the files alone can show is that every re-timed row sits on a
# published bar and changes it, so a series is the published panel with these rows swapped
# in, and that the clock report covers every date of the run.
RETIMED = DATA / "prices_retimed.parquet"
CLOCKS = DATA / "clocks.json"


def test_the_retimed_prices_sit_on_the_published_bars() -> None:
    carried = json.loads(pq.read_schema(RETIMED).metadata[b"crosstick"])
    real = record("real")
    assert carried["source_record"] == "record_real.json"
    assert carried["git_sha"] == real["provenance"]["git_sha"]
    assert carried["dates"] == real["spec"]["dates"]

    retimed = pl.read_parquet(RETIMED)
    assert set(retimed["series"].unique()) == set(carried["series"])
    assert not retimed.select("series", "ts_ns").is_duplicated().any()
    joined = retimed.join(pl.read_parquet(PRICES), on=["date", "ts_ns"], how="left")
    assert joined["coinbase_mid"].null_count() == 0, "a re-timed row is not a published bar"
    venue_mid = pl.when(pl.col("series").str.starts_with("coinbase"))
    published = venue_mid.then(pl.col("coinbase_mid")).otherwise(pl.col("kraken_mid"))
    assert joined.filter(pl.col("mid") == published).is_empty(), "a re-timed row changes nothing"


def test_the_clock_report_covers_every_date() -> None:
    report = json.loads(CLOCKS.read_text(encoding="utf-8"))
    dates = record("real")["spec"]["dates"]
    assert report["source_record"] == "record_real.json"
    for venue in ("coinbase", "kraken"):
        assert sorted(report["venues"][venue]) == dates, venue
        for day in report["venues"][venue].values():
            assert day["stamp_test"]["matched"] > 0 and day["book_updates"] > 0
