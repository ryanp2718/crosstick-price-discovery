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

import polars as pl
import pytest
from research.export import EXPERIMENT
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
