"""The published pair, checked against itself.

Two distinct claims are asserted here, and they fail for different reasons.

The first is that the files are what they say they are: written from a clean tree, from
the same experiment, by the same feature builder. Without that, the placebo nulls out a
different question than the one the real run asked, and once the two files sit side by
side in `data/` nothing downstream can tell. `research.export` already defines what makes
a placebo comparable, so that definition is imported rather than restated.

The second is the leakage gate. The pipeline that produced R2 0.1035 on the real target
produces R2 -0.0001 on a target displaced by an hour, and the interval straddles zero.
Recomputed here from the committed rows rather than read from the record, because reading
the record back would only prove the record is a file.
"""

import json
from pathlib import Path

import polars as pl
import pytest
from research.export import EXPERIMENT
from research.runs import read_oos
from research.validation import confidence_intervals

DATA = Path(__file__).resolve().parents[1] / "data"

# The recorded baseline the notebook and the README quote. A rerun of the bootstrap is
# seeded, so this is an equality check with room for float drift across platforms, not a
# tolerance chosen to make a moving number pass.
PLACEBO_R2 = (-0.00010, -0.00028, 0.00011)
REAL_R2 = (0.10349, 0.09075, 0.11202)
REAL_HIT = (0.75719, 0.73511, 0.78052)
TOLERANCE = 5e-5


def record(tag: str) -> dict:
    return json.loads((DATA / f"record_{tag}.json").read_text(encoding="utf-8"))


def oos(tag: str, model: str = "gbt"):
    return read_oos(pl.read_parquet(DATA / f"oos_{tag}.parquet"))[model]


@pytest.mark.parametrize("tag", ["real", "placebo"])
def test_record_has_provenance(tag: str) -> None:
    provenance = record(tag)["provenance"]
    assert not provenance["git_dirty"], "recorded from a dirty tree, so its sha is not its code"
    assert provenance["git_sha"]
    assert len(provenance["source_fingerprints"]) == len(record(tag)["spec"]["dates"])


def test_placebo_is_the_same_experiment() -> None:
    real, placebo = record("real")["spec"], record("placebo")["spec"]
    assert not real["placebo"] and placebo["placebo"]
    assert {field: real[field] for field in EXPERIMENT} == {
        field: placebo[field] for field in EXPERIMENT
    }


def test_pair_shares_a_feature_builder() -> None:
    real, placebo = record("real")["provenance"], record("placebo")["provenance"]
    assert real["builder_fingerprint"] == placebo["builder_fingerprint"]
    assert real["feature_schema_digest"] == placebo["feature_schema_digest"]
    assert real["source_fingerprints"] == placebo["source_fingerprints"]


def test_placebo_finds_no_signal() -> None:
    """The gate: displace the target and the whole pipeline has to go quiet."""
    point, low, high = confidence_intervals(oos("placebo"))["r2_vs_zero"]
    assert low < 0 < high, f"placebo R2 interval misses zero: [{low:.5f}, {high:.5f}]"
    assert (point, low, high) == pytest.approx(PLACEBO_R2, abs=TOLERANCE)


def test_real_headline_recomputes() -> None:
    pooled = confidence_intervals(oos("real"))
    assert pooled["r2_vs_zero"] == pytest.approx(REAL_R2, abs=TOLERANCE)
    assert pooled["hit_rate"] == pytest.approx(REAL_HIT, abs=TOLERANCE)


@pytest.mark.parametrize("tag", ["real", "placebo"])
def test_rows_match_the_record(tag: str) -> None:
    """The record counts rows it no longer carries, so the two can disagree."""
    assert len(oos(tag).y) == record(tag)["n_oos"]
