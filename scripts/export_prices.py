"""Export the one-second price panel behind the published information shares.

`data/` carries the forecast's out-of-sample rows and the run records. That recomputes
every forecasting number, but it cannot show the prices themselves or refit the
error-correction model the Hasbrouck bounds come from. This adds the one series both
need: each venue's mid at every one-second bar the real run's VECM was fitted on.

It reads crosstick's local feature cache, not the lake, and it finds the cache entries by
the names the real run's own provenance gives them (one source fingerprint per date, plus
the builder fingerprint), so the rows are the ones that run read rather than a later
rebuild of the same dates. It applies the run's own `clean`, then the selection
`infoshare.price_panel` makes, keeping the timestamps that function discards. It refuses
to write unless refitting `infoshare.information_shares` on the result reproduces every
published estimate, at every stride.

Run it from this repo's environment, which pins the crosstick commit that wrote the run:

    uv run python scripts/export_prices.py ../crosstick/python/.feature_cache

CI has no cache and never runs this. `tests/test_published_runs.py` checks what it wrote.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from research import infoshare
from research.main import builder_fingerprint, cache_name
from research.model import clean
from research.schema import FeatureSchema

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "prices.parquet"


def main(cache: Path) -> None:
    run = json.loads((DATA / "record_real.json").read_text(encoding="utf-8"))
    spec, provenance = run["spec"], run["provenance"]
    venues = tuple(spec["infoshare_venues"])
    mids, ages = [f"{v}_mid" for v in venues], [f"{v}_age_ms" for v in venues]

    # `cache_name` stamps the installed builder's fingerprint into the filename, so this
    # has to be the builder that wrote the run or every name below points at a rebuild.
    assert builder_fingerprint() == provenance["builder_fingerprint"]
    files = [
        cache / cache_name(date, spec["symbol"], tuple(spec["extra_symbols"]), fingerprint)
        for date, fingerprint in provenance["source_fingerprints"].items()
    ]
    if missing := [f.name for f in files if not f.exists()]:
        sys.exit(f"not in {cache}: {missing}")

    # Only the columns `clean` and `price_panel` read. The age columns are taken over the
    # union of every date's schema, as the run's own concat saw them.
    schemas = [pl.read_parquet_schema(f) for f in files]
    book, tape = FeatureSchema.from_columns(c for s in schemas for c in s).age_columns()
    wanted = ["ts_ns", "date", spec["target"], *mids, *book, *tape]
    df = pl.concat(
        [pl.read_parquet(f, columns=[c for c in wanted if c in s]) for f, s in zip(files, schemas)],
        how="diagonal",
    ).sort("ts_ns")
    assert df.height == run["n_rows_before"], f"{df.height} rows, record says {run['n_rows_before']}"
    df = clean(df, spec["target"], spec["max_age_ms"])
    assert df.height == run["n_rows"], f"{df.height} clean rows, record says {run['n_rows']}"

    panel = (
        df.select(["ts_ns", "date", *mids, *ages])
        .drop_nulls()
        .filter(*(pl.col(a) <= spec["max_age_ms"] for a in ages))
        .sort("ts_ns")
        .select(["date", "ts_ns", *mids])
    )

    published = {(b["stride"], e["date"]): e for b in run["infoshare"] for e in b["estimates"]}
    reproduced = set()
    for (date,), part in df.group_by(["date"], maintain_order=True):
        prices = np.log(panel.filter(pl.col("date") == date).select(mids).to_numpy())
        assert np.array_equal(prices, infoshare.price_panel(part, venues, 1, spec["max_age_ms"]))
        for stride in spec["infoshare_strides"]:
            if (stride, date) not in published:
                continue
            got, want = infoshare.information_shares(prices[::stride], venues), published[stride, date]
            assert got.n == want["n"], (stride, date)
            for field in ("alpha", "bounds", "component_shares", "residual_corr"):
                np.testing.assert_allclose(getattr(got, field), want[field], rtol=1e-9, atol=1e-12)
            reproduced.add((stride, date))
    assert reproduced == set(published), f"not reproduced: {sorted(set(published) - reproduced)}"

    # The provenance travels inside the file, so the test can hold it to the record.
    meta = {
        "source_record": "record_real.json",
        "git_sha": provenance["git_sha"],
        "builder_fingerprint": provenance["builder_fingerprint"],
        "source_fingerprints": provenance["source_fingerprints"],
        "target": spec["target"],
        "max_age_ms": spec["max_age_ms"],
        "venues": list(venues),
    }
    table = panel.to_arrow()
    table = table.replace_schema_metadata({b"crosstick": json.dumps(meta).encode()})
    # Lossless either way. One-second timestamps delta-encode to almost nothing, and
    # splitting the floats' bytes lets zstd see that consecutive mids share most of
    # theirs: 3.8 MB against 13.5 MB for the defaults.
    pq.write_table(
        table,
        OUT,
        compression="zstd",
        compression_level=19,
        use_dictionary=False,
        column_encoding={
            "date": "PLAIN",
            "ts_ns": "DELTA_BINARY_PACKED",
            **{m: "BYTE_STREAM_SPLIT" for m in mids},
        },
    )
    print(
        f"wrote {OUT.relative_to(DATA.parent).as_posix()}: {panel.height:,} bars over "
        f"{panel['date'].n_unique()} dates, {OUT.stat().st_size / 1e6:.1f} MB; "
        f"reproduces all {len(published)} published estimates"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
