"""Publish the prices re-timed off the arrival clock, and what the clocks themselves showed.

`prices.parquet` stamps each venue's mid when it reached the capture machine, so every
delay between an exchange and that machine is in it, and Kraken's updates take about 40ms
longer to arrive than Coinbase's. The appendix refits the information share and the jump
study on two alternatives: each update at the exchange's own stamp, and Kraken sampled a
fixed amount later on the arrival clock. This writes both, from what `extract_clocks.py`
read off crosstick's lake:

    uv run python scripts/export_clocks.py <dir written by extract_clocks.py>

`prices_retimed.parquet` holds, for each re-timed series, only the bars where it differs
from the published panel: a re-timed series is the published one with these rows swapped
in. The bars are the published panel's, so a refit changes the clock and nothing else.
`clocks.json` holds each venue's arrival delays and the test of what its book stamp marks.

It refuses to write unless the arrival clock, rebuilt from `book_quality` and converted
the way crosstick's feature build converts it, equals the published panel bit for bit. That
is what makes the re-timed series comparable: the same books, the same bars, the same
arithmetic, and only the clock changed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "prices_retimed.parquet"
VENUES = ("coinbase", "kraken")


def mid(prefix: str) -> pl.Expr:
    """The feature build's mid: decimals cast to f64 by polars, then averaged."""
    bid, ask = pl.col(f"{prefix}_bid").cast(pl.Float64), pl.col(f"{prefix}_ask").cast(pl.Float64)
    return ((bid + ask) / 2).alias(prefix)


def main(extracted: Path) -> None:
    run = json.loads((DATA / "record_real.json").read_text(encoding="utf-8"))
    report = json.loads((extracted / "clocks.json").read_text(encoding="utf-8"))
    prices = pl.read_parquet(DATA / "prices.parquet")
    bars = pl.read_parquet(extracted / "bars.parquet")
    assert sorted(bars["date"].unique()) == run["spec"]["dates"]

    series = {
        **{f"{v}_exchange": v for v in VENUES},
        **{f"kraken_plus_{s}ms": "kraken" for s in report["shifts_ms"]},
    }
    panel = prices.join(
        bars.select(
            "date", "ts_ns",
            *(mid(f"{v}_arrival") for v in VENUES),
            *(pl.col(f"{v}_arrival_at") for v in VENUES),
            *(mid(name) for name in series),
        ),
        on=["date", "ts_ns"],
        how="left",
    ).sort("ts_ns")
    assert panel.height == prices.height

    max_age_ns = run["spec"]["max_age_ms"] * 1_000_000
    for v in VENUES:
        rebuilt, published = panel[f"{v}_arrival"].to_numpy(), panel[f"{v}_mid"].to_numpy()
        assert np.array_equal(rebuilt, published), f"{v}: the arrival clock is not the published panel"
        assert (panel["ts_ns"] - panel[f"{v}_arrival_at"]).max() <= max_age_ns
    for name in series:
        assert panel[name].null_count() == 0, f"{name} has a bar with no update before it"

    retimed = pl.concat(
        [
            panel.filter(pl.col(name) != pl.col(f"{venue}_mid")).select(
                "date", "ts_ns", pl.lit(name).alias("series"), pl.col(name).alias("mid")
            )
            for name, venue in series.items()
        ]
    )

    meta = {
        "source_record": "record_real.json",
        "git_sha": run["provenance"]["git_sha"],
        "dates": run["spec"]["dates"],
        "series": list(series),
        "shifts_ms": report["shifts_ms"],
    }
    table = retimed.to_arrow().replace_schema_metadata({b"crosstick": json.dumps(meta).encode()})
    pq.write_table(
        table,
        OUT,
        compression="zstd",
        compression_level=19,
        use_dictionary=["date", "series"],
        column_encoding={"ts_ns": "DELTA_BINARY_PACKED", "mid": "BYTE_STREAM_SPLIT"},
    )
    report["source_record"] = "record_real.json"
    (DATA / "clocks.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    changed = retimed.group_by("series", maintain_order=True).len()
    for name, n in changed.iter_rows():
        print(f"  {name:22s} differs from the published panel on {n / prices.height:6.2%} of bars")
    print(f"wrote {OUT.relative_to(DATA.parent).as_posix()}: {retimed.height:,} rows, "
          f"{OUT.stat().st_size / 1e6:.1f} MB; the arrival clock matches the published panel "
          f"on all {prices.height:,} bars")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
