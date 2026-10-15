"""Plan the millisecond read of every jump, and publish what `extract_jumps.py` found.

The essay's jump study finds every second in which one venue's mid moved 2 bp or more, on
the published one-second prices. `extract_jumps.py` reads crosstick's lake around each of
those jumps to the millisecond. This script runs either side of it:

    uv run python scripts/export_jumps.py plan <plan.json>
    uv run python scripts/export_jumps.py publish <dir written by extract_jumps.py>

`plan` lists the jumps, found by the essay's own rule, with the published mids either side
of each so the extraction can hold its books to them, and draws the three Coinbase jumps
the essay draws for its figure. `publish` checks that what came back covers exactly those
jumps, then writes three files to `data/`:

- `jump_paths.parquet`: for each day, venue that jumped and clock, both venues' mids summed
  over the day's jumps on a 10 ms grid around the crossing, with the count, so any pooling
  of days is a sum. Its metadata holds the parameters and every lake file read.
- `jump_crossings.parquet`: one row per jump, with where in its second the crossing fell
  on each clock, or null where the exchange's clock has none inside it.
- `jump_events.parquet`: both books and trades around the three drawn jumps.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data"
VENUES = ("coinbase", "kraken")
# The essay's jump study.
JUMP_BP, BEFORE, AFTER = 2.0, 10, 15
EVENTS, EVENT_SEED = 3, 1


def jumps() -> pl.DataFrame:
    """Every jump the essay's rule finds, for each venue, with both venues' published mids
    at the bar before it and at its own bar."""
    prices = pl.read_parquet(DATA / "prices.parquet").sort("ts_ns")
    ts, day = prices["ts_ns"].to_numpy(), prices["date"].to_numpy()
    mids = prices.select([f"{v}_mid" for v in VENUES]).to_numpy()
    logp = np.log(mids) * 1e4
    follows = np.r_[False, (np.diff(ts) == 1_000_000_000) & (day[1:] == day[:-1])]
    stretch = np.cumsum(~follows)
    out = []
    for k, leader in enumerate(VENUES):
        move = np.r_[np.nan, np.diff(logp[:, k])]
        at = np.flatnonzero(follows & (np.abs(move) >= JUMP_BP))
        at = at[(at >= BEFORE) & (at + AFTER < len(logp))]
        at = at[stretch[at - BEFORE] == stretch[at + AFTER]]
        out.append(pl.DataFrame({
            "date": day[at], "ts_ns": ts[at], "leader": leader,
            "sign": np.sign(move[at]).astype(np.int8),
            **{f"{v}_mids": np.stack([mids[at - 1, j], mids[at, j]], axis=1).tolist()
               for j, v in enumerate(VENUES)},
        }))
    return pl.concat(out)


def drawn(found: pl.DataFrame) -> list[list]:
    """The essay's three Coinbase jumps, drawn at random."""
    coinbase = found.filter(pl.col("leader") == "coinbase").sort("ts_ns")
    pick = np.sort(np.random.default_rng(EVENT_SEED).choice(coinbase.height, EVENTS, replace=False))
    return [[d, int(t)] for d, t in coinbase[pick].select("date", "ts_ns").iter_rows()]


def plan(path: Path) -> None:
    found = jumps()
    path.write_text(json.dumps({"jumps": found.to_dicts(), "events": drawn(found)}))
    print(f"planned {found.height:,} jumps and {EVENTS} drawn events to {path}")


def publish(extracted: Path) -> None:
    found = jumps()
    report = json.loads((extracted / "report.json").read_text(encoding="utf-8"))
    paths = pl.read_parquet(extracted / "paths.parquet")
    rows = pl.read_parquet(extracted / "jumps.parquet")
    events = pl.read_parquet(extracted / "events.parquet")

    key = ["date", "ts_ns", "leader", "sign"]
    assert rows.select(key).sort(key).equals(
        found.select(key).with_columns(pl.col("sign").cast(rows["sign"].dtype)).sort(key)
    ), "the extraction does not cover exactly the essay's jumps"
    assert rows["cross_arrival_ms"].null_count() == 0, "a jump has no crossing on arrival"
    totals = paths.unique(["date", "leader", "clock"]).group_by("leader", "clock").agg(pl.col("n").sum())
    for leader, clock, n in totals.iter_rows():
        counted = rows.filter((pl.col("leader") == leader) & pl.col(f"cross_{clock}_ms").is_not_null()).height
        assert n == counted, (leader, clock, n, counted)
    picked = drawn(found)
    assert sorted(map(list, events.select("date", "jump_ts_ns").unique().iter_rows())) == picked
    for (date, t), g in events.group_by("date", "jump_ts_ns"):
        assert set(g.filter(pl.col("kind") == "book")["venue"]) == set(VENUES), (date, t)
        assert g["cross_ns"].n_unique() == 1, f"{date} {t} was cut around more than one crossing"

    run = json.loads((DATA / "record_real.json").read_text(encoding="utf-8"))
    meta = {"source": "crosstick's lake, silver book_quality and trades", "dates": run["spec"]["dates"],
            "jump_bp": JUMP_BP, "events": picked, "event_seed": EVENT_SEED, **report}
    for name, table in (("jump_paths", paths), ("jump_crossings", rows), ("jump_events", events)):
        arrow = table.to_arrow()
        if name == "jump_paths":
            arrow = arrow.replace_schema_metadata({b"crosstick": json.dumps(meta).encode()})
        pq.write_table(arrow, DATA / f"{name}.parquet", compression="zstd", compression_level=19)
        print(f"wrote data/{name}.parquet: {table.height:,} rows, "
              f"{(DATA / f'{name}.parquet').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    {"plan": plan, "publish": publish}[sys.argv[1]](Path(sys.argv[2]))
