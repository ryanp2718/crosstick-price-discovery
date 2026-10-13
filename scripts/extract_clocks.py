"""Read both venues' books off crosstick's lake on two clocks, for `export_clocks.py`.

Every price in `data/` is stamped when it reached the capture machine, which is the clock
crosstick's feature build runs on. Kraken's updates reach that machine about 40ms after
Coinbase's, so on that clock Coinbase starts every second slightly ahead. This reads the
same books with each update placed at the exchange's own stamp instead, and with Kraken
sampled a fixed amount later, so the published estimates can be refitted on each.

It runs where the lake is reachable and needs only crosstick's `common` package, pyarrow
and numpy. On the local stack the materializer container has all three and the lake's
credentials, so nothing has to be copied out of crosstick's `.env`:

    docker exec -i crosstick-materializer-1 python - /tmp/clocks "$DATES" \
        < scripts/extract_clocks.py
    docker cp crosstick-materializer-1:/tmp/clocks <dir>
    uv run python scripts/export_clocks.py <dir>

`$DATES` is the real run's `spec.dates` as JSON. It reads silver `book_quality`, not
`quotes`: both come from one book fold (crosstick's `silver/dq.py`), and a quote is a
`book_quality` row with no invariant and both sides present, stamped on arrival. Only
`book_quality` also keeps the exchange's stamp. Prices leave as the lake's exact decimals,
so `export_clocks.py` can convert them the way the feature build did and hold the arrival
clock to the published panel bit for bit.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from common.lake import filesystem_from_env

VENUES = {"coinbase": "BTC-USD", "kraken": "BTC-USD"}
# Kraken sampled this long after each bar closes on the arrival clock: as if each of its
# updates had happened that much earlier than it reached us.
SHIFTS_MS = (20, 40, 60, 80, 100, 150, 200)
# crosstick stores a Coinbase trade's side as the side of the resting order it filled, and
# every other venue's as the side that took liquidity, so Coinbase's is flipped here. The
# quote test holds the correction to the book on every day: a taker buy pays the ask, so most
# of the trades taken as buys should print above the mid, and most sells below it.
MAKER_SIDE = {"coinbase"}
NS = 1_000_000_000
MS = 1_000_000


def _quantiles(x: np.ndarray) -> dict:
    return {f"p{q}": float(np.percentile(x, q)) for q in (10, 50, 90)}


def _asof(stamps: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the last update stamped at or before each `at`, or -1 before the first."""
    return np.searchsorted(stamps, at, side="right") - 1


def _take(column: pa.ChunkedArray, index: np.ndarray) -> pa.Array:
    return pc.take(column, pa.array(index, mask=index < 0))


def _taker(venue: str, trades: pa.Table) -> np.ndarray:
    """Each trade's taker side, "bid" for a buy and "ask" for a sell."""
    side = np.asarray(trades["side"].to_pylist())
    if venue in MAKER_SIDE:
        side = np.where(side == "bid", "ask", np.where(side == "ask", "bid", side))
    return side


def _quote_test(arrival, bid, ask, trades, taker) -> dict:
    """How often a trade taken as a buy printed above the mid of the book that had arrived
    before it, and a sell below it."""
    i = _asof(arrival, trades["ts_ns"].to_numpy())
    ok = i >= 0
    mid = (bid[i[ok]] + ask[i[ok]]) / 2
    price = pc.cast(trades["price"], pa.float64(), safe=False).to_numpy()[ok]
    return {
        "buys_above_mid": float((price > mid)[taker[ok] == "bid"].mean()),
        "sells_below_mid": float((price < mid)[taker[ok] == "ask"].mean()),
    }


def _stamp_test(s, bid, ask, trades, taker) -> dict:
    """Where does a book update's stamp sit against the trade that caused it?

    A taker buy that takes out the best ask moves the ask up, and a taker sell that takes
    out the best bid moves the bid down. For each such move we find the last trade on that
    side at the old touch price, stamped within a second before the update. Every venue
    documents its trade stamp as the time of the trade, so if the update's stamp marks
    the same engine event the two are equal, and if it marks when the update was sent it
    is later by however long sending took.
    """
    tt = trades["exchange_ts_ns"].to_numpy()
    tp = pc.cast(trades["price"], pa.float64(), safe=False).to_numpy()
    side = taker
    order = np.argsort(tt, kind="stable")
    tt, tp, side = tt[order], tp[order], side[order]
    gaps = []
    for taker, moved, old in (
        ("bid", ask[1:] > ask[:-1], ask[:-1]),
        ("ask", bid[1:] < bid[:-1], bid[:-1]),
    ):
        mine = side == taker
        st, sp = tt[mine], tp[mine]
        for i in np.flatnonzero(moved) + 1:
            j = np.searchsorted(st, s[i], side="right") - 1
            while j >= 0 and s[i] - st[j] <= NS and sp[j] != old[i - 1]:
                j -= 1
            if j >= 0 and s[i] - st[j] <= NS:
                gaps.append(s[i] - st[j])
    gaps = np.array(gaps, dtype=np.int64)
    return {
        "matched": int(len(gaps)),
        "equal": int((gaps == 0).sum()),
        "gap_ms": _quantiles(gaps / MS),
    }


def main(out: Path, dates: list[str]) -> None:
    fs = filesystem_from_env()
    out.mkdir(parents=True, exist_ok=True)
    frames, report = [], {"shifts_ms": list(SHIFTS_MS), "venues": {}}

    for date in dates:
        day0 = int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()) * NS
        bars = day0 + np.arange(0, 86_401, dtype=np.int64) * NS
        cols = {"date": pa.array([date] * len(bars)), "ts_ns": pa.array(bars)}

        for venue, symbol in VENUES.items():
            entry = report["venues"].setdefault(venue, {})[date] = {"sources": []}
            tables = {}
            for dataset, columns in (
                ("book_quality", ["kind", "exchange_ts_ns", "local_recv_ts_ns", "local_ts_ns",
                                  "invariant_kind", "best_bid", "best_ask"]),
                ("trades", ["exchange_ts_ns", "ts_ns", "price", "side"]),
            ):
                path = f"silver/{dataset}/exchange={venue}/symbol={symbol}/date={date}"
                d = ds.dataset(path, filesystem=fs, format="parquet")
                entry["sources"] += [
                    {"path": i.path, "size": i.size, "mtime_ns": i.mtime_ns}
                    for i in fs.get_file_info(d.files)
                ]
                tables[dataset] = d.to_table(columns=columns)

            book = tables["book_quality"]
            quote = pc.and_(
                pc.is_null(book["invariant_kind"]),
                pc.and_(pc.is_valid(book["best_bid"]), pc.is_valid(book["best_ask"])),
            )
            book = book.filter(quote)
            arrival = pc.coalesce(book["local_recv_ts_ns"], book["local_ts_ns"]).to_numpy()
            stamp = book["exchange_ts_ns"].to_numpy()
            delta = np.asarray(pc.equal(book["kind"], "delta")) & (stamp > 0)

            # The arrival clock, ordered the way the feature build sorts `quotes`.
            by_arrival = np.argsort(arrival, kind="stable")
            on_arrival = arrival[by_arrival]
            i = _asof(on_arrival, bars)
            pick = np.where(i >= 0, by_arrival[np.maximum(i, 0)], -1)
            cols[f"{venue}_arrival_bid"] = _take(book["best_bid"], pick)
            cols[f"{venue}_arrival_ask"] = _take(book["best_ask"], pick)
            cols[f"{venue}_arrival_at"] = pa.array(np.where(i >= 0, on_arrival[np.maximum(i, 0)], -1))

            # The exchange's clock. Snapshots carry no stamp of the exchange's and drop out:
            # the delta right behind each one carries the resynced book anyway. Stamps rise
            # within a connection, but a reconnect can leave two connections' rows out of
            # order (Kraken, 2026-07-01), so this sorts and counts how often it had to.
            rows = np.flatnonzero(delta)
            entry["reordered"] = int((np.diff(stamp[rows]) < 0).sum())
            rows = rows[np.argsort(stamp[rows], kind="stable")]
            i = _asof(stamp[rows], bars)
            pick = np.where(i >= 0, rows[np.maximum(i, 0)], -1)
            cols[f"{venue}_exchange_bid"] = _take(book["best_bid"], pick)
            cols[f"{venue}_exchange_ask"] = _take(book["best_ask"], pick)
            cols[f"{venue}_exchange_at"] = pa.array(np.where(i >= 0, stamp[rows][np.maximum(i, 0)], -1))

            if venue == "kraken":
                for shift in SHIFTS_MS:
                    i = _asof(on_arrival, bars + shift * MS)
                    pick = np.where(i >= 0, by_arrival[np.maximum(i, 0)], -1)
                    cols[f"kraken_plus_{shift}ms_bid"] = _take(book["best_bid"], pick)
                    cols[f"kraken_plus_{shift}ms_ask"] = _take(book["best_ask"], pick)

            # How long each kind of update took to reach us, against its own stamp.
            trades = tables["trades"]
            entry["book_delay_ms"] = _quantiles((arrival[delta] - stamp[delta]) / MS)
            entry["book_updates"] = int(delta.sum())
            entry["trade_delay_ms"] = _quantiles(
                (trades["ts_ns"].to_numpy() - trades["exchange_ts_ns"].to_numpy()) / MS
            )
            entry["trades"] = trades.num_rows
            taker = _taker(venue, trades)
            bid = pc.cast(book["best_bid"], pa.float64(), safe=False).to_numpy()
            ask = pc.cast(book["best_ask"], pa.float64(), safe=False).to_numpy()
            entry["quote_test"] = _quote_test(
                on_arrival, bid[by_arrival], ask[by_arrival], trades, taker
            )
            assert min(entry["quote_test"].values()) > 0.5, (date, venue, entry["quote_test"])
            on_stamp = np.flatnonzero(delta)[np.argsort(stamp[delta], kind="stable")]
            entry["stamp_test"] = _stamp_test(
                stamp[on_stamp], bid[on_stamp], ask[on_stamp], trades, taker
            )
            print(f"{date} {venue:9s} book delay p50 {entry['book_delay_ms']['p50']:5.1f} ms, "
                  f"stamp equal to its trade's in {entry['stamp_test']['equal']:,} of "
                  f"{entry['stamp_test']['matched']:,}", file=sys.stderr, flush=True)
        frames.append(pa.table(cols))

    pq.write_table(pa.concat_tables(frames), out / "bars.parquet", compression="zstd")
    (out / "clocks.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), json.loads(sys.argv[2]))
