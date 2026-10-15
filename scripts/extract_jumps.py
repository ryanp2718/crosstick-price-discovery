"""Read both venues' books off crosstick's lake around every jump, to the millisecond, and
their trades around the jumps the essay draws, for `export_jumps.py`.

The essay's jump study works on one-second bars, so inside the second a jump falls in it
cannot say which venue moved first. The books can: each update carries the time it reached
the capture machine, and the exchange's own stamp. This lines every jump up on the instant
the jumping venue's mid first stood `CROSS_BP` from where it stood when the second opened,
and records where both venues' mids stood around that instant, on both clocks.

It runs where the lake is reachable and needs only crosstick's `common` package, pyarrow
and numpy, as `extract_clocks.py` does:

    uv run python scripts/export_jumps.py plan /tmp/plan.json
    docker cp /tmp/plan.json crosstick-materializer-1:/tmp/plan.json
    docker exec -i crosstick-materializer-1 python - /tmp/jumps /tmp/plan.json \
        < scripts/extract_jumps.py
    docker cp crosstick-materializer-1:/tmp/jumps <dir>
    uv run python scripts/export_jumps.py publish <dir>

The plan lists every jump with the published mids either side of it, and the jumps the
essay draws for its figure. Books are read from silver `book_quality`, as in
`extract_clocks.py`, which keeps both stamps. A jump is written only if the arrival clock,
rebuilt here, returns the published mids at the jump's own bars, so the lines below are
read off the same books the bars were.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from common.lake import filesystem_from_env

NS, MS = 1_000_000_000, 1_000_000
VENUES = {"coinbase": "BTC-USD", "kraken": "BTC-USD"}
CLOCKS = ("arrival", "exchange")
# The jump is placed where the jumping venue's mid first stood this far, in its direction,
# from its level when the second opened: half the jump study's smallest jump.
CROSS_BP = 1.0
# Each path is read on this grid around the crossing, from where each venue stood a
# second before it.
GRID = np.arange(-2000, 5001, 10) * MS
REFERENCE = -NS
# The drawn events are kept whole over this stretch around the crossing.
EVENT = (-2 * NS, 5 * NS)
# crosstick stores a Coinbase trade's side as the side of the resting order it filled, and
# every other venue's as the side that took liquidity (see `extract_clocks.py`).
MAKER_SIDE = {"coinbase"}


def _sources(fs, d: ds.Dataset) -> list[dict]:
    return [{"path": i.path, "size": i.size, "mtime_ns": i.mtime_ns}
            for i in fs.get_file_info(d.files)]


def _book(fs, venue: str, date: str, sources: list) -> dict:
    """Each clock's update times, with the bid, ask and log mid in bp after each update."""
    d = ds.dataset(f"silver/book_quality/exchange={venue}/symbol={VENUES[venue]}/date={date}",
                   filesystem=fs, format="parquet")
    sources += _sources(fs, d)
    t = d.to_table(columns=["kind", "exchange_ts_ns", "local_recv_ts_ns", "local_ts_ns",
                            "invariant_kind", "best_bid", "best_ask"])
    t = t.filter(pc.and_(pc.is_null(t["invariant_kind"]),
                         pc.and_(pc.is_valid(t["best_bid"]), pc.is_valid(t["best_ask"]))))
    arrival = pc.coalesce(t["local_recv_ts_ns"], t["local_ts_ns"]).to_numpy()
    stamp = t["exchange_ts_ns"].to_numpy()
    bid = pc.cast(t["best_bid"], pa.float64(), safe=False).to_numpy()
    ask = pc.cast(t["best_ask"], pa.float64(), safe=False).to_numpy()
    out = {}
    # The arrival clock, ordered the way the feature build sorts `quotes`.
    order = np.argsort(arrival, kind="stable")
    out["arrival"] = order, arrival[order]
    # The exchange's clock, as in `extract_clocks.py`: snapshots carry no stamp and drop out.
    rows = np.flatnonzero(np.asarray(pc.equal(t["kind"], "delta")) & (stamp > 0))
    rows = rows[np.argsort(stamp[rows], kind="stable")]
    out["exchange"] = rows, stamp[rows]
    return {clock: (times, bid[rows], ask[rows], np.log((bid[rows] + ask[rows]) / 2) * 1e4)
            for clock, (rows, times) in out.items()}


def _trades(fs, venue: str, date: str, sources: list) -> dict:
    """Each trade on the arrival clock, with its price, size and taker side, +1 a buy."""
    d = ds.dataset(f"silver/trades/exchange={venue}/symbol={VENUES[venue]}/date={date}",
                   filesystem=fs, format="parquet")
    sources += _sources(fs, d)
    t = d.to_table(columns=["ts_ns", "price", "size", "side"])
    order = np.argsort(t["ts_ns"].to_numpy(), kind="stable")
    side = np.where(np.asarray(t["side"].to_pylist()) == "bid", 1, -1)
    if venue in MAKER_SIDE:
        side = -side
    return {"ts": t["ts_ns"].to_numpy()[order],
            "price": pc.cast(t["price"], pa.float64(), safe=False).to_numpy()[order],
            "size": pc.cast(t["size"], pa.float64(), safe=False).to_numpy()[order],
            "taker": side[order]}


def _asof(times: np.ndarray, values: np.ndarray, at) -> np.ndarray:
    i = np.searchsorted(times, at, side="right") - 1
    return np.where(i >= 0, values[np.maximum(i, 0)], np.nan)


def _crossing(times, logmid, t0: int, t1: int, sign: int) -> int | None:
    """The first update in the second (t0, t1] at which the mid stands CROSS_BP from its
    level at t0, in the jump's direction."""
    start = _asof(times, logmid, t0)
    i0, i1 = np.searchsorted(times, t0, "right"), np.searchsorted(times, t1, "right")
    hit = np.flatnonzero(sign * (logmid[i0:i1] - start) >= CROSS_BP)
    return int(times[i0 + hit[0]]) if len(hit) else None


def main(out: Path, plan_file: str) -> None:
    fs = filesystem_from_env()
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(plan_file).read_text())
    events = {tuple(e) for e in plan["events"]}
    sums, rows, slices, sources = {}, [], [], {}

    for date in sorted({j["date"] for j in plan["jumps"]}):
        src = sources.setdefault(date, [])
        books = {v: _book(fs, v, date, src) for v in VENUES}
        # Trades are drawn only in the figure's events.
        drawn = any(d == date for d, _ in events)
        trades = {v: _trades(fs, v, date, src) for v in VENUES} if drawn else None
        for j in (j for j in plan["jumps"] if j["date"] == date):
            leader, t1, sign = j["leader"], j["ts_ns"], j["sign"]
            other = next(v for v in VENUES if v != leader)
            t0 = t1 - NS
            # The arrival clock rebuilt here has to return the published mids at the
            # jump's own bars, or these are not the books the bars were read from.
            for venue in VENUES:
                times, _, _, logmid = books[venue]["arrival"]
                got = _asof(times, logmid, np.array([t0, t1]))
                want = np.log(np.array(j[f"{venue}_mids"])) * 1e4
                assert np.allclose(got, want, rtol=0, atol=1e-6), (date, t1, venue, got, want)
            row = {"date": date, "ts_ns": t1, "leader": leader, "sign": sign}
            for clock in CLOCKS:
                lt, _, _, lm = books[leader][clock]
                ot, _, _, om = books[other][clock]
                c = _crossing(lt, lm, t0, t1, sign)
                row[f"cross_{clock}_ms"] = None if c is None else (c - t0) / MS
                if c is None:
                    continue
                ref_l, ref_o = _asof(lt, lm, c + REFERENCE), _asof(ot, om, c + REFERENCE)
                path_l = sign * (_asof(lt, lm, c + GRID) - ref_l)
                path_o = sign * (_asof(ot, om, c + GRID) - ref_o)
                assert not (np.isnan(path_l).any() or np.isnan(path_o).any()), (date, t1, clock)
                s = sums.setdefault((date, leader, clock), [0, np.zeros(len(GRID)), np.zeros(len(GRID))])
                s[0] += 1
                s[1] += path_l
                s[2] += path_o
                # A second can hold a jump at both venues; the events are Coinbase's.
                if clock == "arrival" and leader == "coinbase" and (date, t1) in events:
                    slices += _event(books, trades, date, t1, c, sign)
            rows.append(row)
        print(f"{date}: {sum(1 for r in rows if r['date'] == date):,} jumps", file=sys.stderr, flush=True)

    found = {(d, ts) for d, ts in ((r["date"], r["ts_ns"]) for r in rows)}
    assert all(e in found for e in events), "a drawn event is not among the jumps"
    grid = (GRID // MS).astype(np.int32)
    pq.write_table(pa.table({
        "date": [k[0] for k in sums for _ in grid],
        "leader": [k[1] for k in sums for _ in grid],
        "clock": [k[2] for k in sums for _ in grid],
        "ms": np.tile(grid, len(sums)),
        "n": np.repeat([v[0] for v in sums.values()], len(grid)).astype(np.int32),
        "leader_sum": np.concatenate([v[1] for v in sums.values()]),
        "other_sum": np.concatenate([v[2] for v in sums.values()]),
    }), out / "paths.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(rows), out / "jumps.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(slices), out / "events.parquet", compression="zstd")
    (out / "report.json").write_text(json.dumps({
        "cross_bp": CROSS_BP, "reference_ms": REFERENCE // MS, "event_ms": [EVENT[0] // MS, EVENT[1] // MS], "sources": sources}, indent=1))


def _event(books, trades, date, t1, c, sign) -> list[dict]:
    """Both venues' books and trades around one drawn jump, on the arrival clock, with the
    update standing when the stretch opens."""
    lo, hi = c + EVENT[0], c + EVENT[1]
    out = []
    for venue in VENUES:
        times, bid, ask, _ = books[venue]["arrival"]
        i0, i1 = max(np.searchsorted(times, lo, "right") - 1, 0), np.searchsorted(times, hi, "right")
        out += [{"date": date, "jump_ts_ns": t1, "sign": sign, "cross_ns": c, "venue": venue,
                 "kind": "book", "ts_ns": int(times[i]), "bid": float(bid[i]), "ask": float(ask[i]),
                 "price": None, "size": None, "taker": None} for i in range(i0, i1)]
        tr = trades[venue]
        i0, i1 = np.searchsorted(tr["ts"], lo, "left"), np.searchsorted(tr["ts"], hi, "right")
        out += [{"date": date, "jump_ts_ns": t1, "sign": sign, "cross_ns": c, "venue": venue,
                 "kind": "trade", "ts_ns": int(tr["ts"][i]), "bid": None, "ask": None,
                 "price": float(tr["price"][i]), "size": float(tr["size"][i]),
                 "taker": int(tr["taker"][i])} for i in range(i0, i1)]
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
