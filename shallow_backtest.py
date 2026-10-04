"""
Shallow Pullback Backtest (rules locked in BEFORE running; tested on untouched 2024+ data)

HYPOTHESIS: in an uptrend, a stock that pulls back only 23.6%-38.2% of its prior big move
tends to keep rising over the next 10 days, by more than typical uptrending stocks do.

ENTRY (signal at a close, buy the NEXT open; all must be true):
  1. Uptrend: close > 50-day average, and that average is higher than 10 days earlier
  2. Swing high = highest high of last 20 days; swing low = lowest low in the 40 days BEFORE it;
     the move from low to high is at least 20%
  3. The swing high was at least 3 days ago
  4. Retracement (high - close) / (high - low) is between 0.236 and 0.382
  5. Price has not traded below the swing low since the high
  6. SPY's close is above its own 50-day average
  7. No earnings date and no analyst downgrade in the prior 7 calendar days
  8. At most one trade per ticker per 10 days
  (Mechanical skip: next open gaps below the swing low or above the swing high.)

EXITS (no profit targets):  A = hold 10 days, no stop (PRIMARY test)
                            B = swing-low stop, else hold 10 days
                            C = stop 2 x 14-day average daily range below entry, else hold 10 days
  Costs: 0.2% slippage each side. Gaps through a stop fill at the open.

CONTROLS: SPY over the same dates (paired); other_depth (pullbacks outside 23.6-38.2%);
  uptrend_base (rules 1, 6, 7, 8 only, no pullback required; no swing low, so no variant B).

PASS BAR for the PRIMARY test (variant A), all required:
  - at least 100 trades (fewer = INCONCLUSIVE, not failed)
  - average > 0 and larger than its margin of error
  - beats SPY, other_depth and uptrend_base by more than the margin
  - positive average (at least 30 trades) in BOTH 2024-01..2025-06 and 2025-07..now

RUN-ONCE: this tests the untouched 2024+ period, so it can only be run once.
  python shallow_backtest.py          (first run)
  python shallow_backtest.py again    (re-run; no longer a clean test)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TICKERS = [
    "GME", "AMC", "TSLA", "NVDA", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "COIN",
    "MARA", "RIOT", "HOOD", "AMD", "SMCI", "UPST", "AFRM", "CVNA", "BB", "PLUG",
    "FUBO", "SNAP", "RBLX", "DKNG", "OPEN",
]
HISTORY = "10y"
TEST_START = np.datetime64("2023-12-31")   # trades must enter AFTER this date
MID_DATE = np.datetime64("2025-06-30")
MAX_HOLD = 10
SLIP = 0.002
MIN_MOVE = 0.20
HIGH_WINDOW, LOW_WINDOW, MIN_DAYS_SINCE_HIGH = 20, 40, 3
COOLDOWN = 10
ATR_MULT = 2.0
EVENT_DAYS = 7
ZONE = (0.236, 0.382)
MIN_TRADES = 100
LOCK_FILE = ".shallow_test_done"
VARIANTS = ["A", "B", "C"]
ARR = {}


# ---------------- data loading ----------------
def clean_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def load_prices(sym):
    h = yf.Ticker(sym).history(period=HISTORY, auto_adjust=True)
    if h is None or len(h) < 300:
        return None
    h = h[["Open", "High", "Low", "Close"]].copy()
    h.index = clean_index(h.index)
    return h[~h.index.duplicated()].dropna()


def load_events(sym):
    tk = yf.Ticker(sym)
    try:
        e = tk.get_earnings_dates(limit=80)
        earn = clean_index(e.index).sort_values() if e is not None and len(e) else None
    except Exception:
        earn = None
    if earn is None or len(earn) == 0:
        return None
    start = earn.min()
    down = pd.DatetimeIndex([])
    try:
        u = tk.upgrades_downgrades
        if u is not None and len(u):
            idx = clean_index(u.index)
            act = u["Action"].astype(str).str.lower().values
            down = idx[act == "down"].sort_values()
            start = max(start, idx.min())
    except Exception:
        pass
    return dict(earn=earn.values.astype("datetime64[D]"),
                down=down.values.astype("datetime64[D]"),
                start=np.datetime64(start, "D"))


def has_event(dates, d):
    if len(dates) == 0:
        return False
    lo = np.searchsorted(dates, d - np.timedelta64(EVENT_DAYS, "D"), side="left")
    return np.searchsorted(dates, d, side="right") > lo


# ---------------- candidate trades ----------------
def build_candidates(sym, px, spy_ok, ev):
    o, h, l, c = (px[k].values for k in ["Open", "High", "Low", "Close"])
    ARR[sym] = (o, h, l, c)
    dates = px.index.values.astype("datetime64[D]")
    ok = spy_ok.reindex(px.index).values
    sma50 = pd.Series(c).rolling(50).mean().values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    out, last = [], {"shallow": -99, "other_depth": -99, "uptrend_base": -99}
    for t in range(75, len(c) - MAX_HOLD + 1):
        s, d = t - 1, dates[t]
        if d <= TEST_START or d < ev["start"] or not (ok[s] == 1.0):
            continue
        if not (c[s] > sma50[s] and sma50[s] > sma50[s - 10]):
            continue
        if has_event(ev["earn"], d) or has_event(ev["down"], d):
            continue
        entry = o[t] * (1 + SLIP)
        atr = tr[t - 14:t].mean()
        base = dict(tk=sym, t=t, date=d, entry=entry, atr=atr, lo=None)
        if t - last["uptrend_base"] >= COOLDOWN:
            out.append({**base, "zone": "uptrend_base"})
            last["uptrend_base"] = t
        wh = h[s - HIGH_WINDOW + 1:s + 1]
        hi_i, hi = s - HIGH_WINDOW + 1 + int(np.argmax(wh)), wh.max()
        if s - hi_i < MIN_DAYS_SINCE_HIGH:
            continue
        lo = l[hi_i - LOW_WINDOW:hi_i].min()
        if (hi - lo) / lo < MIN_MOVE or l[hi_i:s + 1].min() <= lo or not (lo < entry < hi):
            continue
        r = (hi - c[s]) / (hi - lo)
        z = "shallow" if ZONE[0] <= r <= ZONE[1] else "other_depth"
        if t - last[z] >= COOLDOWN:
            out.append({**base, "zone": z, "lo": lo})
            last[z] = t
    return out


# ---------------- simulation ----------------
def simulate(cand, sprice):
    o, hh, ll, cc = ARR[cand["tk"]]
    t, entry = cand["t"], cand["entry"]
    for d in range(MAX_HOLD):
        i = t + d
        if sprice is not None:
            if d > 0 and o[i] <= sprice:
                return o[i] * (1 - SLIP) / entry - 1, "stop"
            if ll[i] <= sprice:
                return sprice * (1 - SLIP) / entry - 1, "stop"
    return cc[t + MAX_HOLD - 1] * (1 - SLIP) / entry - 1, "time"


def eval_variant(cands, v):
    if v == "B" and any(c["lo"] is None for c in cands):
        return np.array([]), np.array([])
    res = []
    for c in cands:
        sp = None if v == "A" else (c["lo"] if v == "B" else c["entry"] - ATR_MULT * c["atr"])
        res.append(simulate(c, sp))
    return np.array([x[0] for x in res]), np.array([x[1] for x in res])


def summarize(r, why):
    n = len(r)
    if n == 0:
        return None
    w, lz = r[r > 0], r[r <= 0]
    return dict(
        n=n, win_pct=100 * (r > 0).mean(), avg_pct=100 * r.mean(), median_pct=100 * np.median(r),
        plus_minus=100 * 2 * (r.std() if n > 1 else 0) / np.sqrt(n),
        avg_win=100 * w.mean() if len(w) else 0, avg_loss=100 * lz.mean() if len(lz) else 0,
        reward_risk=(w.mean() / abs(lz.mean())) if len(w) and len(lz) and lz.mean() != 0 else np.nan,
        stopped_pct=100 * (why == "stop").mean(), time_pct=100 * (why == "time").mean())


def cell(s):
    return "n/a" if s is None else f"{s['avg_pct']:+.2f}±{s['plus_minus']:.2f} (n={s['n']})"


def spy_returns(cands, spy):
    """SPY open-to-close return over the same 10-day window as each trade (no costs)."""
    pos = spy.index.get_indexer(pd.DatetimeIndex([pd.Timestamp(c["date"]) for c in cands]))
    o, c = spy["Open"].values, spy["Close"].values
    out = np.full(len(cands), np.nan)
    for k, p in enumerate(pos):
        if p >= 0 and p + MAX_HOLD - 1 < len(c):
            out[k] = c[p + MAX_HOLD - 1] / o[p] - 1
    return out


# ---------------- checklist ----------------
def checklist(by_zone, cache, spy_r):
    rows = []
    g = by_zone["shallow"]
    for v in VARIANTS:
        s = cache[("shallow", v)]
        row = {"variant": v + (" (PRIMARY)" if v == "A" else " (secondary)")}
        if s is None or s["n"] < MIN_TRADES:
            n = 0 if s is None else s["n"]
            rows.append({**row, "n": n, "avg>0&>margin": "-", "beats_SPY": "-", "beats_other": "-",
                         "beats_base": "-", "both_halves": "-", "RESULT": "INCONCLUSIVE (n<100)"})
            continue
        r, _ = eval_variant(g, v)
        ok_pair = ~np.isnan(spy_r)
        diff = r[ok_pair] - spy_r[ok_pair]
        d_margin = 2 * diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else np.inf

        def beats(o):
            return o is not None and s["avg_pct"] - o["avg_pct"] > np.hypot(s["plus_minus"], o["plus_minus"])

        c_zero = s["avg_pct"] > 0 and s["avg_pct"] > s["plus_minus"]
        c_spy = len(diff) > 0 and diff.mean() * 100 > d_margin * 100
        c_oth = beats(cache[("other_depth", v)])
        base = cache[("uptrend_base", v)]
        c_base = beats(base) if v != "B" else None
        ok_halves = True
        for part in ([c for c in g if c["date"] <= MID_DATE], [c for c in g if c["date"] > MID_DATE]):
            rr, _ = eval_variant(part, v)
            ok_halves = ok_halves and len(rr) >= 30 and rr.mean() > 0
        checks = [c_zero, c_spy, c_oth, ok_halves] + ([c_base] if c_base is not None else [])
        f = lambda b: "pass" if b else "FAIL"
        rows.append({**row, "n": s["n"], "avg>0&>margin": f(c_zero), "beats_SPY": f(c_spy),
                     "beats_other": f(c_oth), "beats_base": "n/a" if c_base is None else f(c_base),
                     "both_halves": f(ok_halves), "RESULT": "PASS" if all(checks) else "FAIL"})
    return pd.DataFrame(rows)


# ---------------- main ----------------
def main():
    again = "again" in [a.lower() for a in sys.argv[1:]]
    if os.path.exists(LOCK_FILE) and not again:
        print("This test has ALREADY been run on the untouched 2024+ data.")
        print("Re-running it is no longer a clean test. To run anyway: python shallow_backtest.py again")
        return
    pd.set_option("display.width", 250, "display.max_columns", 30)

    print(f"Downloading {HISTORY} of data (SPY + {len(TICKERS)} tickers)...")
    spy = load_prices("SPY")
    if spy is None:
        print("Could not download SPY data. Check your connection.")
        return
    sma = spy["Close"].rolling(50).mean()
    spy_ok = (spy["Close"] > sma).astype(float).where(sma.notna())

    cands, used, skipped = [], [], []
    for sym in TICKERS:
        try:
            px = load_prices(sym)
            ev = load_events(sym) if px is not None else None
            if px is None or ev is None:
                skipped.append(sym)
                continue
            cands += build_candidates(sym, px, spy_ok, ev)
            used.append(sym)
        except Exception as e:
            skipped.append(f"{sym} ({e})")
    print(f"Used {len(used)} tickers. Skipped (no price or earnings history): {skipped or 'none'}")
    print("\nTEST PERIOD: entries after 2023-12-31 (untouched data).")

    by_zone = {z: [c for c in cands if c["zone"] == z] for z in ["shallow", "other_depth", "uptrend_base"]}
    print("Trades per group: " + ", ".join(f"{z}={len(v)}" for z, v in by_zone.items()))
    if len(by_zone["shallow"]) < 10:
        print("Far too few shallow-pullback trades to say anything.")
        return

    cache = {}
    for z, cs in by_zone.items():
        for v in VARIANTS:
            cache[(z, v)] = summarize(*eval_variant(cs, v))

    spy_r = spy_returns(by_zone["shallow"], spy)
    print(f"SPY average {MAX_HOLD}-day open-to-close return over the same windows as the shallow trades: "
          f"{100 * np.nanmean(spy_r):+.2f}%")

    names = {"A": "A: hold 10 days", "B": "B: swing-low stop", "C": "C: 2xATR stop"}
    rows = [dict(variant=names[v], **cache[("shallow", v)]) for v in VARIANTS if cache[("shallow", v)]]
    tab = pd.DataFrame(rows)
    print("\n=== SHALLOW PULLBACK TRADES (avg_pct = average return per trade after costs) ===")
    print(tab.round(2).to_string(index=False))

    print("\n=== DOES THE SHALLOW PULLBACK MATTER? (avg ± margin of error, n) ===")
    cmp_rows = []
    for v in VARIANTS:
        cmp_rows.append({"variant": names[v], "shallow": cell(cache[("shallow", v)]),
                         "other_depth": cell(cache[("other_depth", v)]),
                         "uptrend_base": cell(cache[("uptrend_base", v)])})
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    print("\n=== PRE-REGISTERED CHECKLIST (variant A decides; B and C are informational) ===")
    v = checklist(by_zone, cache, spy_r)
    print(v.to_string(index=False))
    res = v.iloc[0]["RESULT"]
    if res == "PASS":
        print("\nPRIMARY TEST PASSED. That means 'worth tracking going forward' (paper trade / journal), "
              "NOT 'proven'. This is the third idea tested, so luck is still possible.")
    elif res.startswith("INCONCLUSIVE"):
        print("\nPRIMARY TEST INCONCLUSIVE: too few trades to judge. Do not adjust rules and re-run.")
    else:
        print("\nPRIMARY TEST FAILED. Per our plan, do not adjust the rules and re-run. Move on to Option B.")

    tab.to_csv("shallow_results.csv", index=False)
    with open(LOCK_FILE, "w") as f:
        f.write("done")
    print("""
CAUTIONS
  - Survivorship bias: tickers still trade today and the rules select uptrends, which flatters results.
  - Uptrend stocks drift up on average; that's why the uptrend_base and SPY comparisons matter most.
  - Third idea tested on this universe: a lucky pass is possible. Confirm going forward.
  - Daily data can't show order of events inside a day; stops assumed to hit first.""")


if __name__ == "__main__":
    main()
