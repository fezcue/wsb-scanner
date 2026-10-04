"""
Dip-Buy Backtest (rules locked in BEFORE looking at results)

IDEA: buy a stock at the open after it fell >5% over 5 days, on a day the market opens up,
then sell at a target or stop. We test 4 targets x 3 stops = 12 primary combinations.

ENTRY (at the open of day T, using only info known before it):
  - stock's 5-day return (as of the prior close) is below -5%
  - SPY opens above its prior close (stand-in for positive futures)
  - no earnings date and no analyst downgrade in the prior 7 calendar days (through T)
  - swing high (highest high of last 10 days) happened BEFORE swing low (lowest low of last 5 days)
  - the 50% Fibonacci level is at least 3% above the entry price (enough room to be worth it)
  - at most one trade per ticker per 5 days

TARGETS: above_entry (first CLOSE above entry), plus3 (+3%), fib50, fib618 (golden pocket edges)
STOPS:   low5d (5-day low, min 2% away; 20% below entry if the low was yesterday),
         fixed5 (-5%), atr (1.5 x 14-day average daily range)
EXIT:    first of target/stop, else the close of day 5. If stop and target both fit in one day,
         the stop is assumed to hit first. Gaps through a level fill at the open.
COSTS:   0.2% slippage on each side.

CONTROLS: the same entry/exit rules on days that did NOT have the -5% dip, plus simple
fixed-% targets matched to the typical Fibonacci distance, plus "just hold 5 days".

USAGE:
  python dip_backtest.py            -> DESIGN period only (through 2023-12-31)
  python dip_backtest.py final      -> HELD-OUT period (2024 onward). Run this ONCE, at the end.
  python dip_backtest.py GME AMC    -> custom tickers (can combine with 'final')
"""
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
SPLIT_DATE = np.datetime64("2023-12-31")
DIP_THRESHOLD = -0.05
MAX_HOLD = 5
SLIP = 0.002
MIN_ROOM = 0.03
MIN_STOP = 0.02
FIXED_STOP = 0.05
ATR_MULT = 1.5
EVENT_DAYS = 7

ARR = {}  # ticker -> (open, high, low, close) arrays


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
    """Returns dict(earn, down, start) or None if earnings history is unavailable."""
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
            start = max(start, idx.min())  # can't filter before downgrade history begins
    except Exception:
        pass
    return dict(earn=earn.values.astype("datetime64[D]"),
                down=down.values.astype("datetime64[D]"),
                start=np.datetime64(start, "D"))


def has_event(dates, d):
    if len(dates) == 0:
        return False
    lo = np.searchsorted(dates, d - np.timedelta64(EVENT_DAYS, "D"), side="left")
    hi = np.searchsorted(dates, d, side="right")
    return hi > lo


# ---------------- build candidate trades ----------------
def build_candidates(sym, px, spy_gap, ev):
    o, h, l, c = (px[k].values for k in ["Open", "High", "Low", "Close"])
    ARR[sym] = (o, h, l, c)
    dates = px.index.values.astype("datetime64[D]")
    gap = spy_gap.reindex(px.index).values
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    out, last = [], {True: -99, False: -99}
    for t in range(20, len(c) - MAX_HOLD + 1):
        d = dates[t]
        if d < ev["start"] or not (gap[t] > 0):
            continue
        dip = bool(c[t - 1] / c[t - 6] - 1 < DIP_THRESHOLD)
        if t - last[dip] < MAX_HOLD:
            continue
        wh, wl = h[t - 10:t], l[t - 5:t]
        hi_i, lo_i = t - 10 + int(np.argmax(wh)), t - 5 + int(np.argmin(wl))
        if hi_i >= lo_i:
            continue
        hi, lo = wh.max(), wl.min()
        entry = o[t] * (1 + SLIP)
        t50, t618 = lo + 0.5 * (hi - lo), lo + 0.618 * (hi - lo)
        if t50 < entry * (1 + MIN_ROOM):
            continue
        if has_event(ev["earn"], d) or has_event(ev["down"], d):
            continue
        s_low = entry * 0.80 if lo_i == t - 1 else lo
        out.append(dict(
            tk=sym, t=t, date=d, dip=dip, entry=entry, t50=t50, t618=t618,
            s_low5d=min(s_low, entry * (1 - MIN_STOP)),
            s_fixed5=entry * (1 - FIXED_STOP),
            s_atr=entry - ATR_MULT * tr[t - 14:t].mean(),
        ))
        last[dip] = t
    return out


# ---------------- trade simulation ----------------
def simulate(cand, kind, tprice, sprice):
    o, hh, ll, cc = ARR[cand["tk"]]
    t, entry = cand["t"], cand["entry"]
    for d in range(MAX_HOLD):
        i = t + d
        if sprice is not None:
            if d > 0 and o[i] <= sprice:
                return o[i] * (1 - SLIP) / entry - 1, "stop"
            if ll[i] <= sprice:
                return sprice * (1 - SLIP) / entry - 1, "stop"
        if kind == "close" and cc[i] > entry:
            return cc[i] * (1 - SLIP) / entry - 1, "target"
        if kind == "limit":
            if d > 0 and o[i] >= tprice:
                return o[i] * (1 - SLIP) / entry - 1, "target"
            if hh[i] >= tprice:
                return tprice * (1 - SLIP) / entry - 1, "target"
    return cc[t + MAX_HOLD - 1] * (1 - SLIP) / entry - 1, "time"


def run_group(cands, p50, p618):
    targets = {
        "above_entry": lambda c: ("close", None),
        "plus3": lambda c: ("limit", c["entry"] * 1.03),
        "fib50": lambda c: ("limit", c["t50"]),
        "fib618": lambda c: ("limit", c["t618"]),
        "ctrl_fixed_A": lambda c: ("limit", c["entry"] * (1 + p50)),
        "ctrl_fixed_B": lambda c: ("limit", c["entry"] * (1 + p618)),
    }
    stops = {"low5d": "s_low5d", "fixed5": "s_fixed5", "atr": "s_atr"}
    combos = [(t, s, "primary") for t in ["above_entry", "plus3", "fib50", "fib618"] for s in stops]
    combos += [(t, s, "control") for t in ["ctrl_fixed_A", "ctrl_fixed_B"] for s in stops]
    rows = []
    for tn, sn, kind in combos + [("hold5", "none", "control")]:
        res = []
        for c in cands:
            if tn == "hold5":
                res.append(simulate(c, "none", None, None))
            else:
                k, p = targets[tn](c)
                res.append(simulate(c, k, p, c[stops[sn]]))
        r = np.array([x[0] for x in res])
        why = np.array([x[1] for x in res])
        if len(r) == 0:
            continue
        w, lz = r[r > 0], r[r <= 0]
        rows.append(dict(
            target=tn, stop=sn, type=kind, n=len(r),
            win_pct=100 * (r > 0).mean(), avg_pct=100 * r.mean(), median_pct=100 * np.median(r),
            plus_minus=100 * 2 * r.std() / np.sqrt(len(r)),
            avg_win=100 * w.mean() if len(w) else 0, avg_loss=100 * lz.mean() if len(lz) else 0,
            reward_risk=(w.mean() / abs(lz.mean())) if len(w) and len(lz) and lz.mean() != 0 else np.nan,
            stopped_pct=100 * (why == "stop").mean(), target_pct=100 * (why == "target").mean(),
            time_pct=100 * (why == "time").mean(),
        ))
    return pd.DataFrame(rows)


# ---------------- main ----------------
def main():
    args = sys.argv[1:]
    final = "final" in [a.lower() for a in args]
    tickers = [a.upper().lstrip("$") for a in args if a.lower() != "final"] or TICKERS
    pd.set_option("display.width", 250, "display.max_columns", 30)

    print(f"Downloading {HISTORY} of data (SPY + {len(tickers)} tickers)...")
    spy = load_prices("SPY")
    if spy is None:
        print("Could not download SPY data. Check your connection.")
        return
    spy_gap = spy["Open"] / spy["Close"].shift(1) - 1

    cands, used, skipped = [], [], []
    for sym in tickers:
        try:
            px, ev = load_prices(sym), None
            if px is not None:
                ev = load_events(sym)
            if px is None or ev is None:
                skipped.append(sym)
                continue
            cands += build_candidates(sym, px, spy_gap, ev)
            used.append(sym)
        except Exception as e:
            skipped.append(f"{sym} ({e})")
    print(f"Used {len(used)} tickers. Skipped (no price or earnings history): {skipped or 'none'}")

    if final:
        cands = [c for c in cands if c["date"] > SPLIT_DATE]
        print("\n*** HELD-OUT PERIOD (2024 onward). Look at this ONCE. Do not tune rules afterward. ***")
    else:
        cands = [c for c in cands if c["date"] <= SPLIT_DATE]
        print("\nDESIGN PERIOD (through 2023-12-31). Held-out period is untouched.")

    dip = [c for c in cands if c["dip"]]
    nodip = [c for c in cands if not c["dip"]]
    print(f"Trades: {len(dip)} dip trades, {len(nodip)} control trades (same rules, no -5% dip)")
    if len(dip) < 30:
        print("Too few dip trades to say anything. Try more tickers.")
        return

    p50 = float(np.median([c["t50"] / c["entry"] - 1 for c in dip]))
    p618 = float(np.median([c["t618"] / c["entry"] - 1 for c in dip]))
    print(f"Typical Fibonacci distance above entry: 50% level = +{p50:.1%}, 61.8% level = +{p618:.1%}")

    sp = spy["Close"].shift(-(MAX_HOLD - 1)) / spy["Open"] - 1
    lo, hi = (None, SPLIT_DATE) if not final else (SPLIT_DATE, None)
    sp = sp[[(lo is None or d > pd.Timestamp(lo)) and (hi is None or d <= pd.Timestamp(hi))
             for d in sp.index]].dropna()
    print(f"Benchmark: SPY average {MAX_HOLD}-day open-to-close return in this period: {sp.mean():.2%}")

    d_tab = run_group(dip, p50, p618)
    n_tab = run_group(nodip, p50, p618)

    fmt = {c: "{:.2f}" for c in d_tab.columns if c not in ("target", "stop", "type", "n")}
    print("\n=== DIP TRADES: all combinations (12 primary + controls) ===")
    print("(avg_pct = average return per trade after costs; plus_minus = rough 95% margin of error)")
    print(d_tab.round(2).to_string(index=False))

    cmp_ = d_tab.merge(n_tab, on=["target", "stop"], suffixes=("_dip", "_nodip"))
    cmp_["dip_minus_nodip"] = cmp_["avg_pct_dip"] - cmp_["avg_pct_nodip"]
    print("\n=== Does the DIP itself matter? Same rules on non-dip days ===")
    print(cmp_[["target", "stop", "n_dip", "avg_pct_dip", "n_nodip", "avg_pct_nodip",
                "dip_minus_nodip"]].round(2).to_string(index=False))

    d_tab.to_csv("dip_results_final.csv" if final else "dip_results_design.csv", index=False)
    print("""
HOW TO READ THIS
  - avg_pct above 0 AND bigger than its plus_minus is the minimum bar. Many rows will look
    positive by luck: 12 combos were tested, so about 1 in 20 per combo can pass by chance alone.
  - Compare to the benchmark and to 'hold5'. A strategy that doesn't beat just holding isn't adding anything.
  - Compare dip rows to non-dip rows. If non-dip days do as well, the dip isn't the edge.
  - Compare fib50/fib618 to ctrl_fixed_A/B. If plain fixed targets do as well, Fibonacci adds nothing.
  - reward_risk below 1 means average losses are bigger than average wins; the win rate must make up for it.
CAUTIONS
  - Survivorship bias: tickers are companies that still trade, which flatters dip buying.
  - Daily data can't show order of events inside a day. Stops assumed to hit first.
  - Earnings/downgrade history from Yahoo may be incomplete; trades before its coverage are excluded.
  - Same-day exits ('above_entry') would count as day trades; check your broker's rules.""")


if __name__ == "__main__":
    main()
