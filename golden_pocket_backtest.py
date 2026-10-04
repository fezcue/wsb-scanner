"""
Golden Pocket Pullback Backtest (rules locked in BEFORE looking at results)

IDEA: a stock in an uptrend makes a big move up, pulls back 50%-61.8% of that move
(the "golden pocket"), then resumes upward. Signal is read at a close; we buy the NEXT open.

ENTRY (all must be true, using only information known at the prior close):
  1. Uptrend: close > its 50-day average, and that average is higher than it was 10 days earlier
  2. Swing high = highest high of the last 20 days. Swing low = lowest low in the 40 days BEFORE
     that high. The move from low to high must be at least 20%
  3. The swing high happened at least 3 days ago (a real pullback has occurred)
  4. Retracement = (high - close) / (high - low) is between 0.50 and 0.618
  5. Price has not traded below the swing low since the high
  6. SPY's close is above its own 50-day average
  7. No earnings date and no analyst downgrade in the prior 7 calendar days
  8. At most one trade per ticker per 10 days
  (Mechanical skip: if the next open gaps below the swing low or above the swing high.)

EXITS (first to happen): target = retest of swing high OR 1.272 extension;
  stop = swing low OR 2 x 14-day average daily range below entry; else close of day 10.
  If stop and target fit in one day the stop is assumed first. Gaps fill at the open.
  Costs: 0.2% slippage each side.   => 2 targets x 2 stops = 4 primary combinations.

CONTROLS (identical except the pullback depth): shallow (23.6-38.2%), deep (70-85%),
  any_depth (no pullback requirement), plus "hold 10 days" with no exits.

PASS BAR (design period, per combination) - all four must be true:
  1. At least 150 trades
  2. Average return > 0 and larger than its margin of error
  3. Beats hold-10, shallow and deep controls by more than the combined margin of error
  4. Positive average in BOTH 2016-2019 and 2020-2023 (at least 30 trades each)

USAGE:
  python golden_pocket_backtest.py          -> DESIGN period (through 2023-12-31)
  python golden_pocket_backtest.py final    -> HELD-OUT period (2024+). Run ONCE, and only if
                                               a combination passed all four checks.
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
MID_DATE = np.datetime64("2019-12-31")
MAX_HOLD = 10
SLIP = 0.002
MIN_MOVE = 0.20
HIGH_WINDOW, LOW_WINDOW, MIN_DAYS_SINCE_HIGH = 20, 40, 3
COOLDOWN = 10
ATR_MULT = 2.0
EXT = 0.272
EVENT_DAYS = 7
ZONES = {"golden": (0.50, 0.618), "shallow": (0.236, 0.382), "deep": (0.70, 0.85), "any_depth": (0.0, 1.0)}
PRIMARY = [("retest", "swing_low"), ("retest", "atr2"), ("ext1272", "swing_low"), ("ext1272", "atr2")]

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
    out, last = [], {z: -99 for z in ZONES}
    for t in range(75, len(c) - MAX_HOLD + 1):
        s, d = t - 1, dates[t]
        if d < ev["start"] or not (ok[s] == 1.0):
            continue
        if not (c[s] > sma50[s] and sma50[s] > sma50[s - 10]):
            continue
        wh = h[s - HIGH_WINDOW + 1:s + 1]
        hi_i, hi = s - HIGH_WINDOW + 1 + int(np.argmax(wh)), wh.max()
        if s - hi_i < MIN_DAYS_SINCE_HIGH:
            continue
        lo = l[hi_i - LOW_WINDOW:hi_i].min()
        if (hi - lo) / lo < MIN_MOVE or l[hi_i:s + 1].min() <= lo:
            continue
        r = (hi - c[s]) / (hi - lo)
        entry = o[t] * (1 + SLIP)
        if not (lo < entry < hi):
            continue
        if has_event(ev["earn"], d) or has_event(ev["down"], d):
            continue
        atr = tr[t - 14:t].mean()
        for z, (a, b) in ZONES.items():
            if a <= r <= b and t - last[z] >= COOLDOWN:
                out.append(dict(tk=sym, t=t, date=d, zone=z, entry=entry, hi=hi, lo=lo,
                                ext=hi + EXT * (hi - lo), atr=atr))
                last[z] = t
    return out


# ---------------- simulation ----------------
def simulate(cand, tprice, sprice):
    o, hh, ll, cc = ARR[cand["tk"]]
    t, entry = cand["t"], cand["entry"]
    for d in range(MAX_HOLD):
        i = t + d
        if sprice is not None:
            if d > 0 and o[i] <= sprice:
                return o[i] * (1 - SLIP) / entry - 1, "stop"
            if ll[i] <= sprice:
                return sprice * (1 - SLIP) / entry - 1, "stop"
        if tprice is not None:
            if d > 0 and o[i] >= tprice:
                return o[i] * (1 - SLIP) / entry - 1, "target"
            if hh[i] >= tprice:
                return tprice * (1 - SLIP) / entry - 1, "target"
    return cc[t + MAX_HOLD - 1] * (1 - SLIP) / entry - 1, "time"


def eval_combo(cands, tn, sn):
    res = []
    for c in cands:
        if tn == "hold10":
            res.append(simulate(c, None, None))
        else:
            tp = c["hi"] if tn == "retest" else c["ext"]
            sp = c["lo"] if sn == "swing_low" else c["entry"] - ATR_MULT * c["atr"]
            res.append(simulate(c, tp, sp))
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
        stopped_pct=100 * (why == "stop").mean(), target_pct=100 * (why == "target").mean(),
        time_pct=100 * (why == "time").mean())


def cell(s):
    return "n/a" if s is None else f"{s['avg_pct']:+.2f}±{s['plus_minus']:.2f} (n={s['n']})"


# ---------------- pass/fail checklist ----------------
def verdict(by_zone, cache, subperiods):
    rows, hold = [], cache[("golden", "hold10", "none")]
    g = by_zone["golden"]
    for tn, sn in PRIMARY:
        s = cache[("golden", tn, sn)]
        row = {"target": tn, "stop": sn}
        if s is None:
            rows.append({**row, "1_trades>=150": "FAIL", "2_beats_zero": "FAIL", "3_beats_controls": "FAIL",
                         "4_both_periods": "FAIL", "OVERALL": "FAIL"})
            continue

        def beats(o):
            return o is not None and s["avg_pct"] - o["avg_pct"] > np.hypot(s["plus_minus"], o["plus_minus"])

        c1 = s["n"] >= 150
        c2 = s["avg_pct"] > 0 and s["avg_pct"] > s["plus_minus"]
        c3 = beats(hold) and beats(cache[("shallow", tn, sn)]) and beats(cache[("deep", tn, sn)])
        c4 = None
        if subperiods:
            ok = True
            for part in ([c for c in g if c["date"] <= MID_DATE], [c for c in g if c["date"] > MID_DATE]):
                r, _ = eval_combo(part, tn, sn)
                ok = ok and len(r) >= 30 and r.mean() > 0
            c4 = ok
        flags = [c1, c2, c3] + ([c4] if c4 is not None else [])
        rows.append({**row, "1_trades>=150": "pass" if c1 else "FAIL", "2_beats_zero": "pass" if c2 else "FAIL",
                     "3_beats_controls": "pass" if c3 else "FAIL",
                     "4_both_periods": "n/a" if c4 is None else ("pass" if c4 else "FAIL"),
                     "OVERALL": "PASS" if all(flags) else "FAIL"})
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
    sma = spy["Close"].rolling(50).mean()
    spy_ok = (spy["Close"] > sma).astype(float).where(sma.notna())

    cands, used, skipped = [], [], []
    for sym in tickers:
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

    if final:
        cands = [c for c in cands if c["date"] > SPLIT_DATE]
        print("\n*** HELD-OUT PERIOD (2024 onward). Look at this ONCE. Do not tune rules afterward. ***")
    else:
        cands = [c for c in cands if c["date"] <= SPLIT_DATE]
        print("\nDESIGN PERIOD (through 2023-12-31). Held-out period is untouched.")

    by_zone = {z: [c for c in cands if c["zone"] == z] for z in ZONES}
    print("Trades per group: " + ", ".join(f"{z}={len(v)}" for z, v in by_zone.items()))
    if len(by_zone["golden"]) < 30:
        print("Too few golden pocket trades to say anything.")
        return

    sp = spy["Close"].shift(-(MAX_HOLD - 1)) / spy["Open"] - 1
    keep = (sp.index > pd.Timestamp(SPLIT_DATE)) if final else (sp.index <= pd.Timestamp(SPLIT_DATE))
    print(f"Benchmark: SPY average {MAX_HOLD}-day open-to-close return in this period: {sp[keep].dropna().mean():.2%}")

    cache, combos = {}, PRIMARY + [("hold10", "none")]
    for z in ZONES:
        for tn, sn in combos:
            cache[(z, tn, sn)] = summarize(*eval_combo(by_zone[z], tn, sn))

    rows = []
    for tn, sn in combos:
        s = cache[("golden", tn, sn)]
        if s:
            rows.append(dict(target=tn, stop=sn, **s))
    tab = pd.DataFrame(rows)
    print("\n=== GOLDEN POCKET TRADES (avg_pct = average return per trade after costs) ===")
    print(tab.round(2).to_string(index=False))

    print("\n=== DOES THE POCKET MATTER? Same exits, different pullback depth (avg ± margin of error) ===")
    cmp_rows = [{"target": tn, "stop": sn, **{z: cell(cache[(z, tn, sn)]) for z in ZONES}} for tn, sn in combos]
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    print("\n=== PRE-REGISTERED PASS/FAIL CHECKLIST ===")
    v = verdict(by_zone, cache, subperiods=not final)
    print(v.to_string(index=False))
    if final:
        print("\n(Held-out run: check 4 doesn't apply. Decide using checks 1-3 only.)")
    elif (v["OVERALL"] == "PASS").any():
        print("\nAt least one combination passed all four checks. It is ELIGIBLE for the held-out test "
              "(python golden_pocket_backtest.py final). Run it once, without changing any rules.")
    else:
        print("\nNo combination passed all four checks. Under the rules we set, this idea is NOT supported. "
              "Do not run 'final', and do not tweak settings to rescue it.")

    tab.to_csv("golden_results_final.csv" if final else "golden_results_design.csv", index=False)
    print("""
CAUTIONS
  - Survivorship bias: tickers still trade today, which flatters buying pullbacks in uptrends.
  - With 4 combinations, one can look good by luck about 1 time in 5 even if nothing works.
  - Daily data can't show the order of events inside a day; stops assumed to hit first.
  - The design period was already used for the dip test; only the held-out period is fully fresh.""")


if __name__ == "__main__":
    main()
