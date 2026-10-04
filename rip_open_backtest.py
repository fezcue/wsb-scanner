"""
Rip Score + Positive-Open Backtest  (EXPLORATORY: rules written before building)

IDEA: each day, rank the stocks by the rip score (at the prior close). If SPY, QQQ and DIA
all open above their prior close (stand-in for positive S&P/Nasdaq/Dow futures), buy the
top-ranked stocks at the open, and sell at +4% or at the close of day 5. NO stop-loss.

RIP SCORE (4 ingredients, short interest excluded - no free history):
  0.40 x volume + 0.30 x momentum + 0.20 x not_overbought + 0.10 x near_high
GROUPS:  high = rank 1-3,  decent = rank 4-8 (ranked among the stocks with data that day;
         needs at least 10 stocks with data)
ENTRY:   next open, market order, 0.2% slippage; one trade per ticker per 5 days
EXIT:    take-profit at entry x 1.04 (gap past it fills at the open), else close of day 5;
         0.2% slippage on exit. No stop. (A same-day exit would count as a day trade.)

SCENARIOS
  high_filter_on   PRIMARY: top 3, open filter on
  decent_filter_on ranks 4-8, open filter on (informational)
  high_filter_off  top 3 on ALL days   -> does the open filter help?
  all_filter_on    all stocks, filter on -> does the score help?
  random_filter_on 3 random stocks, filter on -> is the score better than blind picking?
  SPY (paired): SPY from the same open to the close of day 5, no costs

PASS BAR for high_filter_on (all required):
  1. at least 150 trades          2. average > 0 and larger than its margin of error
  3. beats high_filter_off, all_filter_on, random_filter_on and SPY by more than the margin
  4. positive average (30+ trades) in BOTH 2016-2021 and 2022-now

NOTE: this uses the same 10 years we have already looked at. A pass means "worth tracking
forward", not "confirmed".

USAGE:  python rip_open_backtest.py
"""
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
MAX_HOLD = 5
TAKE_PROFIT = 0.04
SLIP = 0.002
TOP_N, DECENT_END = 3, 8
MIN_UNIVERSE = 10
WEIGHTS = (0.40, 0.30, 0.20, 0.10)
MID_DATE = pd.Timestamp("2021-12-31")
SCENARIOS = ["high_filter_on", "decent_filter_on", "high_filter_off", "all_filter_on", "random_filter_on"]


# ---------------- data ----------------
def load(sym):
    h = yf.Ticker(sym).history(period=HISTORY, auto_adjust=True)
    if h is None or len(h) < 300:
        return None
    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h.index = idx.normalize()
    return h[~h.index.duplicated()].dropna()


def rip_score(px):
    close, vol = px["Close"], px["Volume"]
    rel_vol = vol / vol.shift(1).rolling(20).mean()
    ret5 = close / close.shift(5) - 1
    d = close.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    near = close / close.rolling(126).max()
    volume = ((rel_vol - 1) / 4).clip(0, 1)
    momentum = ((ret5 + 0.05) / 0.25).clip(0, 1)
    not_ob = pd.Series(np.where(rsi <= 70, 1.0, ((90 - rsi) / 20).clip(0, 1)), index=px.index).where(rsi.notna())
    near_high = ((near - 0.5) / 0.5).clip(0, 1)
    w = WEIGHTS
    return (w[0] * volume + w[1] * momentum + w[2] * not_ob + w[3] * near_high).replace([np.inf, -np.inf], np.nan)


# ---------------- simulation ----------------
def simulate(O, H, C, t, entry):
    target = entry * (1 + TAKE_PROFIT)
    for d in range(MAX_HOLD):
        i = t + d
        if d > 0 and O[i] >= target:
            return O[i] * (1 - SLIP) / entry - 1, "target"
        if H[i] >= target:
            return target * (1 - SLIP) / entry - 1, "target"
    return C[t + MAX_HOLD - 1] * (1 - SLIP) / entry - 1, "time"


def summarize(trades):
    if not trades:
        return None
    r = np.array([x["ret"] for x in trades])
    why = np.array([x["why"] for x in trades])
    n = len(r)
    w, lz = r[r > 0], r[r <= 0]
    return dict(
        n=n, win_pct=100 * (r > 0).mean(), avg_pct=100 * r.mean(), median_pct=100 * np.median(r),
        plus_minus=100 * 2 * (r.std() if n > 1 else 0) / np.sqrt(n),
        avg_win=100 * w.mean() if len(w) else 0, avg_loss=100 * lz.mean() if len(lz) else 0,
        reward_risk=(w.mean() / abs(lz.mean())) if len(w) and len(lz) and lz.mean() != 0 else np.nan,
        target_pct=100 * (why == "target").mean(), time_pct=100 * (why == "time").mean(),
        p5_pct=100 * np.percentile(r, 5), worst_pct=100 * r.min())


def cell(s):
    return "n/a" if s is None else f"{s['avg_pct']:+.2f}±{s['plus_minus']:.2f} (n={s['n']})"


# ---------------- main ----------------
def main():
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"Downloading {HISTORY} of data (SPY, QQQ, DIA + {len(TICKERS)} tickers)...")
    idx = {s: load(s) for s in ["SPY", "QQQ", "DIA"]}
    if any(v is None for v in idx.values()):
        print("Could not download SPY/QQQ/DIA. Check your connection.")
        return
    cal = idx["SPY"].index
    n = len(cal)
    gap = np.ones(n, dtype=bool)
    for p in idx.values():
        p = p.reindex(cal)
        gap &= (p["Open"] > p["Close"].shift(1)).values
    spy_o, spy_c = idx["SPY"]["Open"].values, idx["SPY"]["Close"].values

    names, O, H, C, S, OK = [], [], [], [], [], []
    for sym in TICKERS:
        try:
            px = load(sym)
            if px is None:
                continue
            sc = rip_score(px).reindex(cal)
            a = px.reindex(cal)
            good = a[["Open", "High", "Low", "Close"]].notna().all(axis=1).astype(int)
            ok = (good.rolling(MAX_HOLD).sum().shift(-(MAX_HOLD - 1)) == MAX_HOLD).values
            names.append(sym)
            O.append(a["Open"].values); H.append(a["High"].values); C.append(a["Close"].values)
            S.append(sc.values); OK.append(ok)
        except Exception as e:
            print(f"  skipped {sym}: {e}")
    k = len(names)
    print(f"Used {k} tickers over {n} trading days. Days with all three indexes opening up: {gap.mean():.0%}")
    if k < MIN_UNIVERSE:
        print("Not enough tickers with data.")
        return

    rng = np.random.default_rng(42)
    last = {}
    trades = {s: [] for s in SCENARIOS}
    for t in range(130, n - MAX_HOLD + 1):
        valid = [j for j in range(k) if not np.isnan(S[j][t - 1]) and OK[j][t]]
        if len(valid) < MIN_UNIVERSE:
            continue
        order = sorted(valid, key=lambda j: -S[j][t - 1])
        f_on = bool(gap[t])
        picks = {
            "high_filter_on": order[:TOP_N] if f_on else [],
            "decent_filter_on": order[TOP_N:DECENT_END] if f_on else [],
            "high_filter_off": order[:TOP_N],
            "all_filter_on": valid if f_on else [],
            "random_filter_on": list(rng.choice(valid, TOP_N, replace=False)) if f_on else [],
        }
        for sc, js in picks.items():
            for j in js:
                if t - last.get((sc, j), -99) < MAX_HOLD:
                    continue
                last[(sc, j)] = t
                entry = O[j][t] * (1 + SLIP)
                ret, why = simulate(O[j], H[j], C[j], t, entry)
                trades[sc].append(dict(ret=ret, why=why, date=cal[t],
                                       spy=spy_c[t + MAX_HOLD - 1] / spy_o[t] - 1, tk=names[j]))

    print("Trades: " + ", ".join(f"{s}={len(v)}" for s, v in trades.items()))
    prim = trades["high_filter_on"]
    if len(prim) < 30:
        print("Too few trades in the primary group to say anything.")
        return

    cache = {s: summarize(v) for s, v in trades.items()}
    spy_all = np.array([x["spy"] for x in prim])
    print(f"SPY average {MAX_HOLD}-day open-to-close return over the primary trades' windows: "
          f"{100 * spy_all.mean():+.2f}%")

    tab = pd.DataFrame([dict(scenario=s, **cache[s]) for s in SCENARIOS if cache[s]])
    print("\n=== ALL SCENARIOS (avg_pct = average return per trade after costs) ===")
    print(tab.round(2).to_string(index=False))
    print("\n(worst_pct = single worst trade; p5_pct = the 5th-percentile trade. There is no stop-loss, "
          "so these show the downside.)")

    print("\n=== AVERAGE ± MARGIN OF ERROR ===")
    for s in SCENARIOS:
        print(f"  {s:18s} {cell(cache[s])}")

    # ----- checklist -----
    s0 = cache["high_filter_on"]

    def beats(o):
        return o is not None and s0["avg_pct"] - o["avg_pct"] > np.hypot(s0["plus_minus"], o["plus_minus"])

    diff = np.array([x["ret"] for x in prim]) - spy_all
    spy_margin = 2 * diff.std() / np.sqrt(len(diff))
    early = summarize([x for x in prim if x["date"] <= MID_DATE])
    late = summarize([x for x in prim if x["date"] > MID_DATE])
    f = lambda b: "pass" if b else "FAIL"
    checks = {
        "1_trades>=150": s0["n"] >= 150,
        "2_avg>0&>margin": s0["avg_pct"] > 0 and s0["avg_pct"] > s0["plus_minus"],
        "3a_beats_filter_off": beats(cache["high_filter_off"]),
        "3b_beats_all_stocks": beats(cache["all_filter_on"]),
        "3c_beats_random": beats(cache["random_filter_on"]),
        "3d_beats_SPY": diff.mean() > spy_margin,
        "4_both_periods": bool(early and late and early["n"] >= 30 and late["n"] >= 30
                               and early["avg_pct"] > 0 and late["avg_pct"] > 0),
    }
    print("\n=== PRE-REGISTERED CHECKLIST (high_filter_on) ===")
    for name, ok in checks.items():
        print(f"  {name:22s} {f(ok)}")
    print(f"  first half  (<=2021): {cell(early)}")
    print(f"  second half (2022+) : {cell(late)}")
    if all(checks.values()):
        print("\nRESULT: PASS. This means 'worth tracking forward' (paper trade / journal), NOT 'proven'. "
              "We have looked at this history before, so luck is still possible.")
    else:
        print("\nRESULT: FAIL. Per our plan we don't tweak settings to rescue it.")

    tab.to_csv("rip_open_results.csv", index=False)
    print("""
CAUTIONS
  - Survivorship bias: all tickers still trade today. Exploratory: this history has been seen before.
  - No stop-loss: a high win rate can hide a few big losses. Check worst_pct and p5_pct.
  - ~6 comparisons were made, so one may pass by luck.
  - Same-day target hits would count as day trades; check your broker's rules.
  - Index gaps (SPY/QQQ/DIA) stand in for pre-market futures, which free data doesn't include.""")


if __name__ == "__main__":
    main()
