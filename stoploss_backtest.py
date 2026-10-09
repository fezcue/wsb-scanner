"""
Stop-Loss Backtest: rip score + positive-open filter + 4% take-profit, NO time limit
(EXPLORATORY: rules written before building)

ENTRY: each day rank the stocks by the rip score at the prior close. If SPY, QQQ and DIA all
  open above their prior close, buy the top 3 at the open (0.2% slippage).
EXIT (first to happen, no time limit): take-profit at +4% above entry, or stop-loss at
  -4%, -8% or -12% below entry (PRIMARY = -8%). If both fit in one day the stop is assumed
  first. Gaps past a level fill at the open. 0.2% slippage on exit.
  Trades still open when the data ends are marked at the last close and counted separately.
ONE trade per ticker at a time (no re-entry while a position is open).

SCENARIOS (same exits for each): high_filter_on (PRIMARY), decent_filter_on (ranks 4-8),
  high_filter_off (top 3, all days), all_filter_on (every stock), random_filter_on (3 random).
  SPY is compared over each trade's own holding period (paired, no costs).

PASS BAR (high_filter_on at the -8% stop), all required:
  1. at least 150 trades          2. average > 0 and larger than its margin of error
  3. beats high_filter_off, all_filter_on, random_filter_on and SPY by more than the margin
  4. positive average (30+ trades) in BOTH 2016-2021 and 2022-now
  5. fewer than 5% of trades still open at the end of the data

USAGE:  python stoploss_backtest.py
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
TAKE_PROFIT = 0.04
STOPS = [0.04, 0.08, 0.12]
PRIMARY_STOP = 0.08
SLIP = 0.002
TOP_N, DECENT_END, MIN_UNIVERSE = 3, 8, 10
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
def exit_for(O, H, L, C, end, t, entry, stop_pct):
    """Return (exit_index, exit_price, reason) for a trade entered at the open of day t."""
    target, stop = entry * (1 + TAKE_PROFIT), entry * (1 - stop_pct)
    o, h, l = O[t:end + 1], H[t:end + 1], L[t:end + 1]
    s_gap = np.zeros(len(o), bool)
    s_gap[1:] = o[1:] <= stop
    t_gap = np.zeros(len(o), bool)
    t_gap[1:] = o[1:] >= target
    s_hit, t_hit = s_gap | (l <= stop), t_gap | (h >= target)
    si = int(s_hit.argmax()) if s_hit.any() else None
    ti = int(t_hit.argmax()) if t_hit.any() else None
    if si is None and ti is None:
        return end, C[end], "open"
    if ti is None or (si is not None and si <= ti):
        return t + si, (o[si] if s_gap[si] else stop), "stop"
    return t + ti, (o[ti] if t_gap[ti] else target), "target"


def run(picks, stop_pct, data, cal, spy_o, spy_c):
    O, H, L, C, last_valid = data
    open_until, trades = {}, []
    for t in sorted(picks):
        for j in picks[t]:
            if open_until.get(j, -1) >= t:
                continue
            entry = O[j][t] * (1 + SLIP)
            e, price, why = exit_for(O[j], H[j], L[j], C[j], last_valid[j], t, entry, stop_pct)
            open_until[j] = e
            trades.append(dict(ret=price * (1 - SLIP) / entry - 1, why=why, date=cal[t], t=t, e=e,
                               days=e - t + 1, spy=spy_c[e] / spy_o[t] - 1))
    return trades


def summarize(trades):
    if not trades:
        return None
    r = np.array([x["ret"] for x in trades])
    why = np.array([x["why"] for x in trades])
    days = np.array([x["days"] for x in trades])
    n = len(r)
    w, lz = r[r > 0], r[r <= 0]
    return dict(
        n=n, win_pct=100 * (r > 0).mean(), avg_pct=100 * r.mean(), median_pct=100 * np.median(r),
        plus_minus=100 * 2 * (r.std() if n > 1 else 0) / np.sqrt(n),
        avg_win=100 * w.mean() if len(w) else 0, avg_loss=100 * lz.mean() if len(lz) else 0,
        target_pct=100 * (why == "target").mean(), stop_pct=100 * (why == "stop").mean(),
        open_pct=100 * (why == "open").mean(), avg_days=days.mean(), max_days=int(days.max()),
        pct_per_day=100 * r.sum() / days.sum(), worst_pct=100 * r.min())


def max_concurrent(trades, n_days):
    diff = np.zeros(n_days + 2)
    for x in trades:
        diff[x["t"]] += 1
        diff[x["e"] + 1] -= 1
    return int(np.cumsum(diff).max())


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

    names, O, H, L, C, S, last_valid = [], [], [], [], [], [], []
    for sym in TICKERS:
        try:
            px = load(sym)
            if px is None:
                continue
            a = px.reindex(cal)
            names.append(sym)
            O.append(a["Open"].values); H.append(a["High"].values)
            L.append(a["Low"].values); C.append(a["Close"].values)
            S.append(rip_score(px).reindex(cal).values)
            last_valid.append(int(np.where(a["Close"].notna().values)[0].max()))
        except Exception as e:
            print(f"  skipped {sym}: {e}")
    k = len(names)
    print(f"Used {k} tickers over {n} trading days. Days with all three indexes opening up: {gap.mean():.0%}")
    if k < MIN_UNIVERSE:
        print("Not enough tickers with data.")
        return

    # daily picks (shared across stop levels so comparisons are fair)
    rng = np.random.default_rng(42)
    picks = {s: {} for s in SCENARIOS}
    for t in range(130, n - 1):
        valid = [j for j in range(k) if not np.isnan(S[j][t - 1]) and not np.isnan(O[j][t])
                 and not np.isnan(L[j][t])]
        if len(valid) < MIN_UNIVERSE:
            continue
        order = sorted(valid, key=lambda j: -S[j][t - 1])
        f_on = bool(gap[t])
        day = {"high_filter_off": order[:TOP_N]}
        if f_on:
            day.update({"high_filter_on": order[:TOP_N], "decent_filter_on": order[TOP_N:DECENT_END],
                        "all_filter_on": valid,
                        "random_filter_on": [int(x) for x in rng.choice(valid, TOP_N, replace=False)]})
        for s, js in day.items():
            if js:
                picks[s][t] = js

    data = (O, H, L, C, last_valid)
    print("Running simulations (this can take a few minutes)...")
    res = {}
    for s in SCENARIOS:
        for sp in STOPS:
            res[(s, sp)] = run(picks[s], sp, data, cal, spy_o, spy_c)
    cache = {key: summarize(v) for key, v in res.items()}

    prim = res[("high_filter_on", PRIMARY_STOP)]
    if len(prim) < 30:
        print("Too few trades in the primary group to say anything.")
        return

    cols = ["scenario", "n", "win_pct", "avg_pct", "median_pct", "plus_minus", "avg_win", "avg_loss",
            "target_pct", "stop_pct", "open_pct", "avg_days", "max_days", "pct_per_day", "worst_pct"]
    rows = [dict(scenario=s, **cache[(s, PRIMARY_STOP)]) for s in SCENARIOS if cache[(s, PRIMARY_STOP)]]
    print(f"\n=== ALL SCENARIOS at the {PRIMARY_STOP:.0%} stop (avg_pct = average return per trade after costs) ===")
    print(pd.DataFrame(rows)[cols].round(2).to_string(index=False))
    print("(pct_per_day = total return divided by total trading days held; open_pct = still open at data end)")

    print("\n=== PRIMARY SCENARIO (high_filter_on) AT EACH STOP LEVEL ===")
    rows = []
    for sp in STOPS:
        s = cache[("high_filter_on", sp)]
        if s:
            rows.append(dict(stop=f"-{sp:.0%}", **{c: s[c] for c in cols[1:]},
                             max_open_positions=max_concurrent(res[("high_filter_on", sp)], n)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n=== AVERAGE ± MARGIN OF ERROR, every scenario and stop ===")
    for s in SCENARIOS:
        print(f"  {s:18s} " + "   ".join(f"-{sp:.0%}: {cell(cache[(s, sp)])}" for sp in STOPS))

    # ----- checklist -----
    print("\n=== PRE-REGISTERED CHECKLIST (high_filter_on; the -8% stop decides, others informational) ===")
    verdicts = {}
    for sp in STOPS:
        tr = res[("high_filter_on", sp)]
        s0 = cache[("high_filter_on", sp)]
        if not s0:
            continue

        def beats(o):
            return o is not None and s0["avg_pct"] - o["avg_pct"] > np.hypot(s0["plus_minus"], o["plus_minus"])

        r = np.array([x["ret"] for x in tr])
        sp_r = np.array([x["spy"] for x in tr])
        diff = r - sp_r
        early = summarize([x for x in tr if x["date"] <= MID_DATE])
        late = summarize([x for x in tr if x["date"] > MID_DATE])
        checks = {
            "1_trades>=150": s0["n"] >= 150,
            "2_avg>0&>margin": s0["avg_pct"] > 0 and s0["avg_pct"] > s0["plus_minus"],
            "3a_beats_filter_off": beats(cache[("high_filter_off", sp)]),
            "3b_beats_all_stocks": beats(cache[("all_filter_on", sp)]),
            "3c_beats_random": beats(cache[("random_filter_on", sp)]),
            "3d_beats_SPY": diff.mean() > 2 * diff.std() / np.sqrt(len(diff)),
            "4_both_periods": bool(early and late and early["n"] >= 30 and late["n"] >= 30
                                   and early["avg_pct"] > 0 and late["avg_pct"] > 0),
            "5_<5%_open_at_end": s0["open_pct"] < 5,
        }
        verdicts[sp] = all(checks.values())
        tag = "PRIMARY" if sp == PRIMARY_STOP else "informational"
        print(f"\n  Stop -{sp:.0%} ({tag}): " + ("PASS" if verdicts[sp] else "FAIL"))
        print("    " + ", ".join(f"{k}={'pass' if v else 'FAIL'}" for k, v in checks.items()))
        print(f"    first half {cell(early)}  |  second half {cell(late)}")

    if verdicts.get(PRIMARY_STOP):
        print("\nRESULT: PRIMARY PASSED. That means 'worth tracking forward', NOT 'proven'. "
              "This history has been seen before and many comparisons were made.")
    else:
        print("\nRESULT: PRIMARY FAILED. Per our plan we don't tweak settings to rescue it.")

    pd.DataFrame([dict(scenario=s, stop=sp, **cache[(s, sp)]) for (s, sp) in cache if cache[(s, sp)]]
                 ).to_csv("stoploss_results.csv", index=False)
    print("""
CAUTIONS
  - Survivorship bias is worst with no time limit: every stock here still trades today, so the ones that
    fell and never recovered are missing. This flatters 'hold until it works' strategies.
  - Unlimited capital is assumed; check max_open_positions for how much money this would tie up.
  - Same-day target/stop exits would count as day trades; check your broker's rules.
  - Index gaps (SPY/QQQ/DIA) stand in for pre-market futures, which free data doesn't include.""")


if __name__ == "__main__":
    main()
