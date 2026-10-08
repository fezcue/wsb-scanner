"""
Big Mover Test (quick, exploratory; rules set before running)

QUESTION: when the rip score is high, is the stock's move over the next 5 days BIGGER than
normal for that stock, and does it go up or down?

- Score read at each close; window = next open to close of day 5. No trading costs.
- Each move is divided by the stock's own normal 5-day move (60-day daily volatility x sqrt(5)),
  so 2.0 means "twice as big as usual for this stock".
- One observation per ticker every 5 days (limits overlap).
- PASS: top score group's normalized move beats the bottom group's by more than the combined
  margin of error, AND the top group is higher in both halves of the data.

USAGE:  python big_mover_test.py
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
H = 5
STEP = 5
BIG = 0.10
WEIGHTS = {"volume": 0.40, "momentum": 0.30, "not_overbought": 0.20, "near_high": 0.10}


def load(sym):
    h = yf.Ticker(sym).history(period=HISTORY, auto_adjust=True)
    if h is None or len(h) < 300:
        return None
    idx = pd.DatetimeIndex(h.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
    h.index = idx.normalize()
    return h[~h.index.duplicated()].dropna()


def build(sym):
    px = load(sym)
    if px is None:
        return None
    o, c, v = px["Open"], px["Close"], px["Volume"]
    rel_vol = v / v.shift(1).rolling(20).mean()
    ret5 = c / c.shift(5) - 1
    d = c.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    near = c / c.rolling(126).max()
    parts = pd.DataFrame({
        "volume": ((rel_vol - 1) / 4).clip(0, 1),
        "momentum": ((ret5 + 0.05) / 0.25).clip(0, 1),
        "not_overbought": pd.Series(np.where(rsi <= 70, 1.0, ((90 - rsi) / 20).clip(0, 1)), index=px.index).where(rsi.notna()),
        "near_high": ((near - 0.5) / 0.5).clip(0, 1),
    })
    df = parts.copy()
    df["score"] = sum(parts[k] * w for k, w in WEIGHTS.items())
    df["fwd"] = c.shift(-H) / o.shift(-1) - 1          # next open -> close of day 5
    normal_move = c.pct_change().rolling(60).std() * np.sqrt(H)
    df["norm_move"] = df["fwd"].abs() / normal_move
    df["ticker"] = sym
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.iloc[::STEP]


def rank_corr(a, b):
    return a.rank().corr(b.rank())


def bucket_table(df, col="score", n=5):
    b = pd.qcut(df[col].rank(method="first"), n, labels=range(1, n + 1))
    g = df.groupby(b, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "avg_norm_move": g["norm_move"].mean(),
        "margin": 2 * g["norm_move"].std() / np.sqrt(g.size()),
        "avg_abs_move_%": 100 * g["fwd"].apply(lambda x: x.abs().mean()),
        "up>10%_%": 100 * g["fwd"].apply(lambda x: (x > BIG).mean()),
        "down<-10%_%": 100 * g["fwd"].apply(lambda x: (x < -BIG).mean()),
        "avg_signed_%": 100 * g["fwd"].mean(),
    })
    return out


def main():
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"Downloading {HISTORY} of data for {len(TICKERS)} tickers...")
    frames = []
    for s in TICKERS:
        try:
            f = build(s)
            if f is not None:
                frames.append(f)
        except Exception as e:
            print(f"  skipped {s}: {e}")
    if not frames:
        print("No data came back. Check your connection.")
        return
    df = pd.concat(frames).sort_index()
    print(f"\nObservations: {len(df):,} across {df['ticker'].nunique()} tickers")
    print(f"Overall: average normalized move {df['norm_move'].mean():.2f} (1.0 ~ a typical week), "
          f"{100 * (df['fwd'] > BIG).mean():.1f}% rose >10%, {100 * (df['fwd'] < -BIG).mean():.1f}% fell >10%")

    print("\n=== SCORE QUINTILES (1 = lowest scores, 5 = highest) ===")
    bt = bucket_table(df)
    print(bt.round(2).to_string())

    diff = bt["avg_norm_move"].iloc[-1] - bt["avg_norm_move"].iloc[0]
    dm = np.hypot(bt["margin"].iloc[-1], bt["margin"].iloc[0])
    print(f"\nTop minus bottom normalized move: {diff:+.2f} (margin of error ±{dm:.2f})")
    print(f"Rank correlation, score vs move SIZE:      {rank_corr(df['score'], df['norm_move']):+.3f}")
    print(f"Rank correlation, score vs move DIRECTION: {rank_corr(df['score'], df['fwd']):+.3f}")

    print("\n=== EACH INGREDIENT vs move SIZE and DIRECTION (rank correlation) ===")
    rows = [{"ingredient": k, "vs_size": rank_corr(df[k], df["norm_move"]),
             "vs_direction": rank_corr(df[k], df["fwd"])} for k in WEIGHTS]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== HOLDS UP OVER TIME? (top vs bottom quintile, normalized move) ===")
    mid = len(df) // 2
    halves_ok = True
    for name, part in [("first half", df.iloc[:mid]), ("second half", df.iloc[mid:])]:
        b = bucket_table(part)
        d = b["avg_norm_move"].iloc[-1] - b["avg_norm_move"].iloc[0]
        halves_ok = halves_ok and d > 0
        print(f"  {name}: top {b['avg_norm_move'].iloc[-1]:.2f} vs bottom {b['avg_norm_move'].iloc[0]:.2f} (diff {d:+.2f})")

    print("\n=== VERDICT ===")
    if diff > dm and halves_ok:
        print("PASS: high scores come before bigger-than-usual moves, in both halves.")
    else:
        print("FAIL: no reliable link between the score and move size.")
    up, dn = bt["up>10%_%"].iloc[-1], bt["down<-10%_%"].iloc[-1]
    print(f"Direction check (top group): {up:.1f}% rose >10% vs {dn:.1f}% fell >10%.")
    print("""
HOW TO READ THIS
  - If size passes but direction doesn't (up and down about equal), the score is a 'something is
    about to move' gauge, not a buy signal. That is a useful but different thing.
  - Moves are measured before costs, and this is not a trading strategy.
CAUTIONS
  - Survivorship bias: all tickers still trade today. History has been seen before (exploratory).
  - Score and volatility can both be high just because a stock is in the news; this finds a link,
    not a cause.""")
    bt.to_csv("big_mover_results.csv")


if __name__ == "__main__":
    main()
