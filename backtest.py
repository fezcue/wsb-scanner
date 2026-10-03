"""
Backtest: did high rip scores actually come before big price jumps?

For many past dates, it computes the score using ONLY data available on that date,
then looks at what the stock did over the next HORIZON trading days.

Usage:
  python backtest.py
  python backtest.py GME AMC SOFI      (your own tickers)

IMPORTANT LIMITS (read the notes at the bottom of the output too):
  - Short interest is left out, because free data only has TODAY's value, not history.
  - Reddit buzz is left out for the same reason (no historical data).
  - The tickers are stocks that exist today, which flatters the results (survivorship bias).
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "GME", "AMC", "TSLA", "NVDA", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "COIN",
    "MARA", "RIOT", "HOOD", "AMD", "SMCI", "UPST", "AFRM", "CVNA", "BB", "PLUG",
    "FUBO", "SNAP", "RBLX", "DKNG", "OPEN",
]
HORIZON = 5          # look this many trading days ahead
RIP_THRESHOLD = 0.10 # "ripped" = up more than 10% over the horizon
STEP = 5             # sample every 5th day so windows overlap less
HISTORY = "3y"

WEIGHTS = {"volume": 0.40, "momentum": 0.30, "not_overbought": 0.20, "near_high": 0.10}


def clip01(s):
    return s.clip(0, 1)


def build_features(ticker):
    """One row per past date, using only information known on that date."""
    h = yf.Ticker(ticker).history(period=HISTORY, auto_adjust=True)
    if len(h) < 200:
        return None
    close, vol = h["Close"], h["Volume"]

    rel_volume = vol / vol.shift(1).rolling(20).mean()
    ret_5d = close / close.shift(5) - 1
    d = close.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    pct_of_high = close / close.rolling(126).max()   # ~6 months of trading days
    fwd_ret = close.shift(-HORIZON) / close - 1      # the future: used ONLY to grade

    df = pd.DataFrame({
        "volume": clip01((rel_volume - 1) / 4),
        "momentum": clip01((ret_5d + 0.05) / 0.25),
        "not_overbought": np.where(rsi <= 70, 1.0, clip01((90 - rsi) / 20)),
        "near_high": clip01((pct_of_high - 0.5) / 0.5),
        "fwd_ret": fwd_ret,
    }, index=h.index)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().iloc[::STEP]
    df["ticker"] = ticker
    return df


def rank_corr(a, b):
    """Spearman correlation: do higher values line up with higher results? (-1 to 1)"""
    return a.rank().corr(b.rank())


def bucket_table(df, col, n=5):
    df = df.copy()
    df["bucket"] = pd.qcut(df[col].rank(method="first"), n, labels=range(1, n + 1))
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "hit_rate": g["ripped"].mean(),
        "avg_fwd_ret": g["fwd_ret"].mean(),
        "median_fwd_ret": g["fwd_ret"].median(),
    })


def main():
    tickers = [t.upper().lstrip("$") for t in sys.argv[1:]] or DEFAULT_TICKERS
    print(f"Downloading {HISTORY} of history for {len(tickers)} tickers...")
    frames = []
    for t in tickers:
        try:
            f = build_features(t)
            if f is not None:
                frames.append(f)
            else:
                print(f"  skipped {t} (not enough data)")
        except Exception as e:
            print(f"  skipped {t}: {e}")
    if not frames:
        print("No data. Check your internet connection or ticker spelling.")
        return

    df = pd.concat(frames).sort_index()
    df["score"] = sum(df[k] * w for k, w in WEIGHTS.items())
    df["ripped"] = (df["fwd_ret"] > RIP_THRESHOLD).astype(int)

    base = df["ripped"].mean()
    print(f"\nTested {len(df):,} stock-days across {df['ticker'].nunique()} tickers.")
    print(f"'Ripped' = up more than {RIP_THRESHOLD:.0%} within {HORIZON} trading days.")
    print(f"Base rate (any random stock-day): {base:.1%} ripped, "
          f"average {HORIZON}-day return {df['fwd_ret'].mean():.2%}")

    pd.set_option("display.width", 200)
    print("\n=== Score buckets (1 = lowest scores, 5 = highest scores) ===")
    bt = bucket_table(df, "score")
    print(bt.round(3).to_string())
    lift = bt["hit_rate"].iloc[-1] / bt["hit_rate"].iloc[0] if bt["hit_rate"].iloc[0] else float("nan")
    print(f"\nTop bucket vs bottom bucket hit rate: {bt['hit_rate'].iloc[-1]:.1%} vs "
          f"{bt['hit_rate'].iloc[0]:.1%}  (ratio {lift:.2f}x)")
    print(f"Rank correlation, score vs later return: {rank_corr(df['score'], df['fwd_ret']):.3f}")

    print("\n=== Each ingredient on its own ===")
    rows = []
    for k in WEIGHTS:
        b = bucket_table(df, k)
        rows.append({"component": k,
                     "rank_corr": rank_corr(df[k], df["fwd_ret"]),
                     "top_bucket_hit": b["hit_rate"].iloc[-1],
                     "bottom_bucket_hit": b["hit_rate"].iloc[0]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Does it hold up across time? (top-bucket hit rate, first vs second half) ===")
    cut = df["score"].quantile(0.8)
    mid = len(df) // 2
    for name, part in [("first half", df.iloc[:mid]), ("second half", df.iloc[mid:])]:
        top = part[part["score"] >= cut]
        print(f"  {name}: top-scoring days ripped {top['ripped'].mean():.1%} "
              f"(base rate that half: {part['ripped'].mean():.1%}, n={len(top)})")

    df.to_csv("backtest_data.csv")
    print("\nSaved backtest_data.csv")
    print("""
HOW TO READ THIS
  - If hit_rate does NOT climb from bucket 1 to bucket 5, the score has no edge.
  - Rank correlation near 0 (say, between -0.05 and 0.05) means basically no relationship.
  - If the first and second halves disagree, the result is probably noise.
CAUTIONS
  - Survivorship bias: these are stocks that still trade today, so results look better than reality.
  - Tickers were picked in hindsight as 'WSB-type' names.
  - Neighboring days are not independent, so real certainty is lower than the row counts suggest.
  - No trading costs or slippage are included.""")


if __name__ == "__main__":
    main()
