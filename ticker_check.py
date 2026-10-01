"""
Ticker Check: the market-data half of the WSB scanner. NO Reddit keys needed.

You give it stock tickers, it fetches real market data and gives each one a
heuristic "rip score". It's a ranking tool, not a real probability.

Usage:
  python ticker_check.py GME AMC TSLA
  python ticker_check.py            (uses the DEFAULT_TICKERS list below)
"""
import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["GME", "AMC", "TSLA", "NVDA", "PLTR"]  # edit this list if you like


def rsi(close, n=14):
    """RSI: a 0-100 gauge of how 'overbought' a stock is. Above 70 = run up a lot."""
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1]))


def get_metrics(ticker):
    """Download recent prices and compute the numbers we score on."""
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period="6mo")
        if len(h) < 30:
            return None  # not a real ticker, or too little data
        close, vol = h["Close"], h["Volume"]
        info = tk.info or {}
        return {
            "ticker": ticker,
            "price": float(close.iloc[-1]),
            "rel_volume": float(vol.iloc[-1] / vol.iloc[-21:-1].mean()),
            "ret_5d": float(close.iloc[-1] / close.iloc[-6] - 1),
            "rsi": rsi(close),
            "pct_of_high": float(close.iloc[-1] / close.max()),
            "short_float": float(info.get("shortPercentOfFloat") or 0),
        }
    except Exception as e:
        print(f"  could not get data for {ticker}: {e}")
        return None


def clip01(x):
    """Force a number into the 0 to 1 range."""
    return float(np.clip(x, 0, 1))


def score(m):
    """Turn each metric into a 0-1 score, then blend them with weights."""
    parts = {
        "rel_volume": clip01((m["rel_volume"] - 1) / 4),      # 1x to 5x normal volume
        "momentum": clip01((m["ret_5d"] + 0.05) / 0.25),      # -5% to +20% in 5 days
        "squeeze": clip01(m["short_float"] / 0.30),           # up to 30% shorted
        "not_overbought": clip01((85 - m["rsi"]) / 35),       # lower RSI scores higher
        "near_high": 1.0 if m["pct_of_high"] > 0.8 else clip01(1 - m["pct_of_high"]),
    }
    weights = {"rel_volume": .30, "momentum": .25, "squeeze": .25,
               "not_overbought": .15, "near_high": .05}
    raw = sum(parts[k] * weights[k] for k in weights)
    rip = 1 / (1 + math.exp(-8 * (raw - 0.40)))   # squash to 0-1 (uncalibrated)
    return round(raw, 3), round(rip, 3)


def main():
    tickers = [t.upper().lstrip("$") for t in sys.argv[1:]] or DEFAULT_TICKERS
    print(f"Checking: {', '.join(tickers)}\n")

    rows = []
    for t in tickers:
        m = get_metrics(t)
        if m:
            m["score"], m["rip_score"] = score(m)
            rows.append(m)

    if not rows:
        print("No data came back. Check the ticker spelling or your internet connection.")
        return

    df = pd.DataFrame(rows).sort_values("rip_score", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.round(3).to_string(index=False))
    df.to_csv("ticker_results.csv", index=False)
    print("\nSaved ticker_results.csv")


if __name__ == "__main__":
    main()
