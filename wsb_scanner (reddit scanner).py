"""
WSB Scanner: finds tickers trending on r/wallstreetbets, then scores them
with sentiment, engagement, and market-data tests.

Setup:
  pip install praw yfinance vaderSentiment pandas numpy
  Create a Reddit "script" app at https://www.reddit.com/prefs/apps, then set:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Run:  python wsb_scanner.py
NOTE: the score is a heuristic, NOT a calibrated probability. See backtest notes at bottom.
"""
import os, re, math
import numpy as np
import pandas as pd
import praw
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

POST_LIMIT = 200        # posts pulled from each of hot/new
MIN_MENTIONS = 3        # ignore tickers mentioned fewer times
TOP_N = 15

# Common all-caps words that are not tickers
BLACKLIST = {
    "YOLO", "DD", "CEO", "CFO", "IPO", "ETF", "USA", "GDP", "FOMO", "ATH", "EPS", "IMO",
    "LOL", "WSB", "FED", "SEC", "IRS", "FDA", "AI", "ALL", "FOR", "THE", "AND", "ARE",
    "NOT", "BUT", "CAN", "NEW", "OUT", "NOW", "BUY", "SELL", "PUT", "PUTS", "CALL",
    "CALLS", "HOLD", "MOON", "EDIT", "TLDR", "OP", "PM", "AM", "US", "UK", "EU", "IV",
    "RSI", "PE", "ITM", "OTM", "DCA", "TA", "FD", "FDS", "LEAP", "LEAPS", "GAIN", "LOSS",
}

# WSB slang that VADER doesn't know about
WSB_LEXICON = {
    "moon": 3.0, "rocket": 3.0, "tendies": 2.5, "squeeze": 2.0, "rip": 2.5, "ripping": 3.0,
    "calls": 1.5, "bullish": 2.5, "diamond": 2.0, "undervalued": 2.0, "breakout": 2.5,
    "puts": -1.5, "bearish": -2.5, "dump": -2.5, "bagholder": -2.5, "rugpull": -3.0,
    "drill": -2.5, "crash": -2.5, "overvalued": -2.0, "short": -1.0,
}

CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
BARE = re.compile(r"\b([A-Z]{2,5})\b")
IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


# ---------- 1. Reddit scan ----------
def get_reddit():
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "wsb-scanner/0.1"),
    )


def extract_tickers(text):
    found = {m.upper() for m in CASHTAG.findall(text)}           # strong signal
    found |= {m for m in BARE.findall(text) if m not in BLACKLIST}  # weak signal
    return found - BLACKLIST


def scan_reddit():
    reddit = get_reddit()
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(WSB_LEXICON)
    sub = reddit.subreddit("wallstreetbets")

    seen, rows = set(), []
    for post in list(sub.hot(limit=POST_LIMIT)) + list(sub.new(limit=POST_LIMIT)):
        if post.id in seen:
            continue
        seen.add(post.id)
        text = f"{post.title}\n{post.selftext or ''}"
        tickers = extract_tickers(text)
        if not tickers:
            continue
        sentiment = sia.polarity_scores(text)["compound"]
        has_image = post.url.lower().endswith(IMG_EXT)  # hook for screenshot analysis
        engagement = math.log1p(max(post.score, 0)) + math.log1p(post.num_comments)
        for t in tickers:
            rows.append(dict(ticker=t, sentiment=sentiment, engagement=engagement,
                             has_image=has_image, upvote_ratio=post.upvote_ratio))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby("ticker").agg(
        mentions=("ticker", "size"),
        avg_sentiment=("sentiment", "mean"),
        engagement=("engagement", "sum"),
        images=("has_image", "sum"),
        upvote_ratio=("upvote_ratio", "mean"),
    ).reset_index()
    return agg[agg.mentions >= MIN_MENTIONS]


# ---------- 2. Market-data tests ----------
def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])) if loss.iloc[-1] else 100.0


def market_metrics(ticker):
    """Returns None if the ticker isn't a real, tradable symbol."""
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period="6mo")
        if len(h) < 30:
            return None
        close, vol = h["Close"], h["Volume"]
        info = tk.info or {}
        return dict(
            price=float(close.iloc[-1]),
            rel_volume=float(vol.iloc[-1] / vol.iloc[-21:-1].mean()),
            ret_5d=float(close.iloc[-1] / close.iloc[-6] - 1),
            rsi=rsi(close),
            pct_of_high=float(close.iloc[-1] / close.max()),
            short_float=float(info.get("shortPercentOfFloat") or 0),
            mkt_cap=info.get("marketCap") or 0,
        )
    except Exception:
        return None


# ---------- 3. Scoring ----------
def clip01(x):
    return float(np.clip(x, 0, 1))


def rip_score(r):
    """Each component is scaled 0..1, then combined with hand-picked weights."""
    comps = {
        "sentiment": clip01((r.avg_sentiment + 1) / 2),          # social mood
        "buzz": clip01(math.log1p(r.mentions) / math.log1p(30)),  # mention volume
        "engagement": clip01(r.engagement / 150),
        "rel_volume": clip01((r.rel_volume - 1) / 4),             # 1x..5x average volume
        "momentum": clip01((r.ret_5d + 0.05) / 0.25),             # -5%..+20% over 5d
        "squeeze": clip01(r.short_float / 0.30),                  # short interest up to 30%
        "not_overbought": clip01((85 - r.rsi) / 35),              # penalize RSI > 70
        "room_to_run": clip01(1 - r.pct_of_high) * 0.5 + 0.5 * (1 if r.pct_of_high > 0.8 else 0),
    }
    w = dict(sentiment=.15, buzz=.10, engagement=.10, rel_volume=.20,
             momentum=.15, squeeze=.15, not_overbought=.10, room_to_run=.05)
    raw = sum(comps[k] * w[k] for k in w)
    prob = 1 / (1 + math.exp(-8 * (raw - 0.45)))  # squash to 0..1 (uncalibrated)
    return raw, prob


# ---------- 4. Main ----------
def main():
    print("Scanning r/wallstreetbets...")
    social = scan_reddit()
    if social.empty:
        print("No tickers found.")
        return
    out = []
    for r in social.itertuples():
        m = market_metrics(r.ticker)
        if not m:
            continue  # filters out false-positive "tickers"
        row = {**r._asdict(), **m}
        full = pd.Series(row)
        raw, prob = rip_score(full)
        row.update(score=round(raw, 3), rip_prob=round(prob, 3))
        out.append(row)

    res = pd.DataFrame(out).drop(columns="Index").sort_values("rip_prob", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    cols = ["ticker", "mentions", "avg_sentiment", "rel_volume", "ret_5d", "rsi",
            "short_float", "price", "score", "rip_prob"]
    print(res[cols].head(TOP_N).round(3).to_string(index=False))
    res.to_csv("wsb_scan_results.csv", index=False)
    print("\nSaved wsb_scan_results.csv")


if __name__ == "__main__":
    main()

# ---------- Next steps ----------
# BACKTEST: log each scan (with timestamp) to disk, then label each row by whether
#   the stock rose >10% in the following 5 trading days. Fit sklearn LogisticRegression
#   on the component features to replace the hand-picked weights; that is what turns
#   rip_prob into an actual calibrated probability.
# IMAGES: for posts with has_image=True, download post.url and send it to a vision model
#   (e.g. the Claude API) to extract the ticker, position type, and P/L from the screenshot.
# COMMENTS: also pull top comments per post (post.comments) for richer sentiment.
