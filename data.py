"""Daily prices for a small global macro universe, from Yahoo's chart API, cached in data/.

Fifteen ETFs across equities, bonds, commodities and currencies (adjusted closes, so
dividends are in), plus three market indicators: VIX, the 10-year yield, the 3-month bill.
The cache is not redistributed.
"""
import calendar
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}

ASSETS = {
    "SPY": "equity", "EFA": "equity", "EEM": "equity", "EWJ": "equity",
    "TLT": "bond", "IEF": "bond", "LQD": "bond", "HYG": "bond",
    "GLD": "commodity", "SLV": "commodity", "DBC": "commodity", "USO": "commodity",
    "UUP": "fx", "FXE": "fx", "FXY": "fx",
}
INDICATORS = ("^VIX", "^TNX", "^IRX")
START, END = dt.date(2007, 1, 1), dt.date(2026, 11, 19)


def fetch(symbol):
    path = os.path.join(CACHE, symbol.replace("^", "_") + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    a = calendar.timegm(START.timetuple())
    b = calendar.timegm(END.timetuple())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(symbol)
           + f"?period1={a}&period2={b}&interval=1d&events=div%2Csplit")
    raw = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40))
    result = raw["chart"]["result"][0]
    quotes = result["indicators"]
    close = quotes["adjclose"][0]["adjclose"] if quotes.get("adjclose") else quotes["quote"][0]["close"]
    out = {"t": result["timestamp"], "close": close}
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    time.sleep(0.4)
    return out


def series(symbol):
    d = fetch(symbol)
    index = pd.to_datetime(d["t"], unit="s").normalize()
    s = pd.Series(d["close"], index=index, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def panel():
    """Prices and indicators on SPY's trading days, from the first day every asset trades."""
    prices = pd.DataFrame({s: series(s) for s in ASSETS})
    indicators = pd.DataFrame({s: series(s) for s in INDICATORS})
    days = prices["SPY"].dropna().index
    prices = prices.reindex(days).ffill(limit=5)
    indicators = indicators.reindex(days).ffill(limit=5)
    start = max(prices[c].first_valid_index() for c in prices)
    return prices.loc[start:], indicators.loc[start:]
