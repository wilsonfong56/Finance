#!/usr/bin/env python3
"""
EMA Flip Stock Screener
========================
Identifies the "EMA Flip" setup — first tight flag supported by the 8/21 EMAs
after price flips these EMAs from a downtrend.  Gives early entries before
classic pivot breakouts.

Bullish:
  Setting Up  — price is flagging at/near the 8/21 EMAs after bullish cross
  Triggered   — bounce candle off EMAs (low wicks to/below EMA, closes above 8EMA)

Bearish:
  Setting Up  — price is flagging at/near the 8/21 EMAs after bearish cross
  Triggered   — rejection candle off EMAs (high wicks to/above EMA, closes below 8EMA)

Usage:
  python ema_flip.py              # prompts: spy/qqq/iwm/dia
  python ema_flip.py --ticker AAPL  # single-stock debug mode
  python ema_flip.py --side bear    # bearish only
"""

import argparse
import io
import json
import sys
import threading
import time
import webbrowser

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SCREENER_NAME = "EMA Flip"
DATA_PARAMS = {"period": "6mo", "interval": "1d"}


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _calc_ema(closes, period):
    """EMA matching the project's existing pattern."""
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return np.array(ema)


def _calc_atr(highs, lows, closes, period):
    """Average True Range over *period* bars (simple moving average)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1: i + 1])
    return atr


# ---------------------------------------------------------------------------
# Universe fetching (shared pattern with wedge_pop.py)
# ---------------------------------------------------------------------------

UNIVERSES = {
    "iwm": ("Russell 2000", "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"),
    "spy": ("S&P 500",      "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    "qqq": ("Nasdaq 100",   "https://en.wikipedia.org/wiki/Nasdaq-100"),
    "dia": ("Dow Jones 30", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"),
}


def _scrape_wiki_table(url, ticker_col_names, min_rows=0):
    """Generic Wikipedia table scraper that finds a ticker column."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        print(f"[!] Failed to fetch index list: {e}")
        sys.exit(1)

    for tbl in tables:
        if len(tbl) < min_rows:
            continue
        cols_lower = [str(c).lower() for c in tbl.columns]
        for col_name in ticker_col_names:
            if col_name in cols_lower:
                idx = cols_lower.index(col_name)
                tickers = tbl.iloc[:, idx].astype(str).tolist()
                tickers = [t.strip().replace(".", "-") for t in tickers if t.strip()]
                return tickers
    return None


def fetch_russell2000():
    """Fetch Russell 2000 constituents from iShares IWM ETF holdings CSV."""
    url = ("https://www.ishares.com/us/products/239710/"
           "ishares-russell-2000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWM_holdings&dataType=fund")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch IWM holdings: {e}")
        sys.exit(1)

    lines = resp.text.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,Name,"):
            header_idx = i
            break
    if header_idx is None:
        print("[!] Could not find header row in IWM holdings CSV")
        sys.exit(1)

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    equity = df[df["Asset Class"] == "Equity"]
    tickers = equity["Ticker"].astype(str).tolist()
    tickers = [t.strip().replace(".", "-") for t in tickers if t.strip()]
    return tickers


def fetch_sp500():
    tickers = _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)
    if tickers:
        return tickers
    print("[!] Could not find ticker column in S&P 500 Wikipedia table")
    sys.exit(1)


def fetch_nasdaq100():
    tickers = _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)
    if tickers:
        return tickers
    print("[!] Could not find ticker column in Nasdaq 100 Wikipedia table")
    sys.exit(1)


def fetch_dowjones():
    tickers = _scrape_wiki_table(UNIVERSES["dia"][1], ["symbol", "ticker"], min_rows=25)
    if tickers:
        return tickers
    print("[!] Could not find ticker column in Dow Jones Wikipedia table")
    sys.exit(1)


FETCH_FUNCTIONS = {
    "iwm": fetch_russell2000,
    "spy": fetch_sp500,
    "qqq": fetch_nasdaq100,
    "dia": fetch_dowjones,
}


def fetch_all_universes():
    all_tickers = set()
    for key, (name, _) in UNIVERSES.items():
        print(f"  Fetching {name}...")
        t = FETCH_FUNCTIONS[key]()
        print(f"    {len(t)} tickers")
        all_tickers.update(t)
    result = sorted(all_tickers)
    print(f"  {len(result)} unique tickers after deduplication")
    return result


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(tickers, months=6):
    all_tickers = list(set(tickers + ["SPY"]))
    period = f"{months}mo"
    chunk_size = 100
    frames = []

    total = len(all_tickers)
    for i in range(0, total, chunk_size):
        chunk = all_tickers[i: i + chunk_size]
        pct = min((i + chunk_size) / total * 100, 100)
        print(f"\r  Downloading... {pct:5.1f}%  ({min(i + chunk_size, total)}/{total} tickers)",
              end="", flush=True)
        try:
            df = yf.download(chunk, period=period, group_by="ticker",
                             progress=False, threads=True)
            frames.append(df)
        except Exception as e:
            print(f"\n  [!] Chunk download error: {e}")
    print()

    if not frames:
        print("[!] No data downloaded")
        sys.exit(1)

    data = pd.concat(frames, axis=1)
    return data


def extract_ohlcv(data, ticker):
    try:
        sub = data[ticker]
        o = sub["Open"].values.flatten().astype(float)
        h = sub["High"].values.flatten().astype(float)
        lo = sub["Low"].values.flatten().astype(float)
        c = sub["Close"].values.flatten().astype(float)
        v = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return None

    valid = ~np.isnan(c)
    if valid.sum() < 60:
        return None
    first = np.argmax(valid)
    return o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=120):
    try:
        sub = data[ticker]
        dates = sub.index
        o = sub["Open"].values.flatten().astype(float)
        h = sub["High"].values.flatten().astype(float)
        lo = sub["Low"].values.flatten().astype(float)
        c = sub["Close"].values.flatten().astype(float)
        v = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return []

    valid = ~np.isnan(c)
    if valid.sum() == 0:
        return []
    first = np.argmax(valid)
    dates, o, h, lo, c, v = dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]

    n = len(c)
    start = max(0, n - bars)
    records = []
    for i in range(start, n):
        if np.isnan(c[i]) or np.isnan(o[i]) or np.isnan(h[i]) or np.isnan(lo[i]):
            continue
        records.append({
            "time": dates[i].strftime("%Y-%m-%d"),
            "open": round(float(o[i]), 2),
            "high": round(float(h[i]), 2),
            "low": round(float(lo[i]), 2),
            "close": round(float(c[i]), 2),
            "volume": int(v[i]) if not np.isnan(v[i]) else 0,
        })
    return records


# ---------------------------------------------------------------------------
# Scan logic — Bullish
# ---------------------------------------------------------------------------

def scan_ticker_bullish(closes, highs, lows, volumes, spy_closes):
    """
    Run the bullish EMA Flip scan on a single ticker.

    Returns dict with phase info or None if no setup found.
    """
    n = len(closes)
    if n < 60:
        return None

    ema8 = _calc_ema(closes, 8)
    ema21 = _calc_ema(closes, 21)

    # --- Condition 1: 8EMA currently above 21EMA ---
    if ema8[-1] <= ema21[-1]:
        return None

    # --- Condition 2: Price at or above 21EMA (support holding, 1% tolerance) ---
    if closes[-1] < ema21[-1] * 0.99:
        return None

    # --- Condition 3: Recent bullish 8/21 EMA crossover (within last 25 bars) ---
    cross_idx = None
    for i in range(n - 1, max(n - 25, 0), -1):
        if i < 1:
            break
        if ema8[i] > ema21[i] and ema8[i - 1] <= ema21[i - 1]:
            cross_idx = i
            break

    if cross_idx is None:
        return None

    # --- Condition 4: Prior bearish period ---
    bearish = sum(1 for i in range(max(0, cross_idx - 30), cross_idx)
                  if ema8[i] < ema21[i])
    if bearish < 5:
        return None

    below_21 = sum(1 for i in range(max(0, cross_idx - 15), cross_idx)
                   if closes[i] < ema21[i])
    if below_21 < 3:
        return None

    # --- Condition 5: Flip confirmed — price closed above 8EMA after cross ---
    if not any(closes[i] > ema8[i] for i in range(cross_idx, n)):
        return None

    # --- Condition 6: Near EMAs (pullback) in last 5 bars ---
    near_ema = False
    for i in range(max(n - 5, cross_idx), n):
        to_8 = (lows[i] - ema8[i]) / ema8[i] * 100
        to_21 = (lows[i] - ema21[i]) / ema21[i] * 100
        if -1.5 <= to_8 <= 1.5 or -1.5 <= to_21 <= 1.5:
            near_ema = True
            break
        if ema21[i] <= closes[i] <= ema8[i]:
            near_ema = True
            break

    if not near_ema:
        return None

    # --- Quality metrics ---
    atr5 = _calc_atr(highs, lows, closes, 5)
    atr20 = _calc_atr(highs, lows, closes, 20)
    tight = (not np.isnan(atr5[-1]) and not np.isnan(atr20[-1])
             and atr5[-1] < atr20[-1])

    flag_high = float(max(highs[cross_idx:]))
    price = closes[-1]
    rs = _calc_rs(closes, spy_closes)
    cross_bars_ago = n - 1 - cross_idx
    pct_above_8 = (price - ema8[-1]) / ema8[-1] * 100

    # --- Phase: Triggered vs Setting Up ---
    triggered = False
    for i in range(max(n - 2, cross_idx), n):
        wick_to_ema = lows[i] <= ema8[i] * 1.015 or lows[i] <= ema21[i] * 1.015
        closes_above_8 = closes[i] > ema8[i]
        if wick_to_ema and closes_above_8:
            triggered = True
            break

    if triggered and pct_above_8 > 3.0:
        return None
    if not triggered and pct_above_8 > 2.0:
        return None

    signal = "triggered" if triggered else "forming"

    return {
        "phase": 4 if triggered else 3,
        "signal": signal,
        "type": "bullish",
        "price": round(float(price), 2),
        "flag_high": round(flag_high, 2),
        "flag_low": None,
        "ema8": round(float(ema8[-1]), 2),
        "ema21": round(float(ema21[-1]), 2),
        "pct_from_8ema": round(float(pct_above_8), 2),
        "cross_bars_ago": int(cross_bars_ago),
        "tight_flag": bool(tight),
        "rs": round(float(rs), 1),
    }


# ---------------------------------------------------------------------------
# Scan logic — Bearish
# ---------------------------------------------------------------------------

def scan_ticker_bearish(closes, highs, lows, volumes, spy_closes):
    """
    Run the bearish EMA Flip scan on a single ticker.

    Mirror of the bullish pattern: bearish 8/21 cross, price flagging below
    EMAs, rejection candle (high wicks to EMA, closes below 8EMA).

    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None

    ema8 = _calc_ema(closes, 8)
    ema21 = _calc_ema(closes, 21)

    # --- Condition 1: 8EMA currently below 21EMA ---
    if ema8[-1] >= ema21[-1]:
        return None

    # --- Condition 2: Price at or below 21EMA (resistance holding, 1% tolerance) ---
    if closes[-1] > ema21[-1] * 1.01:
        return None

    # --- Condition 3: Recent bearish 8/21 EMA crossover (within last 25 bars) ---
    cross_idx = None
    for i in range(n - 1, max(n - 25, 0), -1):
        if i < 1:
            break
        if ema8[i] < ema21[i] and ema8[i - 1] >= ema21[i - 1]:
            cross_idx = i
            break

    if cross_idx is None:
        return None

    # --- Condition 4: Prior bullish period ---
    # 8EMA above 21EMA for 5+ bars before cross
    bullish = sum(1 for i in range(max(0, cross_idx - 30), cross_idx)
                  if ema8[i] > ema21[i])
    if bullish < 5:
        return None

    # Price above 21EMA for 3+ of the 15 bars before cross
    above_21 = sum(1 for i in range(max(0, cross_idx - 15), cross_idx)
                   if closes[i] > ema21[i])
    if above_21 < 3:
        return None

    # --- Condition 5: Flip confirmed — price closed below 8EMA after cross ---
    if not any(closes[i] < ema8[i] for i in range(cross_idx, n)):
        return None

    # --- Condition 6: Near EMAs (bounce up to resistance) in last 5 bars ---
    near_ema = False
    for i in range(max(n - 5, cross_idx), n):
        to_8 = (highs[i] - ema8[i]) / ema8[i] * 100
        to_21 = (highs[i] - ema21[i]) / ema21[i] * 100
        if -1.5 <= to_8 <= 1.5 or -1.5 <= to_21 <= 1.5:
            near_ema = True
            break
        if ema8[i] <= closes[i] <= ema21[i]:
            near_ema = True
            break

    if not near_ema:
        return None

    # --- Quality metrics ---
    atr5 = _calc_atr(highs, lows, closes, 5)
    atr20 = _calc_atr(highs, lows, closes, 20)
    tight = (not np.isnan(atr5[-1]) and not np.isnan(atr20[-1])
             and atr5[-1] < atr20[-1])

    flag_low = float(min(lows[cross_idx:]))
    price = closes[-1]
    rs = _calc_rs(closes, spy_closes)
    cross_bars_ago = n - 1 - cross_idx
    pct_below_8 = (ema8[-1] - price) / ema8[-1] * 100

    # --- Phase: Triggered vs Setting Up ---
    # Triggered: rejection candle — high wicks to/above 8EMA or 21EMA, closes below 8EMA
    triggered = False
    for i in range(max(n - 2, cross_idx), n):
        wick_to_ema = highs[i] >= ema8[i] * 0.985 or highs[i] >= ema21[i] * 0.985
        closes_below_8 = closes[i] < ema8[i]
        if wick_to_ema and closes_below_8:
            triggered = True
            break

    if triggered and pct_below_8 > 3.0:
        return None
    if not triggered and pct_below_8 > 2.0:
        return None

    signal = "triggered" if triggered else "forming"

    return {
        "phase": 4 if triggered else 3,
        "signal": signal,
        "type": "bearish",
        "price": round(float(price), 2),
        "flag_high": None,
        "flag_low": round(flag_low, 2),
        "ema8": round(float(ema8[-1]), 2),
        "ema21": round(float(ema21[-1]), 2),
        "pct_from_8ema": round(float(-pct_below_8), 2),
        "cross_bars_ago": int(cross_bars_ago),
        "tight_flag": bool(tight),
        "rs": round(float(rs), 1),
    }


def _calc_rs(closes, spy_closes):
    """3-month relative strength: stock return minus SPY return (pct points)."""
    period = 63
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    if not results:
        print("\nNo EMA Flip setups found today.")
        return

    results.sort(key=lambda r: (
        0 if r["type"] == "bullish" else 1,
        0 if r["signal"] == "triggered" else 1,
        -r["rs"],
    ))

    hdr = (f"{'Ticker':<8} {'Type':<10} {'Signal':<12} {'Price':>8} {'8EMA':>8} {'21EMA':>8} "
           f"{'Cross':>7} {'Tight':>5} {'RS vs SPY':>10}  TradingView")
    print(f"\n{'=' * len(hdr)}")
    print(f" EMA Flip Screener \u2014 {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sig_label = "TRIGGERED" if r["signal"] == "triggered" else "Setting Up"
        tv_link = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        tight_str = "Y" if r["tight_flag"] else ""
        cross_str = f"{r['cross_bars_ago']}d"
        print(f"{r['ticker']:<8} {r['type'].upper():<10} {sig_label:<12} {r['price']:>8.2f} "
              f"{r['ema8']:>8.2f} {r['ema21']:>8.2f} {cross_str:>7} "
              f"{tight_str:>5} {r['rs']:>+10.1f}  {tv_link}")

    print()


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMA Flip Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28; --cyan: #39d2e0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  header { background: var(--surface); border-bottom: 1px solid var(--border);
           padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; }
  main { padding: 20px 24px; max-width: 1800px; margin: 0 auto; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); }
  .chart-card-header .ticker { font-size: 15px; font-weight: 700; color: var(--accent);
    text-decoration: none; }
  .chart-card-header .ticker:hover { text-decoration: underline; }
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .chart-card-body { height: 220px; position: relative; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }
  .badge.bull-triggered { background: rgba(63,185,80,0.2); color: var(--green); }
  .badge.bull-forming   { background: rgba(210,153,34,0.2); color: var(--yellow); }
  .badge.bear-triggered { background: rgba(248,81,73,0.2); color: var(--red); }
  .badge.bear-forming   { background: rgba(219,109,40,0.2); color: var(--orange); }
  .badge.tight { background: rgba(57,210,224,0.15); color: var(--cyan); }
  .info { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>EMA Flip Screener</h1>
  <span style="color:var(--muted);font-size:13px">__SCANNED__ tickers scanned in __ELAPSED__s &mdash; __COUNT__ setups</span>
</header>
<main>
  <div class="chart-grid" id="chart-grid"></div>
</main>
<script>
var RESULTS = __RESULTS_JSON__;
var CHART_DATA = __CHART_DATA_JSON__;

function calcEMA(closes, period) {
  var k = 2 / (period + 1);
  var ema = [closes[0]];
  for (var i = 1; i < closes.length; i++) {
    ema.push(closes[i] * k + ema[i - 1] * (1 - k));
  }
  return ema;
}

function buildCards() {
  var grid = document.getElementById('chart-grid');
  RESULTS.forEach(function(r) {
    var ohlcv = CHART_DATA[r.ticker];
    if (!ohlcv || !ohlcv.length) return;

    var card = document.createElement('div');
    card.className = 'chart-card';
    var isBull = r.type === 'bullish';
    var isTrig = r.signal === 'triggered';
    var badgeCls = (isBull ? 'bull' : 'bear') + '-' + (isTrig ? 'triggered' : 'forming');
    var badgeLabel = (isBull ? 'BULL ' : 'BEAR ') + (isTrig ? 'TRIGGERED' : 'FORMING');
    var rsStr = (r.rs >= 0 ? '+' : '') + r.rs.toFixed(1);
    var tvUrl = 'https://www.tradingview.com/chart/?symbol=' + r.ticker;
    var tightBadge = r.tight_flag ? '<span class="badge tight">TIGHT</span>' : '';
    var crossStr = r.cross_bars_ago + 'd ago';
    card.innerHTML =
      '<div class="chart-card-header">' +
        '<a href="' + tvUrl + '" target="_blank" class="ticker">' + r.ticker +
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></a>' +
        '<div class="meta">' +
          '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
          tightBadge +
          '<span class="info">RS ' + rsStr + '</span>' +
          '<span class="info">Cross ' + crossStr + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="chart-card-body" id="chart-' + r.ticker + '"></div>';
    grid.appendChild(card);

    var container = card.querySelector('.chart-card-body');
    var ch = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 220,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#1e252e' }, horzLines: { color: '#1e252e' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { visible: false },
      leftPriceScale: { visible: false },
      timeScale: { visible: false },
      handleScroll: false, handleScale: false,
    });

    var candles = ch.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderUpColor: '#3fb950', borderDownColor: '#f85149',
      wickUpColor: '#3fb950', wickDownColor: '#f85149',
      priceLineVisible: false, lastValueVisible: false,
    });
    candles.setData(ohlcv.map(function(d) {
      return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
    }));

    // Price line: flag_high for bullish, flag_low for bearish
    var priceLevel = isBull ? r.flag_high : r.flag_low;
    if (priceLevel) {
      candles.createPriceLine({
        price: priceLevel,
        color: isBull ? '#58a6ff' : '#f85149',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false,
      });
    }

    var vol = ch.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      priceLineVisible: false, lastValueVisible: false,
    });
    ch.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    vol.setData(ohlcv.map(function(d) {
      return { time: d.time, value: d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,0.2)' : 'rgba(248,81,73,0.2)' };
    }));

    var closes = ohlcv.map(function(d) { return d.close; });
    var ema8vals = calcEMA(closes, 8);
    var ema8s = ch.addLineSeries({
      color: '#39d2e0', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema8s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema8vals[i].toFixed(2)) };
    }));

    var ema21vals = calcEMA(closes, 21);
    var ema21s = ch.addLineSeries({
      color: '#FF9800', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema21s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema21vals[i].toFixed(2)) };
    }));

    ch.timeScale().fitContent();

    new ResizeObserver(function() {
      ch.applyOptions({ width: container.clientWidth });
    }).observe(container);
  });
}

buildCards();
</script>
</body>
</html>"""


def build_html(results, chart_data, elapsed, scanned):
    def _convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    html = _HTML_TEMPLATE
    html = html.replace("__RESULTS_JSON__", json.dumps(results, default=_convert))
    html = html.replace("__CHART_DATA_JSON__", json.dumps(chart_data, default=_convert))
    html = html.replace("__ELAPSED__", f"{elapsed:.1f}")
    html = html.replace("__SCANNED__", str(scanned))
    html = html.replace("__COUNT__", str(len(results)))
    return html


def run_web(results, chart_data, elapsed, scanned):
    from flask import Flask

    app = Flask(__name__)
    page = build_html(results, chart_data, elapsed, scanned)

    @app.route("/")
    def index():
        return page

    print("[*] Starting web UI at http://127.0.0.1:5053")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5053")).start()
    app.run(port=5053, debug=False)


# ---------------------------------------------------------------------------
# Standard module interface for combined screener
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None):
    """Self-contained scan: downloads data (unless provided), scans, returns results.
    Returns: (results: list[dict], chart_data: dict, scanned: int)
    """
    if data is None:
        data = download_data(tickers)

    spy_ohlcv = extract_ohlcv(data, "SPY")
    if spy_ohlcv is None:
        return [], {}, 0
    spy_closes = spy_ohlcv[3]

    results = []
    chart_data = {}
    scanned = 0
    for ticker in tickers:
        ohlcv = extract_ohlcv(data, ticker)
        if ohlcv is None:
            continue
        opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if side in ("bull", "both"):
            hit = scan_ticker_bullish(closes, highs, lows, volumes, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                chart_data[ticker] = extract_chart_data(data, ticker)

        if side in ("bear", "both"):
            hit = scan_ticker_bearish(closes, highs, lows, volumes, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if ticker not in chart_data:
                    chart_data[ticker] = extract_chart_data(data, ticker)

    return results, chart_data, scanned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA Flip Stock Screener")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI (terminal output only)")
    parser.add_argument("--side", choices=["bull", "bear", "both"], default="both",
                        help="Scan for bullish, bearish, or both setups (default: both)")
    args = parser.parse_args()

    t0 = time.time()

    # --- Build universe ---
    if args.ticker:
        tickers = [args.ticker.upper().replace(".", "-")]
        print(f"[*] Single-ticker mode: {tickers[0]}")
    else:
        valid = list(UNIVERSES.keys()) + ["all"]
        choice = input(f"Which index? ({'/'.join(valid)}): ").strip().lower()
        if choice not in valid:
            print(f"[!] Invalid choice '{choice}'. Pick from: {', '.join(valid)}")
            sys.exit(1)

        if choice == "all":
            print("[*] Fetching all major indices (deduplicated)...")
            tickers = fetch_all_universes()
        else:
            universe_name = UNIVERSES[choice][0]
            print(f"[*] Fetching {universe_name} universe from Wikipedia...")
            tickers = FETCH_FUNCTIONS[choice]()
            print(f"    {len(tickers)} tickers loaded")

    # --- Download data ---
    print("[*] Downloading price data...")
    data = download_data(tickers)

    # --- Extract SPY data ---
    spy_ohlcv = extract_ohlcv(data, "SPY")
    if spy_ohlcv is None:
        print("[!] Could not load SPY data for relative strength")
        sys.exit(1)
    spy_closes = spy_ohlcv[3]

    # --- Scan ---
    side_label = {"bull": "bullish", "bear": "bearish", "both": "bull+bear"}[args.side]
    print(f"[*] Scanning for EMA Flip setups ({side_label})...")
    results = []
    chart_data = {}
    scanned = 0
    for ticker in tickers:
        ohlcv = extract_ohlcv(data, ticker)
        if ohlcv is None:
            continue
        opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if args.side in ("bull", "both"):
            hit = scan_ticker_bullish(closes, highs, lows, volumes, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if not args.no_web:
                    chart_data[ticker] = extract_chart_data(data, ticker)

        if args.side in ("bear", "both"):
            hit = scan_ticker_bearish(closes, highs, lows, volumes, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if not args.no_web and ticker not in chart_data:
                    chart_data[ticker] = extract_chart_data(data, ticker)

    elapsed = time.time() - t0
    print(f"    Scanned {scanned} tickers in {elapsed:.1f}s")

    # --- Output ---
    print_results(results)

    if not args.no_web and results:
        run_web(results, chart_data, elapsed, scanned)


if __name__ == "__main__":
    main()
