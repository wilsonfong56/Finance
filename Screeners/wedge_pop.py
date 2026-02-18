#!/usr/bin/env python3
"""
Wedge Pop Stock Screener
========================
Identifies Oliver Kell's "Wedge Pop" setup across the Russell 2000 universe.

Bullish (Wedge Pop):
  1. Reversal Extension — price stretched below EMAs
  2. Snapback & Resistance — touches EMA then pulls back
  3. Volatility Contraction (Setting Up) — higher low, shrinking range, ATR squeeze
  4. Wedge Pop Trigger (Triggered) — breakout above swing high on volume

Bearish (Wedge Drop):
  1. Reversal Extension — price stretched above EMAs (overbought)
  2. Snapback & Resistance — drops to EMA then bounces back up
  3. Volatility Contraction (Setting Up) — lower high, shrinking range, ATR squeeze
  4. Wedge Drop Trigger (Triggered) — breakdown below swing low

Usage:
  python wedge_pop.py              # prompts: spy/qqq/iwm/dia
  python wedge_pop.py --ticker AAPL  # single-stock debug mode
  python wedge_pop.py --side bear    # bearish only
"""

import argparse
import io
import json
import math
import sys
import threading
import time
import webbrowser

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SCREENER_NAME = "Wedge Pop"
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
# Universe: Russell 2000 from iShares IWM
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
    """Scrape S&P 500 tickers from Wikipedia."""
    tickers = _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)
    if tickers:
        return tickers
    print("[!] Could not find ticker column in S&P 500 Wikipedia table")
    sys.exit(1)


def fetch_nasdaq100():
    """Scrape Nasdaq 100 tickers from Wikipedia."""
    tickers = _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)
    if tickers:
        return tickers
    print("[!] Could not find ticker column in Nasdaq 100 Wikipedia table")
    sys.exit(1)


def fetch_dowjones():
    """Scrape Dow Jones 30 tickers from Wikipedia."""
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
    """Batch-download OHLCV data for all tickers + SPY."""
    all_tickers = list(set(tickers + ["SPY"]))
    period = f"{months}mo"
    chunk_size = 100
    frames = []

    total = len(all_tickers)
    for i in range(0, total, chunk_size):
        chunk = all_tickers[i: i + chunk_size]
        pct = min((i + chunk_size) / total * 100, 100)
        print(f"\r  Downloading... {pct:5.1f}%  ({min(i + chunk_size, total)}/{total} tickers)", end="", flush=True)
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
    """Pull OHLCV arrays for a single ticker from the batch DataFrame."""
    try:
        sub = data[ticker]
        o = sub["Open"].values.flatten().astype(float)
        h = sub["High"].values.flatten().astype(float)
        lo = sub["Low"].values.flatten().astype(float)
        c = sub["Close"].values.flatten().astype(float)
        v = sub["Volume"].values.flatten().astype(float)
    except (KeyError, TypeError):
        return None

    # drop leading NaNs
    valid = ~np.isnan(c)
    if valid.sum() < 60:
        return None
    first = np.argmax(valid)
    return o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=120):
    """Extract last N bars of OHLCV with dates for charting."""
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

    # Drop leading NaNs
    valid = ~np.isnan(c)
    if valid.sum() == 0:
        return []
    first = np.argmax(valid)
    dates, o, h, lo, c, v = dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]

    # Take last N bars
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
# Phase detection — Bullish (Wedge Pop)
# ---------------------------------------------------------------------------

def scan_ticker_bullish(closes, highs, lows, spy_closes):
    """
    Run the 4-phase bullish wedge-pop scan on a single ticker.

    Returns dict with keys: phase, price, breakout_level, pct_to_breakout, rs,
                             type, signal
            or None if no setup found.
    """
    n = len(closes)
    if n < 60:
        return None

    ema10 = _calc_ema(closes, 10)
    ema20 = _calc_ema(closes, 20)

    # rolling 20-day std dev of closes
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(closes[i - 19: i + 1], ddof=0)

    # ATR 5 and 20
    atr5 = _calc_atr(highs, lows, closes, 5)
    atr20 = _calc_atr(highs, lows, closes, 20)

    # --- Phase 1: Reversal Extension (scan last 60 bars) ---
    search_start = max(0, n - 60)
    phase1_idx = None
    phase1_low = None

    for i in range(search_start, n):
        if np.isnan(std20[i]):
            continue
        dist_below = ema20[i] - closes[i]
        if dist_below <= 0:
            continue

        cond_a = dist_below >= 2 * std20[i] if std20[i] > 0 else False

        cond_b = False
        if i >= 19:
            distances = [max(0, ema20[j] - closes[j]) for j in range(i - 19, i + 1)]
            cond_b = dist_below >= max(distances) - 1e-9

        if cond_a or cond_b:
            phase1_idx = i
            phase1_low = lows[i]

    if phase1_idx is None:
        return None

    if n - 1 - phase1_idx < 3:
        return None

    # --- Phase 2: Snapback & Resistance ---
    phase2_idx = None
    for i in range(phase1_idx + 1, n - 1):
        touched = closes[i] >= ema10[i] or closes[i] >= ema20[i]
        if touched:
            pullback = closes[i + 1] < ema10[i + 1] or closes[i + 1] < ema20[i + 1]
            if pullback:
                phase2_idx = i + 1
                break

    if phase2_idx is None:
        return None

    # --- Phase 3: Volatility Contraction (Mini Base) ---
    phase3_idx = None
    breakout_level = None

    check_start = max(phase2_idx, n - 5)
    for i in range(check_start, n):
        if i < 3:
            continue

        recent_low = min(lows[phase2_idx: i + 1])
        if recent_low <= phase1_low:
            continue

        range_now = highs[i] - lows[i]
        range_3ago = highs[i - 3] - lows[i - 3]
        if range_now >= range_3ago:
            continue

        if np.isnan(atr5[i]) or np.isnan(atr20[i]):
            continue
        if atr5[i] >= atr20[i]:
            continue

        swing_high = max(highs[phase2_idx: i + 1])

        phase3_idx = i
        breakout_level = swing_high

    if phase3_idx is None:
        return None

    # --- Relative Strength vs SPY ---
    rs = _calc_rs(closes, spy_closes)

    price = closes[-1]

    # --- Phase 4: Wedge Pop Trigger ---
    for i in range(max(phase3_idx, n - 2), n):
        if closes[i] > breakout_level and closes[i] > ema10[i] and closes[i] > ema20[i]:
            return {
                "phase": 4,
                "signal": "triggered",
                "type": "bullish",
                "price": price,
                "breakout_level": breakout_level,
                "pct_to_breakout": 0.0,
                "rs": rs,
            }

    if price > breakout_level:
        return None

    pct_to_bo = (breakout_level - price) / price * 100 if price > 0 else 0
    return {
        "phase": 3,
        "signal": "forming",
        "type": "bullish",
        "price": price,
        "breakout_level": breakout_level,
        "pct_to_breakout": pct_to_bo,
        "rs": rs,
    }


# ---------------------------------------------------------------------------
# Phase detection — Bearish (Wedge Drop)
# ---------------------------------------------------------------------------

def scan_ticker_bearish(closes, highs, lows, spy_closes):
    """
    Run the 4-phase bearish wedge-drop scan on a single ticker.
    Mirror of the bullish pattern: overbought -> snapback down -> lower-high
    contraction -> breakdown below swing low.

    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None

    ema10 = _calc_ema(closes, 10)
    ema20 = _calc_ema(closes, 20)

    # rolling 20-day std dev of closes
    std20 = np.full(n, np.nan)
    for i in range(19, n):
        std20[i] = np.std(closes[i - 19: i + 1], ddof=0)

    # ATR 5 and 20
    atr5 = _calc_atr(highs, lows, closes, 5)
    atr20 = _calc_atr(highs, lows, closes, 20)

    # --- Phase 1: Reversal Extension — price stretched ABOVE EMAs (overbought) ---
    search_start = max(0, n - 60)
    phase1_idx = None
    phase1_high = None

    for i in range(search_start, n):
        if np.isnan(std20[i]):
            continue
        dist_above = closes[i] - ema20[i]
        if dist_above <= 0:
            continue

        cond_a = dist_above >= 2 * std20[i] if std20[i] > 0 else False

        cond_b = False
        if i >= 19:
            distances = [max(0, closes[j] - ema20[j]) for j in range(i - 19, i + 1)]
            cond_b = dist_above >= max(distances) - 1e-9

        if cond_a or cond_b:
            phase1_idx = i
            phase1_high = highs[i]

    if phase1_idx is None:
        return None

    if n - 1 - phase1_idx < 3:
        return None

    # --- Phase 2: Snapback down to EMA, then bounces back up ---
    phase2_idx = None
    for i in range(phase1_idx + 1, n - 1):
        touched = closes[i] <= ema10[i] or closes[i] <= ema20[i]
        if touched:
            bounce = closes[i + 1] > ema10[i + 1] or closes[i + 1] > ema20[i + 1]
            if bounce:
                phase2_idx = i + 1
                break

    if phase2_idx is None:
        return None

    # --- Phase 3: Volatility Contraction — lower high, range shrinking ---
    phase3_idx = None
    breakdown_level = None

    check_start = max(phase2_idx, n - 5)
    for i in range(check_start, n):
        if i < 3:
            continue

        # Lower high: recent high since phase2 must be below phase1_high
        recent_high = max(highs[phase2_idx: i + 1])
        if recent_high >= phase1_high:
            continue

        # Range shrinking
        range_now = highs[i] - lows[i]
        range_3ago = highs[i - 3] - lows[i - 3]
        if range_now >= range_3ago:
            continue

        # Volatility squeeze
        if np.isnan(atr5[i]) or np.isnan(atr20[i]):
            continue
        if atr5[i] >= atr20[i]:
            continue

        swing_low = min(lows[phase2_idx: i + 1])

        phase3_idx = i
        breakdown_level = swing_low

    if phase3_idx is None:
        return None

    # --- Relative Strength vs SPY ---
    rs = _calc_rs(closes, spy_closes)

    price = closes[-1]

    # --- Phase 4: Wedge Drop Trigger — breakdown below swing low ---
    for i in range(max(phase3_idx, n - 2), n):
        if closes[i] < breakdown_level and closes[i] < ema10[i] and closes[i] < ema20[i]:
            return {
                "phase": 4,
                "signal": "triggered",
                "type": "bearish",
                "price": price,
                "breakout_level": breakdown_level,
                "pct_to_breakout": 0.0,
                "rs": rs,
            }

    if price < breakdown_level:
        return None

    pct_to_bd = (price - breakdown_level) / price * 100 if price > 0 else 0
    return {
        "phase": 3,
        "signal": "forming",
        "type": "bearish",
        "price": price,
        "breakout_level": breakdown_level,
        "pct_to_breakout": pct_to_bd,
        "rs": rs,
    }


def _calc_rs(closes, spy_closes):
    """3-month relative strength: stock return minus SPY return (pct points)."""
    period = 63  # ~3 months of trading days
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    """Print a formatted table of results."""
    if not results:
        print("\nNo Wedge Pop setups found today.")
        return

    # sort: triggered first, then forming; bullish before bearish; by RS
    results.sort(key=lambda r: (
        0 if r["type"] == "bullish" else 1,
        0 if r["signal"] == "triggered" else 1,
        -r["rs"],
    ))

    hdr = f"{'Ticker':<8} {'Type':<10} {'Signal':<12} {'Price':>8} {'Level':>10} {'% to BO':>8} {'RS vs SPY':>10}  TradingView"
    print(f"\n{'=' * len(hdr)}")
    print(f" Wedge Pop Screener — {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sig_label = "TRIGGERED" if r["signal"] == "triggered" else "Forming"
        tv_link = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        pct_str = f"{r['pct_to_breakout']:+.1f}%" if r["signal"] == "forming" else "  —"
        print(f"{r['ticker']:<8} {r['type'].upper():<10} {sig_label:<12} {r['price']:>8.2f} "
              f"{r['breakout_level']:>10.2f} {pct_str:>8} {r['rs']:>+10.1f}  {tv_link}")

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
<title>Wedge Pop Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28; }
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
  .info { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Wedge Pop Screener</h1>
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
    var pctStr = r.signal === 'forming' ? r.pct_to_breakout.toFixed(1) + '% to level' : '';
    card.innerHTML =
      '<div class="chart-card-header">' +
        '<a href="' + tvUrl + '" target="_blank" class="ticker">' + r.ticker +
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></a>' +
        '<div class="meta">' +
          '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
          '<span class="info">RS ' + rsStr + '</span>' +
          (pctStr ? '<span class="info">' + pctStr + '</span>' : '') +
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

    candles.createPriceLine({
      price: r.breakout_level,
      color: isBull ? '#58a6ff' : '#f85149',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: false,
    });

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
    var ema10vals = calcEMA(closes, 10);
    var ema10s = ch.addLineSeries({
      color: '#4CAF50', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema10s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema10vals[i].toFixed(2)) };
    }));

    var ema20vals = calcEMA(closes, 20);
    var ema20s = ch.addLineSeries({
      color: '#F44336', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema20s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema20vals[i].toFixed(2)) };
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
    """Build the HTML page with embedded chart data."""
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
    """Start a Flask server and open the browser."""
    from flask import Flask

    app = Flask(__name__)
    page = build_html(results, chart_data, elapsed, scanned)

    @app.route("/")
    def index():
        return page

    print("[*] Starting web UI at http://127.0.0.1:5050")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5050")).start()
    app.run(port=5050, debug=False)


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
            hit = scan_ticker_bullish(closes, highs, lows, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                chart_data[ticker] = extract_chart_data(data, ticker)

        if side in ("bear", "both") and ticker not in chart_data:
            hit = scan_ticker_bearish(closes, highs, lows, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                chart_data[ticker] = extract_chart_data(data, ticker)
        elif side in ("bear", "both") and ticker in chart_data:
            # Already have bullish hit; still check bearish (rare for same ticker)
            hit = scan_ticker_bearish(closes, highs, lows, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)

    return results, chart_data, scanned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wedge Pop Stock Screener")
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
    print(f"[*] Scanning for Wedge Pop setups ({side_label})...")
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
            hit = scan_ticker_bullish(closes, highs, lows, spy_closes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if not args.no_web:
                    chart_data[ticker] = extract_chart_data(data, ticker)

        if args.side in ("bear", "both"):
            hit = scan_ticker_bearish(closes, highs, lows, spy_closes)
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
