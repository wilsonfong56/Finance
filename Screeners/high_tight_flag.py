#!/usr/bin/env python3
"""
High Tight Flag Stock Screener
================================
Identifies "High Tight Flag" setups: stocks that made a strong move up
(15%+ within 45 bars after breaking a swing high), then consolidated tightly
(giving back <30% of the move).

Bearish counterpart ("Low Tight Flag"): stocks that made a strong move down
(15%+ within 45 bars after breaking a swing low), then consolidated tightly
(bouncing back <30% of the move).

Signals:
  - Forming   — stock is currently consolidating in the flag
  - Triggered — stock just broke out/down from the flag (last 1-2 bars)

Usage:
  python high_tight_flag.py                  # interactive index selection
  python high_tight_flag.py --ticker AAPL    # single-stock debug mode
  python high_tight_flag.py --side bear      # bearish only
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


SCREENER_NAME = "High Tight Flag"
DATA_PARAMS = {"period": "6mo", "interval": "1d"}


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def _calc_ema(closes, period):
    """EMA matching the project's existing pattern."""
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return np.array(ema)


def _calc_rs(closes, spy_closes):
    """3-month relative strength: stock return minus SPY return (pct points)."""
    period = 63
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


def find_swing_highs(highs, pivot_n=10):
    """Find pivot highs: bars where high is the highest in 2*pivot_n+1 window."""
    n = len(highs)
    swings = []
    for i in range(pivot_n, n - pivot_n):
        window = highs[i - pivot_n: i + pivot_n + 1]
        if highs[i] >= np.max(window):
            swings.append((i, float(highs[i])))
    return swings


def find_swing_lows(lows, pivot_n=10):
    """Find pivot lows: bars where low is the lowest in 2*pivot_n+1 window."""
    n = len(lows)
    swings = []
    for i in range(pivot_n, n - pivot_n):
        window = lows[i - pivot_n: i + pivot_n + 1]
        if lows[i] <= np.min(window):
            swings.append((i, float(lows[i])))
    return swings


# ---------------------------------------------------------------------------
# Universe scrapers
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
        return []

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
    return []


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
        return []

    lines = resp.text.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,Name,"):
            header_idx = i
            break
    if header_idx is None:
        print("[!] Could not find header row in IWM holdings CSV")
        return []

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    equity = df[df["Asset Class"] == "Equity"]
    tickers = equity["Ticker"].astype(str).tolist()
    tickers = [t.strip().replace(".", "-") for t in tickers if t.strip()]
    return tickers


def fetch_sp500():
    return _scrape_wiki_table(UNIVERSES["spy"][1], ["symbol", "ticker"], min_rows=400)


def fetch_nasdaq100():
    return _scrape_wiki_table(UNIVERSES["qqq"][1], ["ticker", "symbol"], min_rows=90)


def fetch_dowjones():
    return _scrape_wiki_table(UNIVERSES["dia"][1], ["symbol", "ticker"], min_rows=25)


FETCH_FUNCTIONS = {
    "iwm": fetch_russell2000,
    "spy": fetch_sp500,
    "qqq": fetch_nasdaq100,
    "dia": fetch_dowjones,
}


def fetch_all_universes():
    """Fetch and deduplicate tickers from all major indices."""
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
# HTF scan — Bullish (High Tight Flag)
# ---------------------------------------------------------------------------

def scan_htf_bullish(closes, highs, lows, spy_closes, volumes=None, vol_filter=False):
    """
    Scan for bullish High Tight Flag pattern.

    1. Find a peak (pole top) in the last ~70 bars
    2. Measure the explosive move from the lowest low within 45 bars before it
    3. Require 15%+ gain and at least one prior swing high broken
    4. Check that post-peak consolidation gives back <30% of the move
    5. (Optional) Pole volume >= 1.5x baseline, flag volume < pole volume
    6. Label as "forming" or "triggered" (close above pole high in last 2 bars)

    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None

    swing_highs = find_swing_highs(highs, pivot_n=10)

    best = None
    search_start = max(45, n - 70)

    for peak_idx in range(search_start, n):
        peak_high = highs[peak_idx]

        lookback_start = max(0, peak_idx - 45)
        if lookback_start >= peak_idx:
            continue
        base_idx = lookback_start + int(np.argmin(lows[lookback_start:peak_idx]))
        base_low = lows[base_idx]
        if base_low <= 0:
            continue

        move_pct = (peak_high - base_low) / base_low * 100
        if move_pct < 15:
            continue

        broke_swing = False
        for sh_idx, sh_val in swing_highs:
            if sh_idx <= base_idx and sh_val < peak_high and base_idx - sh_idx < 40:
                broke_swing = True
                break
            if base_idx < sh_idx < peak_idx and sh_val < peak_high:
                broke_swing = True
                break
        if not broke_swing:
            continue

        if peak_idx >= n - 3:
            continue

        breakout_bar = None
        valid_flag = True
        for i in range(peak_idx + 1, n):
            if closes[i] > peak_high:
                if i >= n - 2:
                    breakout_bar = i
                else:
                    valid_flag = False
                break

        if not valid_flag:
            continue

        consol_end = breakout_bar if breakout_bar else n
        consol_bars = consol_end - peak_idx - 1
        if consol_bars < 3:
            continue

        flag_low = float(np.min(lows[peak_idx + 1:consol_end]))
        move_size = peak_high - base_low
        giveback = peak_high - flag_low
        giveback_pct = giveback / move_size

        if giveback_pct >= 0.30:
            continue

        if vol_filter and volumes is not None:
            baseline_start = max(0, base_idx - 20)
            baseline_vol = float(np.mean(volumes[baseline_start:base_idx])) if baseline_start < base_idx else 0
            pole_vol = float(np.mean(volumes[base_idx:peak_idx + 1]))
            flag_vol = float(np.mean(volumes[peak_idx + 1:consol_end]))
            if baseline_vol > 0 and pole_vol < 1.5 * baseline_vol:
                continue
            if pole_vol > 0 and flag_vol >= pole_vol:
                continue

        signal = "triggered" if breakout_bar else "forming"
        rs = _calc_rs(closes, spy_closes)

        result = {
            "signal": signal,
            "type": "bullish",
            "price": float(closes[-1]),
            "move_pct": float(move_pct),
            "giveback_pct": float(giveback_pct * 100),
            "pole_low": float(base_low),
            "pole_high": float(peak_high),
            "flag_low": float(flag_low),
            "consol_bars": int(consol_bars),
            "rs": float(rs),
        }

        if best is None or move_pct > best["move_pct"]:
            best = result

    return best


# ---------------------------------------------------------------------------
# HTF scan — Bearish (Low Tight Flag)
# ---------------------------------------------------------------------------

def scan_htf_bearish(closes, highs, lows, spy_closes, volumes=None, vol_filter=False):
    """
    Scan for bearish Low Tight Flag pattern.

    Mirror of the bullish HTF:
    1. Find a trough (pole bottom) in the last ~70 bars
    2. Measure the explosive move down from the highest high within 45 bars before it
    3. Require 15%+ decline and at least one prior swing low broken
    4. Check that post-trough consolidation bounces back <30% of the move
    5. (Optional) Pole volume >= 1.5x baseline, flag volume < pole volume
    6. Label as "forming" or "triggered" (close below pole low in last 2 bars)

    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None

    swing_lows = find_swing_lows(lows, pivot_n=10)

    best = None
    search_start = max(45, n - 70)

    for trough_idx in range(search_start, n):
        trough_low = lows[trough_idx]

        # --- Pole top: highest high in 45 bars before trough ---
        lookback_start = max(0, trough_idx - 45)
        if lookback_start >= trough_idx:
            continue
        top_idx = lookback_start + int(np.argmax(highs[lookback_start:trough_idx]))
        top_high = highs[top_idx]
        if top_high <= 0:
            continue

        # --- Move size check (15%+ decline) ---
        move_pct = (top_high - trough_low) / top_high * 100
        if move_pct < 15:
            continue

        # --- Swing low break: move must have broken a prior pivot low ---
        broke_swing = False
        for sl_idx, sl_val in swing_lows:
            if sl_idx <= top_idx and sl_val > trough_low and top_idx - sl_idx < 40:
                broke_swing = True
                break
            if top_idx < sl_idx < trough_idx and sl_val > trough_low:
                broke_swing = True
                break
        if not broke_swing:
            continue

        # --- Consolidation after trough (need at least 3 bars) ---
        if trough_idx >= n - 3:
            continue

        breakdown_bar = None
        valid_flag = True
        for i in range(trough_idx + 1, n):
            if closes[i] < trough_low:
                if i >= n - 2:  # breakdown in last 2 bars = triggered
                    breakdown_bar = i
                else:
                    valid_flag = False  # broke down too long ago
                break

        if not valid_flag:
            continue

        consol_end = breakdown_bar if breakdown_bar else n
        consol_bars = consol_end - trough_idx - 1
        if consol_bars < 3:
            continue

        # --- Flag tightness: bounce < 30% of the move ---
        flag_high = float(np.max(highs[trough_idx + 1:consol_end]))
        move_size = top_high - trough_low
        bounce = flag_high - trough_low
        bounce_pct = bounce / move_size

        if bounce_pct >= 0.30:
            continue

        # --- Volume filter (optional) ---
        if vol_filter and volumes is not None:
            baseline_start = max(0, top_idx - 20)
            baseline_vol = float(np.mean(volumes[baseline_start:top_idx])) if baseline_start < top_idx else 0
            pole_vol = float(np.mean(volumes[top_idx:trough_idx + 1]))
            flag_vol = float(np.mean(volumes[trough_idx + 1:consol_end]))
            if baseline_vol > 0 and pole_vol < 1.5 * baseline_vol:
                continue
            if pole_vol > 0 and flag_vol >= pole_vol:
                continue

        signal = "triggered" if breakdown_bar else "forming"
        rs = _calc_rs(closes, spy_closes)

        result = {
            "signal": signal,
            "type": "bearish",
            "price": float(closes[-1]),
            "move_pct": float(move_pct),
            "giveback_pct": float(bounce_pct * 100),
            "pole_high": float(top_high),
            "pole_low": float(trough_low),
            "flag_high": float(flag_high),
            "consol_bars": int(consol_bars),
            "rs": float(rs),
        }

        if best is None or move_pct > best["move_pct"]:
            best = result

    return best


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    """Print a formatted table of results."""
    if not results:
        print("\nNo High Tight Flag setups found today.")
        return

    results.sort(key=lambda r: (
        0 if r["type"] == "bullish" else 1,
        0 if r["signal"] == "triggered" else 1,
        -r["move_pct"],
    ))

    hdr = (f"{'Ticker':<8} {'Type':<10} {'Signal':<12} {'Price':>8} {'Level':>10} "
           f"{'Move%':>8} {'Giveback':>9} {'Consol':>7} {'RS vs SPY':>10}  TradingView")
    print(f"\n{'=' * len(hdr)}")
    print(f" High Tight Flag Screener — {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        label = "TRIGGERED" if r["signal"] == "triggered" else "Forming"
        tv = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        level = r["pole_high"] if r["type"] == "bullish" else r["pole_low"]
        print(f"{r['ticker']:<8} {r['type'].upper():<10} {label:<12} {r['price']:>8.2f} {level:>10.2f} "
              f"{r['move_pct']:>+7.1f}% {r['giveback_pct']:>8.1f}% {r['consol_bars']:>5}d "
              f"{r['rs']:>+10.1f}  {tv}")

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
<title>High Tight Flag Screener</title>
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
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 12px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); }
  .chart-card-header .ticker { font-size: 15px; font-weight: 700; color: var(--accent);
    text-decoration: none; }
  .chart-card-header .ticker:hover { text-decoration: underline; }
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px; font-size: 11px; }
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
  <h1>High Tight Flag Screener</h1>
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
    var moveSign = isBull ? '+' : '-';
    card.innerHTML =
      '<div class="chart-card-header">' +
        '<a href="' + tvUrl + '" target="_blank" class="ticker">' + r.ticker +
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></a>' +
        '<div class="meta">' +
          '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
          '<span class="info">' + moveSign + r.move_pct.toFixed(0) + '% move</span>' +
          '<span class="info">' + r.giveback_pct.toFixed(0) + '% giveback</span>' +
          '<span class="info">RS ' + rsStr + '</span>' +
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

    // Breakout/breakdown level
    var level = isBull ? r.pole_high : r.pole_low;
    candles.createPriceLine({
      price: level,
      color: isBull ? '#58a6ff' : '#f85149',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: false,
    });

    // Flag boundary
    var flagLevel = isBull ? r.flag_low : r.flag_high;
    candles.createPriceLine({
      price: flagLevel,
      color: '#d29922',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: false,
    });

    // Volume
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

    // EMA 10 & 20
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

    print("[*] Starting web UI at http://127.0.0.1:5051")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5051")).start()
    app.run(port=5051, debug=False)


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

        if closes[-1] < 5:
            scanned += 1
            continue

        scanned += 1

        if side in ("bull", "both"):
            hit = scan_htf_bullish(closes, highs, lows, spy_closes, volumes=volumes)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                chart_data[ticker] = extract_chart_data(data, ticker)

        if side in ("bear", "both"):
            hit = scan_htf_bearish(closes, highs, lows, spy_closes, volumes=volumes)
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
    parser = argparse.ArgumentParser(description="High Tight Flag Stock Screener")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI (terminal output only)")
    parser.add_argument("--volume-filter", action="store_true",
                        help="Require volume expansion on pole and contraction in flag")
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
    print(f"[*] Scanning for High Tight Flag setups ({side_label})...")
    results = []
    chart_data = {}
    scanned = 0
    for ticker in tickers:
        ohlcv = extract_ohlcv(data, ticker)
        if ohlcv is None:
            continue
        opens, highs, lows, closes, volumes = ohlcv

        # Skip penny stocks
        if closes[-1] < 5:
            scanned += 1
            continue

        scanned += 1

        if args.side in ("bull", "both"):
            hit = scan_htf_bullish(closes, highs, lows, spy_closes,
                                   volumes=volumes, vol_filter=args.volume_filter)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if not args.no_web:
                    chart_data[ticker] = extract_chart_data(data, ticker)

        if args.side in ("bear", "both"):
            hit = scan_htf_bearish(closes, highs, lows, spy_closes,
                                   volumes=volumes, vol_filter=args.volume_filter)
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
