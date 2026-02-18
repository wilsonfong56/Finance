#!/usr/bin/env python3
"""
Base Breakout Stock Screener
==============================
Identifies stocks breaking out of (or breaking down from) long-term bases
using weekly data over 5 years.

Bullish Base Breakout:
  - Stock had a prior high from >=26 weeks ago
  - Declined >=25% from that high
  - Now approaching (forming) or breaking above (triggered) the prior high

Bearish Base Breakdown:
  - Stock peaked in the last 13-52 weeks
  - Established a consolidation floor that held as support
  - Now approaching (forming) or breaking below (triggered) the floor

Usage:
  python base_breakout.py                  # interactive index selection
  python base_breakout.py --ticker AAPL    # single-stock debug mode
"""

import argparse
import io
import json
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SCREENER_NAME = "Base Breakout"
DATA_PARAMS = {"period": "5y", "interval": "1wk"}


# ---------------------------------------------------------------------------
# Technical helpers
# ---------------------------------------------------------------------------

def _calc_ema(closes, period):
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return np.array(ema)


def _calc_rs(closes, spy_closes):
    """Relative strength vs SPY over last 26 weeks (~6 months)."""
    period = 26
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


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
# Data download (weekly, 5 years)
# ---------------------------------------------------------------------------

def download_data(tickers):
    """Batch-download weekly OHLCV data for all tickers + SPY (5 years)."""
    all_tickers = list(set(tickers + ["SPY"]))
    chunk_size = 100
    frames = []

    total = len(all_tickers)
    for i in range(0, total, chunk_size):
        chunk = all_tickers[i: i + chunk_size]
        pct = min((i + chunk_size) / total * 100, 100)
        print(f"\r  Downloading... {pct:5.1f}%  ({min(i + chunk_size, total)}/{total} tickers)",
              end="", flush=True)
        try:
            df = yf.download(chunk, period="5y", interval="1wk",
                             group_by="ticker", progress=False, threads=True)
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
    if valid.sum() < 52:
        return None
    first = np.argmax(valid)
    return o[first:], h[first:], lo[first:], c[first:], v[first:]


def extract_chart_data(data, ticker, bars=150):
    """Extract last N bars of weekly OHLCV with dates for charting."""
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
# Base scan
# ---------------------------------------------------------------------------

def scan_base_bullish(closes, highs, lows, spy_closes):
    """
    Bullish base breakout:
      - Ceiling = highest high from >=26 weeks ago
      - Post-ceiling decline >= 25%
      - Price now approaching or breaking above ceiling
    """
    n = len(closes)
    if n < 52 or n <= 26:
        return None

    # Ceiling: highest high excluding the last 26 weeks
    ceiling_search = highs[:n - 26]
    ceiling_idx = int(np.argmax(ceiling_search))
    ceiling = float(highs[ceiling_idx])

    # Post-ceiling low
    low_after_idx = ceiling_idx + int(np.argmin(lows[ceiling_idx:]))
    base_low = float(lows[low_after_idx])

    # Decline check
    if ceiling <= 0:
        return None
    decline_pct = (ceiling - base_low) / ceiling * 100
    if decline_pct < 25:
        return None

    # Base duration
    base_weeks = n - 1 - ceiling_idx
    if base_weeks < 26:
        return None

    current = float(closes[-1])
    pct_below = (ceiling - current) / ceiling * 100

    # --- Triggered: first close above ceiling in last 3 weeks ---
    if pct_below <= 0:
        # Check no prior close exceeded the ceiling (fresh breakout)
        prior_above = False
        for i in range(ceiling_idx + 1, max(0, n - 3)):
            if closes[i] > ceiling:
                prior_above = True
                break
        if prior_above:
            return None  # breakout already happened long ago

        # Confirm breakout in last 3 weeks
        recent_above = False
        for i in range(max(0, n - 3), n):
            if closes[i] > ceiling:
                recent_above = True
                break
        if not recent_above:
            return None

        return {
            "signal": "triggered",
            "type": "bullish",
            "price": current,
            "level": ceiling,
            "base_low": base_low,
            "decline_pct": decline_pct,
            "base_weeks": base_weeks,
            "pct_to_level": 0.0,
            "rs": _calc_rs(closes, spy_closes),
        }

    # --- Forming: within 15% of ceiling, trending up ---
    if pct_below <= 15:
        # Must be trending upward (close > close from 13 weeks ago)
        if n >= 13 and closes[-1] <= closes[-13]:
            return None

        return {
            "signal": "forming",
            "type": "bullish",
            "price": current,
            "level": ceiling,
            "base_low": base_low,
            "decline_pct": decline_pct,
            "base_weeks": base_weeks,
            "pct_to_level": pct_below,
            "rs": _calc_rs(closes, spy_closes),
        }

    return None


def scan_base_bearish(closes, highs, lows, spy_closes):
    """
    Bearish base breakdown:
      - Peak in last 13-52 weeks
      - Established a floor after the peak (support that held)
      - Price now approaching or breaking below the floor
    """
    n = len(closes)
    if n < 52:
        return None

    # Find peak in last year (but at least 13 weeks ago)
    search_start = max(0, n - 52)
    search_end = n - 13
    if search_end <= search_start:
        return None

    peak_range = highs[search_start:search_end]
    peak_rel_idx = int(np.argmax(peak_range))
    peak_idx = search_start + peak_rel_idx
    peak = float(highs[peak_idx])

    # Find floor: lowest low between peak and 4 weeks ago
    floor_end = n - 4
    if floor_end <= peak_idx:
        return None
    floor_range = lows[peak_idx:floor_end]
    if len(floor_range) < 4:
        return None
    floor_rel_idx = int(np.argmin(floor_range))
    floor_idx = peak_idx + floor_rel_idx
    floor_val = float(lows[floor_idx])

    # The floor must have been meaningful support — price bounced at least 10%
    if floor_idx >= n - 6:
        return None
    high_after_floor = float(np.max(highs[floor_idx:floor_end]))
    if floor_val <= 0:
        return None
    bounce_pct = (high_after_floor - floor_val) / floor_val * 100
    if bounce_pct < 10:
        return None

    # Decline from peak to floor must be significant (>=20%)
    decline_from_peak = (peak - floor_val) / peak * 100
    if decline_from_peak < 20:
        return None

    current = float(closes[-1])

    # --- Triggered: close below floor in last 3 weeks ---
    if current < floor_val:
        # Confirm the breakdown is recent (wasn't below floor before last 3 weeks)
        prior_below = False
        for i in range(floor_idx + 1, max(0, n - 3)):
            if closes[i] < floor_val:
                prior_below = True
                break
        if prior_below:
            return None  # already broke down long ago

        return {
            "signal": "triggered",
            "type": "bearish",
            "price": current,
            "level": floor_val,
            "peak": peak,
            "decline_pct": decline_from_peak,
            "base_weeks": n - 1 - peak_idx,
            "pct_to_level": 0.0,
            "rs": _calc_rs(closes, spy_closes),
        }

    # --- Forming: within 5% above floor, trending down ---
    pct_above_floor = (current - floor_val) / floor_val * 100
    if pct_above_floor <= 5:
        # Must be trending downward
        if n >= 13 and closes[-1] >= closes[-13]:
            return None

        return {
            "signal": "forming",
            "type": "bearish",
            "price": current,
            "level": floor_val,
            "peak": peak,
            "decline_pct": decline_from_peak,
            "base_weeks": n - 1 - peak_idx,
            "pct_to_level": pct_above_floor,
            "rs": _calc_rs(closes, spy_closes),
        }

    return None


def scan_base(closes, highs, lows, spy_closes, side="both"):
    """Try bullish and/or bearish base detection based on side filter."""
    results = []
    if side in ("bull", "both"):
        bull = scan_base_bullish(closes, highs, lows, spy_closes)
        if bull:
            results.append(bull)
    if side in ("bear", "both"):
        bear = scan_base_bearish(closes, highs, lows, spy_closes)
        if bear:
            results.append(bear)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    if not results:
        print("\nNo Base Breakout/Breakdown setups found today.")
        return

    # Bullish triggered first, then bullish forming, then bearish
    def sort_key(r):
        type_order = 0 if r["type"] == "bullish" else 1
        sig_order = 0 if r["signal"] == "triggered" else 1
        return (type_order, sig_order, -abs(r.get("decline_pct", 0)))

    results.sort(key=sort_key)

    hdr = (f"{'Ticker':<8} {'Type':<10} {'Signal':<12} {'Price':>8} {'Level':>10} "
           f"{'Decline':>9} {'Base':>7} {'RS vs SPY':>10}  TradingView")
    print(f"\n{'=' * len(hdr)}")
    print(f" Base Breakout Screener — {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        label = "TRIGGERED" if r["signal"] == "triggered" else "Forming"
        tv = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        base_str = f"{r['base_weeks']}w" if r.get("base_weeks") else "—"
        print(f"{r['ticker']:<8} {r['type'].upper():<10} {label:<12} "
              f"{r['price']:>8.2f} {r['level']:>10.2f} "
              f"{r['decline_pct']:>8.1f}% {base_str:>7} "
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
<title>Base Breakout Screener</title>
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
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); }
  .chart-card-header .ticker { font-size: 15px; font-weight: 700; color: var(--accent);
    text-decoration: none; }
  .chart-card-header .ticker:hover { text-decoration: underline; }
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px; font-size: 11px; }
  .chart-card-body { height: 240px; position: relative; }
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
  <h1>Base Breakout Screener</h1>
  <span style="color:var(--muted);font-size:13px">__SCANNED__ tickers scanned in __ELAPSED__s &mdash; __COUNT__ setups (weekly data)</span>
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
    var levelLabel = isBull ? 'Ceiling' : 'Floor';
    var pctStr = r.pct_to_level > 0 ? r.pct_to_level.toFixed(1) + '% to ' + levelLabel.toLowerCase() : '';
    card.innerHTML =
      '<div class="chart-card-header">' +
        '<a href="' + tvUrl + '" target="_blank" class="ticker">' + r.ticker +
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></a>' +
        '<div class="meta">' +
          '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
          '<span class="info">' + levelLabel + ' $' + r.level.toFixed(2) + '</span>' +
          '<span class="info">-' + r.decline_pct.toFixed(0) + '% decline</span>' +
          (r.base_weeks ? '<span class="info">' + r.base_weeks + 'w base</span>' : '') +
          '<span class="info">RS ' + rsStr + '</span>' +
          (pctStr ? '<span class="info">' + pctStr + '</span>' : '') +
        '</div>' +
      '</div>' +
      '<div class="chart-card-body" id="chart-' + r.ticker + '"></div>';
    grid.appendChild(card);

    var container = card.querySelector('.chart-card-body');
    var ch = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 240,
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

    // Level line (ceiling for bull, floor for bear)
    candles.createPriceLine({
      price: r.level,
      color: isBull ? '#58a6ff' : '#f85149',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
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

    // EMA 10w (~50 day) and EMA 40w (~200 day)
    var closes = ohlcv.map(function(d) { return d.close; });
    var ema10vals = calcEMA(closes, 10);
    var ema10s = ch.addLineSeries({
      color: '#4CAF50', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema10s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema10vals[i].toFixed(2)) };
    }));

    var ema40vals = calcEMA(closes, 40);
    var ema40s = ch.addLineSeries({
      color: '#F44336', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema40s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema40vals[i].toFixed(2)) };
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

    print("[*] Starting web UI at http://127.0.0.1:5052")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5052")).start()
    app.run(port=5052, debug=False)


# ---------------------------------------------------------------------------
# Main
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

        hit = scan_base(closes, highs, lows, spy_closes, side=side)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            chart_data[ticker] = extract_chart_data(data, ticker)
        scanned += 1

    return results, chart_data, scanned


def main():
    parser = argparse.ArgumentParser(description="Base Breakout Stock Screener")
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
    print("[*] Downloading weekly price data (5 years)...")
    data = download_data(tickers)

    # --- Extract SPY data ---
    spy_ohlcv = extract_ohlcv(data, "SPY")
    if spy_ohlcv is None:
        print("[!] Could not load SPY data for relative strength")
        sys.exit(1)
    spy_closes = spy_ohlcv[3]

    # --- Scan ---
    side_label = {"bull": "bullish", "bear": "bearish", "both": "bull+bear"}[args.side]
    print(f"[*] Scanning for Base Breakout/Breakdown setups ({side_label})...")
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

        hit = scan_base(closes, highs, lows, spy_closes, side=args.side)
        if hit:
            hit["ticker"] = ticker
            results.append(hit)
            if not args.no_web:
                chart_data[ticker] = extract_chart_data(data, ticker)
        scanned += 1

    elapsed = time.time() - t0
    print(f"    Scanned {scanned} tickers in {elapsed:.1f}s")

    # --- Output ---
    print_results(results)

    if not args.no_web and results:
        run_web(results, chart_data, elapsed, scanned)


if __name__ == "__main__":
    main()
