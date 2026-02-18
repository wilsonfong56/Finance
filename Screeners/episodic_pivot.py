#!/usr/bin/env python3
"""
Episodic Pivot Stock Screener
==============================
Scans for Episodic Pivots — high-conviction gap events driven by major catalysts
(earnings beats, FDA approvals, acquisitions, etc.).

The gap + volume explosion is the quantitative proxy for the catalyst;
the trader manually verifies the news.

Bullish (Gap Up EP):
  1. Gap >= 10% from prior close to open
  2. RVOL >= 3x (vs 20-day average), minimum 300k shares
  3. Price > $5
  4. No prior run-up > 200% in 3 months

Bearish (Gap Down EP):
  1. Gap down >= 10%
  2. Same volume criteria
  3. Price > $5
  4. No prior decline > 66% in 3 months

Usage:
  python episodic_pivot.py                    # Interactive universe selection
  python episodic_pivot.py --ticker FSLY      # Single-stock debug mode
  python episodic_pivot.py --side bull        # Bullish only
  python episodic_pivot.py --lookback 10      # Scan last 10 trading days
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


SCREENER_NAME = "Episodic Pivot"
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


def _calc_rs(closes, spy_closes):
    """3-month relative strength: stock return minus SPY return (pct points)."""
    period = 63
    if len(closes) < period or len(spy_closes) < period:
        return 0.0
    stock_ret = (closes[-1] / closes[-period] - 1) * 100
    spy_ret = (spy_closes[-1] / spy_closes[-period] - 1) * 100
    return stock_ret - spy_ret


def _calc_avg_vol(volumes, period=20):
    """Rolling simple moving average of volume."""
    n = len(volumes)
    avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = volumes[i - period + 1: i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            avg[i] = np.mean(valid)
    return avg


def _fmt_vol(v):
    """Format volume for display."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(int(v))


# ---------------------------------------------------------------------------
# Universe: from Wikipedia
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
            # Suppress yfinance per-ticker error messages (e.g. delisted warnings)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                df = yf.download(chunk, period=period, group_by="ticker",
                                 progress=False, threads=True)
            finally:
                sys.stderr = old_stderr
            frames.append(df)
        except Exception as e:
            print(f"\n  [!] Chunk download error: {e}")
    print()

    if not frames:
        print("[!] No data downloaded")
        sys.exit(1)

    data = pd.concat(frames, axis=1)

    # Report tickers with no data in a single summary line
    failed = []
    for t in all_tickers:
        try:
            sub = data[t]
            if sub["Close"].isna().all():
                failed.append(t)
        except (KeyError, TypeError):
            failed.append(t)
    if failed:
        failed.sort()
        print(f"  [!] No data for {len(failed)} ticker(s): {', '.join(failed)}")

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


def extract_ohlcv_with_dates(data, ticker):
    """Pull OHLCV arrays + date index for a single ticker."""
    try:
        sub = data[ticker]
        dates = sub.index
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
    return dates[first:], o[first:], h[first:], lo[first:], c[first:], v[first:]


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
# Scan logic — Bullish (Gap Up EP)
# ---------------------------------------------------------------------------

def scan_ticker_bullish(dates, opens, highs, lows, closes, volumes,
                        spy_closes, lookback_days=5):
    """
    Scan for bullish episodic pivot (gap up).
    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None
    if closes[-1] < 5.0:
        return None

    # prior run-up check: disqualify if >200% in 3 months
    period_3mo = min(63, n - 1)
    window = closes[max(0, n - period_3mo - 1): n - 1]
    if len(window) > 0 and min(window) > 0:
        runup = (closes[-1] / min(window) - 1) * 100
        if runup > 200:
            return None

    avg_vol = _calc_avg_vol(volumes, 20)

    scan_start = max(20, n - lookback_days)
    best_hit = None

    for i in range(scan_start, n):
        prior_close = closes[i - 1]
        if prior_close <= 0:
            continue

        # gap up >= 10%
        gap_pct = (opens[i] / prior_close - 1) * 100
        if gap_pct < 10.0:
            continue

        # gap must not be closed: low must stay above prior close
        if lows[i] <= prior_close:
            continue

        # volume explosion — use avg from day BEFORE the gap
        if np.isnan(avg_vol[i - 1]) or avg_vol[i - 1] <= 0:
            continue
        if np.isnan(volumes[i]) or volumes[i] < 300_000:
            continue
        rvol = volumes[i] / avg_vol[i - 1]
        if rvol < 3.0:
            continue

        # enhancers
        held_gains = closes[i] >= opens[i]

        gap_day_low = lows[i]
        building_above = True
        for j in range(i + 1, n):
            if lows[j] < gap_day_low:
                building_above = False
                break

        days_since = n - 1 - i
        rs = _calc_rs(closes, spy_closes)

        hit = {
            "type": "bullish",
            "signal": "triggered",
            "price": round(float(closes[-1]), 2),
            "gap_pct": round(float(gap_pct), 1),
            "rvol": round(float(rvol), 1),
            "volume": int(volumes[i]),
            "gap_date": dates[i].strftime("%Y-%m-%d"),
            "days_since_gap": int(days_since),
            "held_gains": bool(held_gains),
            "building_above": bool(building_above),
            "gap_day_low": round(float(gap_day_low), 2),
            "gap_day_high": round(float(highs[i]), 2),
            "rs": round(float(rs), 1),
        }

        # keep most recent; if tied, prefer higher RVOL
        if best_hit is None or days_since < best_hit["days_since_gap"]:
            best_hit = hit
        elif days_since == best_hit["days_since_gap"] and rvol > best_hit["rvol"]:
            best_hit = hit

    return best_hit


# ---------------------------------------------------------------------------
# Scan logic — Bearish (Gap Down EP)
# ---------------------------------------------------------------------------

def scan_ticker_bearish(dates, opens, highs, lows, closes, volumes,
                        spy_closes, lookback_days=5):
    """
    Scan for bearish episodic pivot (gap down).
    Returns dict or None.
    """
    n = len(closes)
    if n < 60:
        return None
    if closes[-1] < 5.0:
        return None

    # prior decline check: disqualify if >66% decline in 3 months
    period_3mo = min(63, n - 1)
    window = closes[max(0, n - period_3mo - 1): n - 1]
    if len(window) > 0 and max(window) > 0:
        decline = (1 - closes[-1] / max(window)) * 100
        if decline > 66:
            return None

    avg_vol = _calc_avg_vol(volumes, 20)

    scan_start = max(20, n - lookback_days)
    best_hit = None

    for i in range(scan_start, n):
        prior_close = closes[i - 1]
        if prior_close <= 0:
            continue

        # gap down >= 10%
        gap_pct = (prior_close - opens[i]) / prior_close * 100
        if gap_pct < 10.0:
            continue

        # gap must not be closed: high must stay below prior close
        if highs[i] >= prior_close:
            continue

        # volume explosion
        if np.isnan(avg_vol[i - 1]) or avg_vol[i - 1] <= 0:
            continue
        if np.isnan(volumes[i]) or volumes[i] < 300_000:
            continue
        rvol = volumes[i] / avg_vol[i - 1]
        if rvol < 3.0:
            continue

        # enhancers
        held_losses = closes[i] <= opens[i]

        gap_day_high = highs[i]
        staying_below = True
        for j in range(i + 1, n):
            if highs[j] > gap_day_high:
                staying_below = False
                break

        days_since = n - 1 - i
        rs = _calc_rs(closes, spy_closes)

        hit = {
            "type": "bearish",
            "signal": "triggered",
            "price": round(float(closes[-1]), 2),
            "gap_pct": round(float(gap_pct), 1),
            "rvol": round(float(rvol), 1),
            "volume": int(volumes[i]),
            "gap_date": dates[i].strftime("%Y-%m-%d"),
            "days_since_gap": int(days_since),
            "held_gains": bool(held_losses),
            "building_above": bool(staying_below),
            "gap_day_low": round(float(lows[i]), 2),
            "gap_day_high": round(float(highs[i]), 2),
            "rs": round(float(rs), 1),
        }

        if best_hit is None or days_since < best_hit["days_since_gap"]:
            best_hit = hit
        elif days_since == best_hit["days_since_gap"] and rvol > best_hit["rvol"]:
            best_hit = hit

    return best_hit


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results):
    """Print a formatted table of results."""
    if not results:
        print("\nNo Episodic Pivot setups found.")
        return

    results.sort(key=lambda r: (
        r["days_since_gap"],
        0 if r["type"] == "bullish" else 1,
        -r["rvol"],
    ))

    hdr = (f"{'Ticker':<8} {'Type':<10} {'Gap%':>7} {'RVOL':>7} {'Volume':>10} "
           f"{'Gap Date':>11} {'Days':>5} {'Held':>5} {'RS':>8}  TradingView")
    print(f"\n{'=' * len(hdr)}")
    print(f" Episodic Pivot Screener — {len(results)} setups found")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        sign = "+" if r["type"] == "bullish" else "-"
        held = "Y" if r["held_gains"] else "N"
        tv_link = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
        print(f"{r['ticker']:<8} {r['type'].upper():<10} {sign}{r['gap_pct']:>5.1f}% "
              f"{r['rvol']:>6.1f}x {_fmt_vol(r['volume']):>10} "
              f"{r['gap_date']:>11} {r['days_since_gap']:>4}d {held:>5} "
              f"{r['rs']:>+7.1f}  {tv_link}")

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
<title>Episodic Pivot Screener</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --cyan: #39d2e0; }
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
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px;
    font-size: 12px; flex-wrap: wrap; }
  .chart-card-body { height: 220px; position: relative; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }
  .badge.bull-triggered { background: rgba(63,185,80,0.2); color: var(--green); }
  .badge.bear-triggered { background: rgba(248,81,73,0.2); color: var(--red); }
  .badge.held { background: rgba(57,210,224,0.15); color: var(--cyan); }
  .info { color: var(--muted); }
  .modal-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.75); z-index:1000; justify-content:center; align-items:center; }
  .modal-overlay.active { display:flex; }
  .modal-content { width:92%; height:88%; background:var(--surface); border-radius:10px;
    overflow:hidden; position:relative; border:1px solid var(--border); }
  .modal-close { position:absolute; top:8px; right:12px; font-size:28px; color:var(--muted);
    cursor:pointer; z-index:1001; background:none; border:none; line-height:1; }
  .modal-close:hover { color:var(--text); }
  .modal-content iframe { width:100%; height:100%; border:none; }
</style>
</head>
<body>
<header>
  <h1>Episodic Pivot Screener</h1>
  <span style="color:var(--muted);font-size:13px">__SCANNED__ tickers scanned in __ELAPSED__s &mdash; __COUNT__ setups</span>
</header>
<main>
  <div class="chart-grid" id="chart-grid"></div>
</main>
<div class="modal-overlay" id="tv-modal" onclick="if(event.target===this)closeModal()">
  <div class="modal-content">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <iframe id="tv-iframe" src=""></iframe>
  </div>
</div>
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

function fmtVol(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return v.toString();
}

function buildCards() {
  var grid = document.getElementById('chart-grid');
  RESULTS.forEach(function(r) {
    var ohlcv = CHART_DATA[r.ticker];
    if (!ohlcv || !ohlcv.length) return;

    var card = document.createElement('div');
    card.className = 'chart-card';
    var isBull = r.type === 'bullish';
    var badgeCls = isBull ? 'bull-triggered' : 'bear-triggered';
    var gapLabel = isBull ? 'GAP UP' : 'GAP DN';
    var gapSign = isBull ? '+' : '-';
    var rsStr = (r.rs >= 0 ? '+' : '') + r.rs.toFixed(1);
    var heldBadge = r.held_gains ? '<span class="badge held">HELD</span>' : '';
    var dateObj = new Date(r.gap_date + 'T12:00:00');
    var dateStr = dateObj.toLocaleDateString('en-US', {month:'short', day:'numeric'});
    var daysStr = r.days_since_gap === 0 ? 'Today' : r.days_since_gap + 'd ago';

    card.innerHTML =
      '<div class="chart-card-header">' +
        '<a href="#" data-ticker="' + r.ticker + '" class="ticker">' + r.ticker +
          ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></a>' +
        '<div class="meta">' +
          '<span class="badge ' + badgeCls + '">' + gapLabel + '</span>' +
          heldBadge +
          '<span class="info">' + gapSign + r.gap_pct.toFixed(1) + '%</span>' +
          '<span class="info">' + r.rvol.toFixed(1) + 'x vol</span>' +
          '<span class="info">' + fmtVol(r.volume) + '</span>' +
          '<span class="info">' + dateStr + ' (' + daysStr + ')</span>' +
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

    // marker on gap day
    candles.setMarkers([{
      time: r.gap_date,
      position: isBull ? 'belowBar' : 'aboveBar',
      color: isBull ? '#3fb950' : '#f85149',
      shape: isBull ? 'arrowUp' : 'arrowDown',
      text: gapSign + r.gap_pct.toFixed(0) + '%',
    }]);

    // key level: gap day low (bull) or gap day high (bear)
    var keyLevel = isBull ? r.gap_day_low : r.gap_day_high;
    candles.createPriceLine({
      price: keyLevel,
      color: isBull ? '#58a6ff' : '#f85149',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: false,
    });

    // volume histogram
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

    // 20/50 EMA
    var closes = ohlcv.map(function(d) { return d.close; });
    var ema20vals = calcEMA(closes, 20);
    var ema20s = ch.addLineSeries({
      color: '#4CAF50', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema20s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema20vals[i].toFixed(2)) };
    }));

    var ema50vals = calcEMA(closes, 50);
    var ema50s = ch.addLineSeries({
      color: '#F44336', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema50s.setData(ohlcv.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema50vals[i].toFixed(2)) };
    }));

    ch.timeScale().fitContent();

    new ResizeObserver(function() {
      ch.applyOptions({ width: container.clientWidth });
    }).observe(container);
  });
}

function openModal(ticker) {
  var iframe = document.getElementById('tv-iframe');
  iframe.src = 'https://s.tradingview.com/widgetembed/?symbol=' + ticker +
    '&interval=D&hideideas=1&theme=dark&style=1&timezone=exchange&withdateranges=1' +
    '&hide_side_toolbar=0&allow_symbol_change=1&saveimage=0&toolbarbg=161b22';
  document.getElementById('tv-modal').classList.add('active');
}
function closeModal() {
  document.getElementById('tv-modal').classList.remove('active');
  document.getElementById('tv-iframe').src = '';
}
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });

buildCards();

document.getElementById('chart-grid').addEventListener('click', function(e) {
  var link = e.target.closest('a.ticker');
  if (link) { e.preventDefault(); openModal(link.dataset.ticker); }
});
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

    print("[*] Starting web UI at http://127.0.0.1:5054")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5054")).start()
    app.run(port=5054, debug=False)


# ---------------------------------------------------------------------------
# Standard module interface for combined screener
# ---------------------------------------------------------------------------

def run_scan(tickers, side="both", data=None, lookback_days=5):
    """Self-contained scan: downloads data (unless provided), scans, returns results.
    Returns: (results: list[dict], chart_data: dict, scanned: int)
    """
    if data is None:
        data = download_data(tickers)

    spy_data = extract_ohlcv_with_dates(data, "SPY")
    if spy_data is None:
        return [], {}, 0
    spy_closes = spy_data[4]

    results = []
    chart_data = {}
    scanned = 0
    for ticker in tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if side in ("bull", "both"):
            hit = scan_ticker_bullish(dates, opens, highs, lows, closes,
                                      volumes, spy_closes, lookback_days)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                chart_data[ticker] = extract_chart_data(data, ticker)

        if side in ("bear", "both"):
            hit = scan_ticker_bearish(dates, opens, highs, lows, closes,
                                      volumes, spy_closes, lookback_days)
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
    parser = argparse.ArgumentParser(description="Episodic Pivot Stock Screener")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI (terminal output only)")
    parser.add_argument("--side", choices=["bull", "bear", "both"], default="both",
                        help="Scan for bullish, bearish, or both setups (default: both)")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Scan for EPs within the last N trading days (default: 5)")
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
    spy_data = extract_ohlcv_with_dates(data, "SPY")
    if spy_data is None:
        print("[!] Could not load SPY data for relative strength")
        sys.exit(1)
    spy_closes = spy_data[4]

    # --- Scan ---
    side_label = {"bull": "bullish", "bear": "bearish", "both": "bull+bear"}[args.side]
    print(f"[*] Scanning for Episodic Pivots ({side_label}, last {args.lookback} days)...")
    results = []
    chart_data = {}
    scanned = 0
    for ticker in tickers:
        ohlcv = extract_ohlcv_with_dates(data, ticker)
        if ohlcv is None:
            continue
        dates, opens, highs, lows, closes, volumes = ohlcv
        scanned += 1

        if args.side in ("bull", "both"):
            hit = scan_ticker_bullish(dates, opens, highs, lows, closes,
                                      volumes, spy_closes, args.lookback)
            if hit:
                hit["ticker"] = ticker
                results.append(hit)
                if not args.no_web:
                    chart_data[ticker] = extract_chart_data(data, ticker)

        if args.side in ("bear", "both"):
            hit = scan_ticker_bearish(dates, opens, highs, lows, closes,
                                      volumes, spy_closes, args.lookback)
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
