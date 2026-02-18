#!/usr/bin/env python3
"""
Combined Stock Screener
========================
Auto-discovers screener modules in the Screeners/ directory and aggregates
their results into a single tabbed web UI.

Only tickers that appear in ALL screeners' results are kept (intersection mode).
Data is downloaded once per unique timeframe and shared across screeners.

Each screener module must expose:
  SCREENER_NAME = "Display Name"
  DATA_PARAMS   = {"period": "6mo", "interval": "1d"}
  def run_scan(tickers, side="both", data=None):
      -> (results: list[dict], chart_data: dict, scanned: int)

Usage:
  python combined_screener.py                          # all screeners, interactive universe
  python combined_screener.py --side bear              # bearish only
  python combined_screener.py --screeners wedge_pop,ema_flip
  python combined_screener.py --ticker AAPL --no-web
"""

import argparse
import importlib
import io
import json
import os
import sys
import threading
import time
import webbrowser
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Auto-discovery of screener modules
# ---------------------------------------------------------------------------

def discover_screeners():
    """Find all .py files in this directory that expose SCREENER_NAME + run_scan."""
    screener_dir = os.path.dirname(os.path.abspath(__file__))
    this_file = os.path.basename(__file__)
    modules = {}

    # Ensure the screener dir is on sys.path for imports
    if screener_dir not in sys.path:
        sys.path.insert(0, screener_dir)

    for fname in sorted(os.listdir(screener_dir)):
        if fname == this_file or not fname.endswith(".py") or fname.startswith("_"):
            continue

        mod_name = fname[:-3]
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        if hasattr(mod, "SCREENER_NAME") and callable(getattr(mod, "run_scan", None)):
            data_params = getattr(mod, "DATA_PARAMS", {"period": "6mo", "interval": "1d"})
            modules[mod_name] = {
                "name": mod.SCREENER_NAME,
                "run_scan": mod.run_scan,
                "data_params": data_params,
            }

    return modules


# ---------------------------------------------------------------------------
# Shared data download
# ---------------------------------------------------------------------------

def download_data(tickers, period="6mo", interval="1d"):
    """Batch-download OHLCV data for all tickers + SPY."""
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
            df = yf.download(chunk, period=period, interval=interval,
                             group_by="ticker", progress=False, threads=True)
            frames.append(df)
        except Exception as e:
            print(f"\n  [!] Chunk download error: {e}")
    print()

    if not frames:
        return None

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Universe scrapers (same shared pattern)
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
# Output formatting
# ---------------------------------------------------------------------------

def print_results(all_results, common_tickers):
    """Print combined results grouped by screener (intersection only)."""
    total = sum(len(r) for r in all_results.values())
    if total == 0:
        print("\nNo tickers matched all screeners.")
        return

    print(f"\n  {len(common_tickers)} ticker(s) matched ALL screeners:")

    for screener_name, results in all_results.items():
        if not results:
            continue

        results.sort(key=lambda r: (
            0 if r.get("type") == "bullish" else 1,
            0 if r.get("signal") == "triggered" else 1,
            -r.get("rs", 0),
        ))

        print(f"\n--- {screener_name} ({len(results)} setups) ---")
        for r in results:
            sig = r.get("signal", "").upper()
            typ = r.get("type", "").upper()
            tv = f"https://www.tradingview.com/chart/?symbol={r['ticker']}"
            rs_str = f"{r.get('rs', 0):+.1f}"
            print(f"  {r['ticker']:<8} {typ:<10} {sig:<12} ${r['price']:>8.2f}  RS {rs_str}  {tv}")

    print(f"\n  {len(common_tickers)} unique ticker(s) confirmed across {len([v for v in all_results.values() if v])} screeners")
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
<title>Combined Screener</title>
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
  .tabs { display: flex; gap: 0; margin: 0 24px; border-bottom: 1px solid var(--border); }
  .tab { padding: 10px 20px; cursor: pointer; color: var(--muted); font-size: 13px;
         font-weight: 600; border-bottom: 2px solid transparent; transition: all 0.2s; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .count { display: inline-block; background: var(--border); color: var(--muted);
    font-size: 10px; padding: 1px 6px; border-radius: 10px; margin-left: 6px; }
  .tab.active .count { background: rgba(88,166,255,0.2); color: var(--accent); }
  main { padding: 20px 24px; max-width: 1800px; margin: 0 auto; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }
  .chart-card { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; overflow: hidden; }
  .chart-card-header { padding: 10px 14px; display: flex; justify-content: space-between;
                       align-items: center; border-bottom: 1px solid var(--border); }
  .chart-card { cursor: pointer; transition: border-color 0.2s; }
  .chart-card:hover { border-color: var(--accent); }
  .chart-card-header .ticker { font-size: 15px; font-weight: 700; color: var(--accent); }
  .chart-card-header .meta { display: flex; align-items: center; gap: 8px; font-size: 11px; }
  .chart-card-body { height: 220px; position: relative; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 7px; border-radius: 4px; letter-spacing: 0.3px; }
  .badge.bull-triggered { background: rgba(63,185,80,0.2); color: var(--green); }
  .badge.bull-forming   { background: rgba(210,153,34,0.2); color: var(--yellow); }
  .badge.bear-triggered { background: rgba(248,81,73,0.2); color: var(--red); }
  .badge.bear-forming   { background: rgba(219,109,40,0.2); color: var(--orange); }
  .info { color: var(--muted); }
  .empty-msg { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }
  /* Modal */
  .modal-overlay { display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.8); align-items: center; justify-content: center; }
  .modal-overlay.active { display: flex; }
  .modal-content { background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; width: 90vw; max-width: 1200px; max-height: 90vh; overflow: auto; }
  .modal-header { display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid var(--border); }
  .modal-header h2 { font-size: 18px; font-weight: 700; }
  .modal-header .tv-link { color: var(--accent); font-size: 13px; text-decoration: none; margin-left: 12px; }
  .modal-header .tv-link:hover { text-decoration: underline; }
  .modal-header .close-btn { background: none; border: none; color: var(--muted);
    font-size: 24px; cursor: pointer; padding: 0 4px; }
  .modal-header .close-btn:hover { color: var(--text); }
  .modal-tf-bar { display: flex; gap: 12px; align-items: center; padding: 10px 20px; flex-wrap: wrap; }
  .tf-group { display: flex; align-items: center; gap: 4px; }
  .tf-label { color: var(--muted); font-size: 12px; margin-right: 4px; }
  .tf-btn { background: var(--bg); border: 1px solid var(--border); color: var(--muted);
    font-size: 11px; padding: 4px 10px; border-radius: 4px; cursor: pointer; }
  .tf-btn:hover { color: var(--text); border-color: var(--muted); }
  .tf-btn.active { color: var(--accent); border-color: var(--accent); background: rgba(88,166,255,0.1); }
  .modal-chart-container { height: 500px; }
  .spinner { width: 28px; height: 28px; border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Combined Screener</h1>
  <span style="color:var(--muted);font-size:13px">__COMMON_COUNT__ tickers matched all __SCREENER_COUNT__ screeners &mdash; __ELAPSED__s</span>
</header>
<div class="tabs" id="tabs"></div>
<main id="main"></main>

<!-- Modal -->
<div class="modal-overlay" id="modal-overlay">
  <div class="modal-content">
    <div class="modal-header">
      <h2 id="modal-title"></h2>
      <div style="display:flex;align-items:center;gap:12px;">
        <a id="modal-tv-link" class="tv-link" href="#" target="_blank">TradingView</a>
        <button class="close-btn" onclick="closeModal()">&times;</button>
      </div>
    </div>
    <div class="modal-tf-bar" id="modal-tf-bar">
      <div class="tf-group">
        <span class="tf-label">Interval:</span>
        <button class="tf-btn active" data-interval="1d" data-range="6mo">Daily</button>
        <button class="tf-btn" data-interval="1wk" data-range="5y">Weekly</button>
        <button class="tf-btn" data-interval="1mo" data-range="max">Monthly</button>
      </div>
      <div class="tf-group">
        <span class="tf-label">Range:</span>
        <button class="tf-btn" data-range="1mo">1M</button>
        <button class="tf-btn" data-range="3mo">3M</button>
        <button class="tf-btn active" data-range="6mo">6M</button>
        <button class="tf-btn" data-range="1y">1Y</button>
        <button class="tf-btn" data-range="5y">5Y</button>
        <button class="tf-btn" data-range="max">Max</button>
      </div>
    </div>
    <div class="modal-chart-container" id="modal-chart-container"></div>
  </div>
</div>

<script>
var ALL_DATA = __ALL_DATA_JSON__;

function calcEMA(closes, period) {
  var k = 2 / (period + 1);
  var ema = [closes[0]];
  for (var i = 1; i < closes.length; i++) {
    ema.push(closes[i] * k + ema[i - 1] * (1 - k));
  }
  return ema;
}

var renderedTabs = {};

function renderCharts(key) {
  if (renderedTabs[key]) return;
  renderedTabs[key] = true;

  var data = ALL_DATA[key];
  var results = data.results;
  var chartData = data.chart_data;
  var contentEl = document.getElementById('content-' + key);
  var cards = contentEl.querySelectorAll('.chart-card-body');

  results.forEach(function(r, idx) {
    var ohlcv = chartData[r.ticker];
    if (!ohlcv || !ohlcv.length) return;
    if (ohlcv.length > 60) ohlcv = ohlcv.slice(-60);

    var clean = ohlcv.filter(function(d) {
      return d.open > 0 && d.high > 0 && d.low > 0 && d.close > 0;
    });
    if (!clean.length) return;

    var container = cards[idx];
    if (!container) return;
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
    candles.setData(clean.map(function(d) {
      return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
    }));

    var vol = ch.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      priceLineVisible: false, lastValueVisible: false,
    });
    ch.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    vol.setData(clean.map(function(d) {
      return { time: d.time, value: d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,0.2)' : 'rgba(248,81,73,0.2)' };
    }));

    var closes = clean.map(function(d) { return d.close; });
    var ema8vals = calcEMA(closes, 8);
    var ema8s = ch.addLineSeries({
      color: '#4CAF50', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema8s.setData(clean.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema8vals[i].toFixed(2)) };
    }));

    var ema21vals = calcEMA(closes, 21);
    var ema21s = ch.addLineSeries({
      color: '#F44336', lineWidth: 1, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
    });
    ema21s.setData(clean.map(function(d, i) {
      return { time: d.time, value: parseFloat(ema21vals[i].toFixed(2)) };
    }));

    ch.timeScale().fitContent();

    new ResizeObserver(function() {
      ch.applyOptions({ width: container.clientWidth });
    }).observe(container);
  });
}

function buildUI() {
  var tabsEl = document.getElementById('tabs');
  var mainEl = document.getElementById('main');
  var screenerNames = Object.keys(ALL_DATA);
  var first = true;

  screenerNames.forEach(function(key) {
    var data = ALL_DATA[key];
    var results = data.results;
    var displayName = data.display_name;

    // Tab
    var tab = document.createElement('div');
    tab.className = 'tab' + (first ? ' active' : '');
    tab.setAttribute('data-tab', key);
    tab.innerHTML = displayName + '<span class="count">' + results.length + '</span>';
    tab.onclick = function() {
      document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
      document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      document.getElementById('content-' + key).classList.add('active');
      renderCharts(key);
    };
    tabsEl.appendChild(tab);

    // Content
    var content = document.createElement('div');
    content.className = 'tab-content' + (first ? ' active' : '');
    content.id = 'content-' + key;

    if (results.length === 0) {
      content.innerHTML = '<div class="empty-msg">No setups found for ' + displayName + '</div>';
      mainEl.appendChild(content);
      first = false;
      return;
    }

    var grid = document.createElement('div');
    grid.className = 'chart-grid';
    content.appendChild(grid);
    mainEl.appendChild(content);

    // Build card HTML only (no charts yet)
    results.forEach(function(r) {
      var ohlcv = data.chart_data[r.ticker];
      if (!ohlcv || !ohlcv.length) return;

      var card = document.createElement('div');
      card.className = 'chart-card';
      var isBull = r.type === 'bullish';
      var isTrig = r.signal === 'triggered';
      var badgeCls = (isBull ? 'bull' : 'bear') + '-' + (isTrig ? 'triggered' : 'forming');
      var badgeLabel = (isBull ? 'BULL ' : 'BEAR ') + (isTrig ? 'TRIGGERED' : 'FORMING');
      var rsVal = r.rs || 0;
      var rsStr = (rsVal >= 0 ? '+' : '') + rsVal.toFixed(1);
      card.innerHTML =
        '<div class="chart-card-header">' +
          '<span class="ticker">' + r.ticker +
            ' <span style="font-weight:400;font-size:12px;color:#8b949e">$' + r.price.toFixed(2) + '</span></span>' +
          '<div class="meta">' +
            '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
            '<span class="info">RS ' + rsStr + '</span>' +
          '</div>' +
        '</div>' +
        '<div class="chart-card-body"></div>';
      card.onclick = function() { openModal(r.ticker); };
      grid.appendChild(card);
    });

    // Render charts for the first (visible) tab immediately
    if (first) {
      renderCharts(key);
    }

    first = false;
  });
}

buildUI();

// ── Modal ──
var modalChart = null;
var modalTicker = null;
var modalInterval = '1d';
var modalRange = '6mo';
var modalResizeObserver = null;

function openModal(ticker) {
  modalTicker = ticker;
  modalInterval = '1d';
  modalRange = '6mo';
  document.getElementById('modal-title').textContent = ticker;
  document.getElementById('modal-tv-link').href = 'https://www.tradingview.com/chart/?symbol=' + ticker;
  document.getElementById('modal-overlay').classList.add('active');

  var bar = document.getElementById('modal-tf-bar');
  var groups = bar.querySelectorAll('.tf-group');
  groups.forEach(function(group, idx) {
    group.querySelectorAll('.tf-btn').forEach(function(b) {
      if (idx === 0) {
        b.classList.toggle('active', b.dataset.interval === '1d');
      } else {
        b.classList.toggle('active', b.dataset.range === '6mo');
      }
    });
  });

  loadModalChart();
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('active');
  if (modalChart) { modalChart.remove(); modalChart = null; }
  if (modalResizeObserver) { modalResizeObserver.disconnect(); modalResizeObserver = null; }
  modalTicker = null;
}

document.getElementById('modal-overlay').addEventListener('click', function(e) {
  if (e.target === e.currentTarget) closeModal();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

// Timeframe bar
document.getElementById('modal-tf-bar').addEventListener('click', function(e) {
  var btn = e.target.closest('.tf-btn');
  if (!btn || !modalTicker) return;

  var group = btn.closest('.tf-group');
  var isIntervalGroup = group === group.parentElement.firstElementChild;

  if (isIntervalGroup) {
    modalInterval = btn.dataset.interval;
    modalRange = btn.dataset.range;
    group.querySelectorAll('.tf-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var rangeGroup = group.nextElementSibling;
    rangeGroup.querySelectorAll('.tf-btn').forEach(function(b) {
      b.classList.toggle('active', b.dataset.range === modalRange);
    });
  } else {
    modalRange = btn.dataset.range;
    group.querySelectorAll('.tf-btn').forEach(function(b) { b.classList.remove('active'); });
    btn.classList.add('active');
  }

  loadModalChart();
});

async function loadModalChart() {
  var container = document.getElementById('modal-chart-container');
  container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%"><div class="spinner"></div></div>';

  if (modalResizeObserver) { modalResizeObserver.disconnect(); modalResizeObserver = null; }
  if (modalChart) { modalChart.remove(); modalChart = null; }

  try {
    var res = await fetch('/api/chart/' + modalTicker + '?interval=' + modalInterval + '&range=' + modalRange);
    var data = await res.json();
    if (data.error) throw new Error(data.error);
    if (!data.length) throw new Error('No data returned');

    container.innerHTML = '';

    modalChart = LightweightCharts.createChart(container, {
      width: container.clientWidth, height: 500,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e' },
      grid: { vertLines: { color: '#1e252e' }, horzLines: { color: '#1e252e' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', timeVisible: false },
    });

    var candleSeries = modalChart.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderUpColor: '#3fb950', borderDownColor: '#f85149',
      wickUpColor: '#3fb950', wickDownColor: '#f85149',
    });
    candleSeries.setData(data.map(function(d) {
      return { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close };
    }));

    var volSeries = modalChart.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      priceLineVisible: false, lastValueVisible: false,
    });
    modalChart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volSeries.setData(data.map(function(d) {
      return { time: d.time, value: d.volume,
        color: d.close >= d.open ? 'rgba(63,185,80,0.25)' : 'rgba(248,81,73,0.25)' };
    }));

    var closes = data.map(function(d) { return d.close; });
    if (closes.length > 0) {
      var ema8vals = calcEMA(closes, 8);
      var ema8s = modalChart.addLineSeries({
        color: '#4CAF50', lineWidth: 2, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      });
      ema8s.setData(data.map(function(d, i) {
        return { time: d.time, value: parseFloat(ema8vals[i].toFixed(2)) };
      }));

      var ema21vals = calcEMA(closes, 21);
      var ema21s = modalChart.addLineSeries({
        color: '#F44336', lineWidth: 2, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      });
      ema21s.setData(data.map(function(d, i) {
        return { time: d.time, value: parseFloat(ema21vals[i].toFixed(2)) };
      }));
    }

    modalChart.timeScale().fitContent();

    modalResizeObserver = new ResizeObserver(function() {
      if (modalChart) modalChart.applyOptions({ width: container.clientWidth });
    });
    modalResizeObserver.observe(container);

  } catch (e) {
    container.innerHTML = '<div style="color:var(--red);padding:20px">Chart error: ' + e.message + '</div>';
  }
}
</script>
</body>
</html>"""


def build_html(all_data, elapsed, common_count, screener_count):
    """Build the combined HTML page."""
    def _convert(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if np.isnan(v) or np.isinf(v) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Sanitize NaN/Inf in results before serialization
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        return obj

    clean_data = _sanitize(all_data)

    html = _HTML_TEMPLATE
    html = html.replace("__ALL_DATA_JSON__", json.dumps(clean_data, default=_convert))
    html = html.replace("__ELAPSED__", f"{elapsed:.1f}")
    html = html.replace("__COMMON_COUNT__", str(common_count))
    html = html.replace("__SCREENER_COUNT__", str(screener_count))
    return html


def run_web(all_data, elapsed, common_count, screener_count):
    """Start a Flask server and open the browser."""
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    page = build_html(all_data, elapsed, common_count, screener_count)

    @app.route("/")
    def index():
        return page

    @app.route("/api/chart/<ticker>")
    def chart_api(ticker):
        interval = request.args.get("interval", "1d")
        rng = request.args.get("range", "6mo")
        try:
            df = yf.download(ticker, period=rng, interval=interval, progress=False)
            if df.empty:
                return jsonify({"error": "No data"})
            records = []
            for dt, row in df.iterrows():
                t = dt.strftime("%Y-%m-%d") if interval in ("1d", "1wk", "1mo") else int(dt.timestamp())
                o = float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"])
                h = float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"])
                l = float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"])
                c = float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"])
                v = float(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else float(row["Volume"])
                records.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
            return jsonify(records)
        except Exception as e:
            return jsonify({"error": str(e)})

    print("[*] Starting combined web UI at http://127.0.0.1:5055")
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5055")).start()
    app.run(port=5055, debug=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combined Stock Screener")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scan a single ticker (debug mode)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web UI (terminal output only)")
    parser.add_argument("--side", choices=["bull", "bear", "both"], default="both",
                        help="Scan for bullish, bearish, or both setups (default: both)")
    parser.add_argument("--screeners", type=str, default=None,
                        help="Comma-separated list of screener module names to run "
                             "(e.g. wedge_pop,ema_flip). Default: all discovered.")
    args = parser.parse_args()

    t0 = time.time()

    # --- Discover screeners ---
    all_screeners = discover_screeners()
    if not all_screeners:
        print("[!] No screener modules found in this directory")
        sys.exit(1)

    # Filter if --screeners specified
    if args.screeners:
        requested = [s.strip() for s in args.screeners.split(",")]
        filtered = {}
        for r in requested:
            if r in all_screeners:
                filtered[r] = all_screeners[r]
            else:
                print(f"[!] Unknown screener '{r}'. Available: {', '.join(all_screeners.keys())}")
        all_screeners = filtered

    if not all_screeners:
        print("[!] No valid screeners selected")
        sys.exit(1)

    print(f"[*] Running {len(all_screeners)} screener(s): {', '.join(s['name'] for s in all_screeners.values())}")

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

    # --- Download data once per unique timeframe ---
    # Group screeners by their DATA_PARAMS so we only download each combo once
    data_groups = defaultdict(list)
    for mod_name, info in all_screeners.items():
        key = (info["data_params"]["period"], info["data_params"]["interval"])
        data_groups[key].append(mod_name)

    downloaded = {}
    for (period, interval), mod_names in data_groups.items():
        screener_names = [all_screeners[m]["name"] for m in mod_names]
        print(f"\n[*] Downloading {interval} data ({period}) for: {', '.join(screener_names)}...")
        data = download_data(tickers, period=period, interval=interval)
        if data is None:
            print(f"  [!] Download failed for {period}/{interval}")
            continue
        downloaded[(period, interval)] = data

    # --- Run each screener with pre-downloaded data ---
    all_results = {}       # display_name -> results list (pre-intersection)
    all_data = {}          # mod_name -> full data dict
    ticker_sets = []       # list of sets for intersection

    for mod_name, screener_info in all_screeners.items():
        display_name = screener_info["name"]
        run_fn = screener_info["run_scan"]
        dp = screener_info["data_params"]
        data_key = (dp["period"], dp["interval"])

        data = downloaded.get(data_key)
        if data is None:
            print(f"  [!] No data available for {display_name}, skipping")
            ticker_sets.append(set())
            all_results[display_name] = []
            all_data[mod_name] = {
                "display_name": display_name,
                "results": [],
                "chart_data": {},
                "scanned": 0,
            }
            continue

        print(f"[*] Running {display_name} screener...")

        try:
            results, chart_data, scanned = run_fn(tickers, side=args.side, data=data)
        except Exception as e:
            print(f"  [!] Error in {display_name}: {e}")
            results, chart_data, scanned = [], {}, 0

        print(f"    {display_name}: {len(results)} setups found ({scanned} scanned)")

        # Track tickers found by this screener
        ticker_sets.append({r["ticker"] for r in results})

        all_results[display_name] = results
        all_data[mod_name] = {
            "display_name": display_name,
            "results": results,
            "chart_data": chart_data,
            "scanned": scanned,
        }

    # --- Intersection: keep only tickers found in ALL screeners ---
    if ticker_sets:
        common_tickers = ticker_sets[0]
        for s in ticker_sets[1:]:
            common_tickers &= s
    else:
        common_tickers = set()

    n_before = sum(len(r) for r in all_results.values())
    print(f"\n[*] Intersection: {len(common_tickers)} ticker(s) matched all {len(all_screeners)} screeners"
          f" (from {n_before} total hits)")

    # Filter results and chart_data to common tickers only
    for mod_name in all_data:
        all_data[mod_name]["results"] = [
            r for r in all_data[mod_name]["results"] if r["ticker"] in common_tickers
        ]
        all_data[mod_name]["chart_data"] = {
            k: v for k, v in all_data[mod_name]["chart_data"].items() if k in common_tickers
        }

    # Rebuild all_results after filtering
    all_results = {
        all_data[m]["display_name"]: all_data[m]["results"]
        for m in all_data
    }

    elapsed = time.time() - t0
    print(f"[*] Total time: {elapsed:.1f}s")

    # --- Output ---
    print_results(all_results, common_tickers)

    if not args.no_web:
        has_results = any(d["results"] for d in all_data.values())
        if has_results:
            run_web(all_data, elapsed, len(common_tickers), len(all_screeners))
        else:
            print("[*] No results to display in web UI")


if __name__ == "__main__":
    main()
