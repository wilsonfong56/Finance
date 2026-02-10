#!/usr/bin/env python3
"""
Sector ETF Dashboard — view all 35 sector ETFs in one place.
Charts with 8/21 EMAs + top 10 holdings per ETF.
Runs on port 5051, separate from the main Finance Dashboard (5050).
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import requests
import yfinance as yf

app = Flask(__name__)
CORS(app)

# ── Mboum API (same key as main app) ────────────────────────────────────────
MBOUM_KEY = os.environ.get("MBOUM_KEY", "")
_chart_cache = {}  # (ticker, interval) -> {"data": [...], "fetched_at": ts}
CHART_CACHE_TTL = 900  # 15 minutes (matches Mboum API delay)
_signals_cache = {"data": None, "fetched_at": 0}
SIGNALS_CACHE_TTL = 900  # 15 minutes

RANGE_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825, "max": 999999,
}

# ── ETF Registry ────────────────────────────────────────────────────────────
ETF_REGISTRY = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XHB": "Homebuilders",
    "XME": "Metals & Mining",
    "XOP": "Oil & Gas Exploration",
    "XRT": "Retail",
    "KBE": "Banks",
    "KRE": "Regional Banks",
    "IBB": "Biotech",
    "IYR": "Real Estate",
    "IYT": "Transportation",
    "ITA": "Aerospace & Defense",
    "IGV": "Software",
    "SMH": "Semiconductors",
    "GDX": "Gold Miners",
    "SLV": "Silver",
    "GLD": "Gold",
    "URA": "Uranium",
    "TAN": "Solar Energy",
    "ARKK": "Innovation (ARK)",
    "HACK": "Cybersecurity",
    "JETS": "Airlines",
    "PAVE": "Infrastructure Development",
    "COPX": "Copper Miners",
    "LIT": "Lithium & Battery Tech",
    "BITO": "Bitcoin Strategy",
}

# ── International ETF Registry ─────────────────────────────────────────────
INTL_REGISTRY = {
    "EWJ": "Japan",
    "KWEB": "China Internet",
    "MCHI": "China Large-Cap",
    "EWZ": "Brazil",
    "INDA": "India",
    "EWT": "Taiwan",
    "EFA": "Developed Markets ex-US",
    "EEM": "Emerging Markets",
    "EWG": "Germany",
    "EWY": "South Korea",
}

# ── Live Holdings via yfinance (cached 24h) ──────────────────────────────────
_holdings_cache = {}  # ticker -> {"data": [...], "fetched_at": timestamp}
HOLDINGS_CACHE_TTL = 86400  # 24 hours

# Single-asset ETFs where yfinance has no holdings data
_SINGLE_ASSET_HOLDINGS = {
    "GLD": [{"ticker": "Gold", "name": "Physical Gold Bullion", "weight": 100.0}],
    "SLV": [{"ticker": "Silver", "name": "Physical Silver Bullion", "weight": 100.0}],
    "BITO": [{"ticker": "BTC", "name": "Bitcoin Futures (CME)", "weight": 100.0}],
}

# ── Risk Classification ─────────────────────────────────────────────────────
RISK_CLASS = {}
for _t in ("XLC", "XLY", "XLK", "XHB", "XRT", "IBB", "IGV", "SMH", "ARKK",
           "HACK", "JETS", "LIT", "BITO", "TAN", "XME", "XOP", "COPX", "URA",
           "KWEB", "EWT", "EWY", "EWZ", "INDA"):
    RISK_CLASS[_t] = "risk-on"
for _t in ("XLP", "XLU", "XLV", "GLD", "SLV", "GDX", "IYR", "EWJ", "EWG"):
    RISK_CLASS[_t] = "risk-off"
for _t in ("XLB", "XLE", "XLF", "XLI", "IYT", "ITA", "KBE", "KRE", "PAVE",
           "MCHI", "EFA", "EEM"):
    RISK_CLASS[_t] = "neutral"


def _fetch_holdings(etf_ticker):
    """Fetch top holdings for an ETF from Yahoo Finance, with 24h cache."""
    if etf_ticker in _SINGLE_ASSET_HOLDINGS:
        return _SINGLE_ASSET_HOLDINGS[etf_ticker]

    cached = _holdings_cache.get(etf_ticker)
    if cached and (time.time() - cached["fetched_at"]) < HOLDINGS_CACHE_TTL:
        return cached["data"]

    try:
        etf = yf.Ticker(etf_ticker)
        df = etf.funds_data.top_holdings
        holdings = []
        for symbol, row in df.iterrows():
            holdings.append({
                "ticker": symbol,
                "name": row["Name"],
                "weight": round(row["Holding Percent"] * 100, 2),
            })
        _holdings_cache[etf_ticker] = {"data": holdings, "fetched_at": time.time()}
        return holdings
    except Exception:
        # Return cached data even if stale, or empty list
        if cached:
            return cached["data"]
        return []


# ── Mboum chart data fetcher (copied from app.py) ──────────────────────────

def _fetch_mboum(ticker, interval):
    """Fetch all history from mboum for a ticker+interval, with simple cache."""
    cache_key = (ticker.upper(), interval)
    cached = _chart_cache.get(cache_key)
    if cached and (time.time() - cached["fetched_at"]) < CHART_CACHE_TTL:
        return cached["data"]

    resp = requests.get(
        "https://api.mboum.com/v1/markets/stock/history",
        params={
            "symbol": ticker.upper(),
            "interval": interval,
            "diffandsplits": "false",
            "apikey": MBOUM_KEY,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    body = data.get("body", {})
    records = []
    for key, val in body.items():
        if key != "events" and isinstance(val, dict) and "date" in val:
            o, h, l, c = val.get("open"), val.get("high"), val.get("low"), val.get("close")
            if o is None or h is None or l is None or c is None:
                continue
            records.append({
                "time": val["date"],
                "open": o, "high": h, "low": l, "close": c,
                "volume": val.get("volume", 0),
            })
    records.sort(key=lambda r: r["time"])
    _chart_cache[cache_key] = {"data": records, "fetched_at": time.time()}
    return records


# ── Signal indicator helpers ────────────────────────────────────────────────

def _calc_rsi(closes, period=14):
    """Wilder's RSI: SMA seed then exponential smoothing."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_ema(closes, period):
    """EMA matching the frontend calcEMA."""
    k = 2 / (period + 1)
    ema = [closes[0]]
    for i in range(1, len(closes)):
        ema.append(closes[i] * k + ema[i - 1] * (1 - k))
    return ema


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _linear_score(val, lo, hi):
    """Map val from [lo, hi] → [1, 10], clamped."""
    if hi == lo:
        return 5.0
    return _clamp(1 + 9 * (val - lo) / (hi - lo), 1, 10)


def _calc_cmf(records, period=20):
    """Chaikin Money Flow over last `period` bars.

    MF Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    MF Volume     = Multiplier × Volume
    CMF           = Sum(MF Volume, N) / Sum(Volume, N)
    Returns float in [-1, +1].
    """
    recent = records[-period:]
    mf_vol_sum = 0.0
    vol_sum = 0.0
    for r in recent:
        hl = r["high"] - r["low"]
        if hl == 0:
            continue
        mf_mult = ((r["close"] - r["low"]) - (r["high"] - r["close"])) / hl
        mf_vol_sum += mf_mult * r["volume"]
        vol_sum += r["volume"]
    return mf_vol_sum / vol_sum if vol_sum > 0 else 0.0


def _calc_mansfield_rs(etf_closes, spy_closes, sma_period=50):
    """Mansfield Relative Strength: RS ratio vs its SMA.

    RS ratio = ETF / SPY (daily).  Mansfield = (ratio / SMA(ratio) - 1) * 100.
    Positive = outperforming SPY, negative = underperforming.
    """
    n = min(len(etf_closes), len(spy_closes))
    if n < sma_period + 1:
        return 0.0
    # Align from the end (most recent bars match)
    ec = etf_closes[-n:]
    sc = spy_closes[-n:]
    ratios = [ec[i] / sc[i] for i in range(n) if sc[i] != 0]
    if len(ratios) < sma_period + 1:
        return 0.0
    sma = sum(ratios[-sma_period:]) / sma_period
    if sma == 0:
        return 0.0
    return (ratios[-1] / sma - 1) * 100


def _compute_signals_for_etf(ticker, description, spy_closes):
    """Compute momentum, CMF, Mansfield RS for one ETF."""
    try:
        records = _fetch_mboum(ticker, "1d")
        if not records or len(records) < 50:
            return None

        closes = [r["close"] for r in records]

        price = closes[-1]
        rsi = _calc_rsi(closes, 14)
        ema21 = _calc_ema(closes, 21)

        # ── Momentum (1-10, higher = stronger uptrend) ──
        if rsi < 30:
            rsi_score = _linear_score(rsi, 0, 30) * 2 / 10
            rsi_score = _clamp(rsi_score, 1, 2)
        elif rsi < 50:
            rsi_score = 3 + (rsi - 30) / 20
        elif rsi < 70:
            rsi_score = 5 + 2 * (rsi - 50) / 20
        else:
            rsi_score = 7 + 3 * (rsi - 70) / 30
        rsi_score = _clamp(rsi_score, 1, 10)

        pct_from_ema = (price - ema21[-1]) / ema21[-1] * 100
        ema_score = _linear_score(pct_from_ema, -10, 10)

        idx_1m = max(0, len(closes) - 22)
        ret_1m = (price - closes[idx_1m]) / closes[idx_1m] * 100
        ret_score = _linear_score(ret_1m, -15, 15)

        momentum = round(rsi_score * 0.4 + ema_score * 0.3 + ret_score * 0.3, 1)

        # ── Chaikin Money Flow (volume-weighted accum/distrib) ──
        cmf = _calc_cmf(records, 20)
        cmf_score = round(_linear_score(cmf, -0.25, 0.25), 1)
        if cmf > 0.05:
            flow = "Accum"
        elif cmf < -0.05:
            flow = "Distrib"
        else:
            flow = "Neutral"

        # ── Mansfield Relative Strength vs SPY ──
        mrs = _calc_mansfield_rs(closes, spy_closes, 50)
        rs_score = round(_linear_score(mrs, -8, 8), 1)

        return {
            "ticker": ticker,
            "description": description,
            "price": round(price, 2),
            "change_1m": round(ret_1m, 2),
            "risk_class": RISK_CLASS.get(ticker, "neutral"),
            "momentum": momentum,
            "rs": rs_score,
            "cmf": round(cmf, 3),
            "cmf_score": cmf_score,
            "flow": flow,
        }
    except Exception as e:
        print(f"[signals] {ticker} failed: {e}")
        return None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("sector_dashboard.html")


@app.route("/api/etfs")
def api_etfs():
    """Return the ETF registry as a list."""
    return jsonify([
        {"ticker": t, "description": d}
        for t, d in ETF_REGISTRY.items()
    ])


@app.route("/api/chart/<ticker>")
def api_chart_data(ticker):
    """Fetch OHLCV data for charting, filtered by range."""
    interval = request.args.get("interval", "1d")
    range_period = request.args.get("range", "1y")
    try:
        records = _fetch_mboum(ticker, interval)

        # Filter by range
        days = RANGE_DAYS.get(range_period, 365)
        if days < 999999:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            records = [r for r in records if r["time"] >= cutoff]

        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/holdings")
def api_holdings():
    """Return all holdings for all ETFs (live from Yahoo Finance)."""
    result = {}
    for ticker in ETF_REGISTRY:
        result[ticker] = _fetch_holdings(ticker)
    return jsonify(result)


@app.route("/api/intl-etfs")
def api_intl_etfs():
    """Return the international ETF registry as a list."""
    return jsonify([
        {"ticker": t, "description": d}
        for t, d in INTL_REGISTRY.items()
    ])


@app.route("/api/intl-holdings")
def api_intl_holdings():
    """Return all holdings for international ETFs (live from Yahoo Finance)."""
    result = {}
    for ticker in INTL_REGISTRY:
        result[ticker] = _fetch_holdings(ticker)
    return jsonify(result)


@app.route("/api/signals")
def api_signals():
    """Return rotation signals + regime summary for all ETFs."""
    now = time.time()
    if _signals_cache["data"] is not None and (now - _signals_cache["fetched_at"]) < SIGNALS_CACHE_TTL:
        return jsonify(_signals_cache["data"])

    try:
        # Fetch SPY benchmark once for Mansfield RS
        try:
            spy_records = _fetch_mboum("SPY", "1d")
            spy_closes = [r["close"] for r in spy_records]
        except Exception:
            spy_closes = []

        results = []
        for ticker, desc in ETF_REGISTRY.items():
            sig = _compute_signals_for_etf(ticker, desc, spy_closes)
            if sig:
                sig["group"] = "sector"
                results.append(sig)
        for ticker, desc in INTL_REGISTRY.items():
            sig = _compute_signals_for_etf(ticker, desc, spy_closes)
            if sig:
                sig["group"] = "intl"
                results.append(sig)

        # ── Regime summary ──
        risk_on = [r for r in results if r["risk_class"] == "risk-on"]
        risk_off = [r for r in results if r["risk_class"] == "risk-off"]

        ro_breadth = (sum(1 for r in risk_on if r["momentum"] >= 5.5) / len(risk_on) * 100) if risk_on else 0
        rf_breadth = (sum(1 for r in risk_off if r["momentum"] >= 5.5) / len(risk_off) * 100) if risk_off else 0
        ro_avg_mom = round(sum(r["momentum"] for r in risk_on) / len(risk_on), 1) if risk_on else 0
        rf_avg_mom = round(sum(r["momentum"] for r in risk_off) / len(risk_off), 1) if risk_off else 0
        accum_count = sum(1 for r in results if r["flow"] == "Accum")
        distrib_count = sum(1 for r in results if r["flow"] == "Distrib")

        if ro_breadth > 50 and rf_breadth < 50:
            regime_label = "RISK-ON"
        elif rf_breadth > 50 and ro_breadth < 50:
            regime_label = "RISK-OFF"
        elif ro_breadth < 40 and rf_breadth < 40:
            regime_label = "LIQUIDATION"
        else:
            regime_label = "MIXED"

        regime = {
            "label": regime_label,
            "risk_on_avg_mom": ro_avg_mom,
            "risk_off_avg_mom": rf_avg_mom,
            "risk_on_breadth": round(ro_breadth, 1),
            "risk_off_breadth": round(rf_breadth, 1),
            "accum_count": accum_count,
            "distrib_count": distrib_count,
        }

        payload = {"regime": regime, "etfs": results}
        _signals_cache["data"] = payload
        _signals_cache["fetched_at"] = now
        return jsonify(payload)
    except Exception as e:
        print(f"[signals] endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Sector ETF Dashboard on http://localhost:5051")
    app.run(debug=True, port=5051)
