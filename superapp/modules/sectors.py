"""Sector ETF Dashboard blueprint — re-uses all sector_dashboard.py logic."""

import os

from flask import Blueprint, jsonify, request
import sector_dashboard as sd

LABEL   = "Sector Dashboard"
ICON    = "🏭"
SECTION = "Markets"
ORDER   = 2

PREFIX = "/sectors"
bp = Blueprint("sectors", __name__, url_prefix=PREFIX)

# ── Serve the dashboard HTML with API paths rewritten ─────────────────────────
_TMPL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates", "sector_dashboard.html",
)
with open(_TMPL_PATH) as _f:
    _HTML = _f.read()

# The template uses `const API_BASE = '';` — just set it to our prefix.
_HTML = _HTML.replace("const API_BASE = '';", f"const API_BASE = '{PREFIX}';")


@bp.route("/")
def index():
    return _HTML


# ── API routes (delegate to sector_dashboard functions) ───────────────────────

@bp.route("/api/etfs")
def api_etfs():
    return jsonify([{"ticker": t, "description": d} for t, d in sd.ETF_REGISTRY.items()])


@bp.route("/api/chart/<ticker>")
def api_chart_data(ticker):
    interval     = request.args.get("interval", "1d")
    range_period = request.args.get("range", "1y")
    try:
        records = sd._fetch_mboum(ticker, interval)
        days    = sd.RANGE_DAYS.get(range_period, 365)
        if days < 999999:
            from datetime import datetime, timedelta
            cutoff  = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            records = [r for r in records if r["time"] >= cutoff]
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/api/holdings")
def api_holdings():
    return jsonify({t: sd._fetch_holdings(t) for t in sd.ETF_REGISTRY})


@bp.route("/api/intl-etfs")
def api_intl_etfs():
    return jsonify([{"ticker": t, "description": d} for t, d in sd.INTL_REGISTRY.items()])


@bp.route("/api/intl-holdings")
def api_intl_holdings():
    return jsonify({t: sd._fetch_holdings(t) for t in sd.INTL_REGISTRY})


@bp.route("/api/signals")
def api_signals():
    import time as _time
    now = _time.time()
    if (sd._signals_cache["data"] is not None
            and (now - sd._signals_cache["fetched_at"]) < sd.SIGNALS_CACHE_TTL):
        return jsonify(sd._signals_cache["data"])
    try:
        try:
            spy_records = sd._fetch_mboum("SPY", "1d")
            spy_closes  = [r["close"] for r in spy_records]
        except Exception:
            spy_closes = []

        results = []
        for ticker, desc in sd.ETF_REGISTRY.items():
            sig = sd._compute_signals_for_etf(ticker, desc, spy_closes)
            if sig:
                sig["group"] = "sector"
                results.append(sig)
        for ticker, desc in sd.INTL_REGISTRY.items():
            sig = sd._compute_signals_for_etf(ticker, desc, spy_closes)
            if sig:
                sig["group"] = "intl"
                results.append(sig)

        risk_on  = [r for r in results if r["risk_class"] == "risk-on"]
        risk_off = [r for r in results if r["risk_class"] == "risk-off"]
        ro_breadth  = (sum(1 for r in risk_on  if r["momentum"] >= 5.5) / len(risk_on)  * 100) if risk_on  else 0
        rf_breadth  = (sum(1 for r in risk_off if r["momentum"] >= 5.5) / len(risk_off) * 100) if risk_off else 0
        ro_avg_mom  = round(sum(r["momentum"] for r in risk_on)  / len(risk_on),  1) if risk_on  else 0
        rf_avg_mom  = round(sum(r["momentum"] for r in risk_off) / len(risk_off), 1) if risk_off else 0
        accum_count   = sum(1 for r in results if r["flow"] == "Accum")
        distrib_count = sum(1 for r in results if r["flow"] == "Distrib")

        if   ro_breadth > 50 and rf_breadth < 50: regime_label = "RISK-ON"
        elif rf_breadth > 50 and ro_breadth < 50: regime_label = "RISK-OFF"
        elif ro_breadth < 40 and rf_breadth < 40: regime_label = "LIQUIDATION"
        else:                                      regime_label = "MIXED"

        regime = {
            "label":              regime_label,
            "risk_on_avg_mom":    ro_avg_mom,
            "risk_off_avg_mom":   rf_avg_mom,
            "risk_on_breadth":    round(ro_breadth, 1),
            "risk_off_breadth":   round(rf_breadth, 1),
            "accum_count":        accum_count,
            "distrib_count":      distrib_count,
        }
        payload = {"regime": regime, "etfs": results}
        sd._signals_cache["data"]       = payload
        sd._signals_cache["fetched_at"] = now
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
