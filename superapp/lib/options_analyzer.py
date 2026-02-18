#!/usr/bin/env python3
"""
Options analyzer to find cheap, liquid options using IV analysis.
Uses CBOE free delayed quotes API (no API key needed).
Stores daily IV30 in a local SQLite DB to build IV Rank over time.
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
import json
import re
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path

CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DB_PATH = Path(__file__).parent / "iv_history.db"


# ── Database for IV history ──────────────────────────────────────────────────

def init_db():
    """Initialize SQLite database for IV30 history tracking."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS iv_history (
            symbol TEXT,
            date TEXT,
            iv30 REAL,
            price REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()
    return conn


def save_iv30(conn, symbol: str, iv30: float, price: float):
    """Save today's IV30 reading to the database."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        "INSERT OR REPLACE INTO iv_history (symbol, date, iv30, price) VALUES (?, ?, ?, ?)",
        (symbol.upper(), today, iv30, price)
    )
    conn.commit()


def get_iv_history(conn, symbol: str, days: int = 252) -> list:
    """Get IV30 history for a symbol (up to N trading days)."""
    rows = conn.execute(
        "SELECT date, iv30 FROM iv_history WHERE symbol = ? ORDER BY date DESC LIMIT ?",
        (symbol.upper(), days)
    ).fetchall()
    return rows


def calculate_iv_rank(current_iv: float, iv_history: list) -> float:
    """IV Rank: (Current - 52wk Low) / (52wk High - 52wk Low) * 100"""
    if len(iv_history) < 2:
        return None
    values = [row[1] for row in iv_history]
    iv_min, iv_max = min(values), max(values)
    if iv_max == iv_min:
        return 50.0
    return ((current_iv - iv_min) / (iv_max - iv_min)) * 100


def calculate_iv_percentile(current_iv: float, iv_history: list) -> float:
    """IV Percentile: % of days in past year where IV30 was below current."""
    if len(iv_history) < 2:
        return None
    values = [row[1] for row in iv_history]
    below = sum(1 for v in values if v < current_iv)
    return (below / len(values)) * 100


# ── CBOE API ─────────────────────────────────────────────────────────────────

def fetch_cboe_quote(ticker: str) -> dict:
    """Fetch stock quote with IV30 from CBOE."""
    url = f"{CBOE_BASE}/quotes/{ticker.upper()}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get("data", {})


def fetch_cboe_options(ticker: str) -> list:
    """Fetch full options chain from CBOE."""
    url = f"{CBOE_BASE}/options/{ticker.upper()}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    return data.get("options", [])


def parse_option_symbol(symbol: str) -> dict:
    """
    Parse OCC option symbol, e.g., AAPL260206C00277500
    Returns: {ticker, expiration, option_type, strike}
    """
    match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', symbol)
    if not match:
        return None
    ticker, exp_str, opt_type, strike_str = match.groups()
    expiration = datetime.strptime(exp_str, "%y%m%d").strftime("%Y-%m-%d")
    strike = int(strike_str) / 1000.0
    return {
        'ticker': ticker,
        'expiration': expiration,
        'option_type': 'call' if opt_type == 'C' else 'put',
        'strike': strike
    }


# ── Probability calculations ─────────────────────────────────────────────────

def prob_itm(option_type: str, spot: float, strike: float,
             tte: float, iv: float, rf: float = 0.045) -> float:
    """Probability of expiring ITM via Black-Scholes d2."""
    if tte <= 0 or iv <= 0:
        return 0.0
    d2 = (np.log(spot / strike) + (rf - 0.5 * iv**2) * tte) / (iv * np.sqrt(tte))
    if option_type == 'call':
        return norm.cdf(d2) * 100
    return norm.cdf(-d2) * 100


def prob_profit(option_type: str, spot: float, strike: float, premium: float,
                tte: float, iv: float, rf: float = 0.045) -> float:
    """Probability of profit for a long option position."""
    if premium <= 0:
        return 0.0
    if option_type == 'call':
        return prob_itm('call', spot, strike + premium, tte, iv, rf)
    return prob_itm('put', spot, strike - premium, tte, iv, rf)


# ── Main analysis ────────────────────────────────────────────────────────────

def is_monthly_expiration(date_str: str) -> bool:
    """Check if a date is the third Friday of its month (standard monthly expiration)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() != 4:  # Not a Friday
        return False
    # Third Friday falls on days 15-21
    return 15 <= dt.day <= 21


def analyze_options(ticker: str, min_volume: int = 1000, min_oi: int = 1000,
                    max_expirations: int = 6, min_dte: int = 0,
                    monthly_only: bool = False) -> dict:
    """
    Fetch and analyze the full options chain for a ticker.

    Returns dict with quote info, IV metrics, and filtered options DataFrame.
    """
    ticker = ticker.upper()

    # 1. Fetch quote
    print(f"Fetching quote for {ticker}...")
    quote = fetch_cboe_quote(ticker)
    if not quote:
        print(f"Could not fetch quote for {ticker}")
        return None

    current_price = quote.get("current_price", 0)
    iv30 = quote.get("iv30", 0)

    # 2. Save IV30 to local DB and compute IV Rank
    conn = init_db()
    save_iv30(conn, ticker, iv30, current_price)
    iv_hist = get_iv_history(conn, ticker, days=252)
    conn.close()

    iv_rank = calculate_iv_rank(iv30, iv_hist)
    iv_pctl = calculate_iv_percentile(iv30, iv_hist)

    # 3. Fetch options chain
    print(f"Fetching options chain...")
    raw_options = fetch_cboe_options(ticker)
    if not raw_options:
        print(f"No options data for {ticker}")
        return None

    # 4. Parse into DataFrame
    rows = []
    for opt in raw_options:
        parsed = parse_option_symbol(opt.get("option", ""))
        if not parsed:
            continue

        exp_dt = datetime.strptime(parsed['expiration'], "%Y-%m-%d")
        dte = (exp_dt - datetime.now()).days
        if dte < min_dte:
            continue
        tte = max(dte, 1) / 365.0

        iv_val = opt.get("iv", 0) or 0
        last_price = opt.get("last_trade_price", 0) or 0
        bid = opt.get("bid", 0) or 0
        ask = opt.get("ask", 0) or 0
        mid = (bid + ask) / 2 if (bid and ask) else last_price

        rows.append({
            'symbol': opt.get("option", ""),
            'type': parsed['option_type'],
            'expiration': parsed['expiration'],
            'dte': dte,
            'strike': parsed['strike'],
            'bid': bid,
            'ask': ask,
            'mid': round(mid, 2),
            'last': last_price,
            'volume': int(opt.get("volume", 0) or 0),
            'oi': int(opt.get("open_interest", 0) or 0),
            'iv': round(iv_val * 100, 2),
            'delta': opt.get("delta", 0) or 0,
            'gamma': opt.get("gamma", 0) or 0,
            'theta': opt.get("theta", 0) or 0,
            'vega': opt.get("vega", 0) or 0,
            'theo': opt.get("theo", 0) or 0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No options parsed")
        return None

    # 5. Filter to monthly expirations if requested
    if monthly_only:
        df = df[df['expiration'].apply(is_monthly_expiration)]
        if df.empty:
            print("No monthly expirations found")
            return None

    # 6. Get unique expirations and limit
    expirations = sorted(df['expiration'].unique())[:max_expirations]
    df = df[df['expiration'].isin(expirations)]

    # 7. Filter for liquidity
    df = df[(df['volume'] >= min_volume) | (df['oi'] >= min_oi)]

    # 8. Calculate extra metrics
    df['ivVsIV30'] = (df['iv'] - iv30).round(2)  # per-strike IV vs overall IV30
    df['moneyness'] = ((df['strike'] - current_price) / current_price * 100).round(1)

    df['probITM'] = df.apply(
        lambda r: round(prob_itm(r['type'], current_price, r['strike'],
                                  r['dte'] / 365.0, r['iv'] / 100.0), 1)
        if r['iv'] > 0 else abs(r['delta']) * 100,
        axis=1
    )

    df['probProfit'] = df.apply(
        lambda r: round(prob_profit(r['type'], current_price, r['strike'], r['mid'],
                                     r['dte'] / 365.0, r['iv'] / 100.0), 1)
        if r['iv'] > 0 and r['mid'] > 0 else 0,
        axis=1
    )

    return {
        'ticker': ticker,
        'price': current_price,
        'iv30': iv30,
        'iv30_change': quote.get("iv30_change", 0),
        'iv_rank': iv_rank,
        'iv_percentile': iv_pctl,
        'iv_history_days': len(iv_hist),
        'expirations': expirations,
        'options': df,
    }


def print_table(df: pd.DataFrame, title: str):
    """Print a formatted options table."""
    print(f"\n{'='*110}")
    print(title)
    print(f"{'='*110}")
    if df.empty:
        print("No options match the criteria")
        return
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(df.to_string(index=False))


def main():
    print("=" * 60)
    print("OPTIONS ANALYZER (CBOE Data - No API Key Needed)")
    print("=" * 60)

    ticker = input("\nEnter ticker symbol: ").strip().upper()
    if not ticker:
        print("No ticker provided")
        return

    min_dte_input = input("Minimum DTE [0]: ").strip()
    min_dte = int(min_dte_input) if min_dte_input else 0

    opt_type_input = input("Option type — (c)alls, (p)uts, or (b)oth [b]: ").strip().lower()
    if opt_type_input in ('c', 'calls'):
        opt_type_filter = 'call'
    elif opt_type_input in ('p', 'puts'):
        opt_type_filter = 'put'
    else:
        opt_type_filter = 'both'

    monthly_input = input("Monthly expirations only? (y/n) [n]: ").strip().lower()
    monthly_only = monthly_input in ('y', 'yes')

    liq_input = input("Apply liquidity filter for cheapest options? (y/n) [y]: ").strip().lower()
    cheap_liq = liq_input != 'n'

    print(f"\nFetching data for {ticker} (min {min_dte} DTE{', monthly only' if monthly_only else ''})...\n")

    try:
        data = analyze_options(ticker, min_dte=min_dte, monthly_only=monthly_only)
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        print("Make sure the ticker is valid and optionable.")
        return

    if not data:
        return

    df = data['options']

    # ── Header ──
    print(f"\n{'='*60}")
    print(f"  {data['ticker']}  |  ${data['price']:.2f}")
    print(f"{'='*60}")
    print(f"  IV30:           {data['iv30']:.1f}%  (chg: {data['iv30_change']:+.1f}%)")
    if data['iv_rank'] is not None:
        print(f"  IV Rank:        {data['iv_rank']:.1f}%")
        print(f"  IV Percentile:  {data['iv_percentile']:.1f}%")
    else:
        print(f"  IV Rank:        N/A (need more daily readings, currently {data['iv_history_days']} day(s))")
    print(f"  Expirations:    {', '.join(data['expirations'])}")

    # ── Cheapest options (lowest per-strike IV vs IV30) ──
    cheap = df[df['iv'] > 0].copy()
    if opt_type_filter != 'both':
        cheap = cheap[cheap['type'] == opt_type_filter]
    if cheap_liq:
        cheap = cheap[(cheap['volume'] >= 1000) | (cheap['oi'] >= 1000)]
    cheap = cheap.sort_values('ivVsIV30').head(20)
    display_cols = ['type', 'strike', 'expiration', 'dte', 'bid', 'ask', 'mid',
                    'volume', 'oi', 'iv', 'ivVsIV30', 'delta', 'probITM', 'probProfit']
    liq_label = "" if cheap_liq else " | No liquidity filter"
    type_label = opt_type_filter.upper() + "S" if opt_type_filter != 'both' else "ALL"
    print_table(cheap[display_cols],
                f"CHEAPEST OPTIONS — {type_label} (Lowest IV vs {data['iv30']:.1f}% IV30{liq_label})")

    # ── Most liquid ──
    liquid = df[(df['volume'] >= 200) & (df['oi'] >= 500)].copy()
    liquid = liquid.sort_values('volume', ascending=False).head(20)
    liquid_cols = ['type', 'strike', 'expiration', 'dte', 'bid', 'ask',
                   'volume', 'oi', 'iv', 'delta', 'probITM']
    print_table(liquid[liquid_cols], "MOST LIQUID OPTIONS")

    # ── Unusual options activity ──
    # Case 1: high volume relative to existing OI (>= 3x ratio, volume >= 1000)
    high_ratio = df[(df['volume'] >= 1000) & (df['oi'] > 0)].copy()
    high_ratio['vol_oi_ratio'] = (high_ratio['volume'] / high_ratio['oi']).round(1)
    high_ratio = high_ratio[high_ratio['vol_oi_ratio'] >= 3]
    # Case 2: zero OI but significant volume (brand new positions)
    zero_oi = df[(df['volume'] >= 2000) & (df['oi'] == 0)].copy()
    zero_oi['vol_oi_ratio'] = float('inf')
    unusual = pd.concat([high_ratio, zero_oi]).sort_values('volume', ascending=False).head(20)
    unusual_cols = ['type', 'strike', 'expiration', 'dte', 'bid', 'ask', 'mid',
                    'volume', 'oi', 'vol_oi_ratio', 'iv', 'delta', 'probITM']
    print_table(unusual[unusual_cols], "UNUSUAL OPTIONS ACTIVITY (Volume >= 3x Open Interest)")

    # ── Summary ──
    calls = df[df['type'] == 'call']
    puts = df[df['type'] == 'put']
    avg_call_iv = calls[calls['iv'] > 0]['iv'].mean() if len(calls[calls['iv'] > 0]) else 0
    avg_put_iv = puts[puts['iv'] > 0]['iv'].mean() if len(puts[puts['iv'] > 0]) else 0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Avg Call IV:  {avg_call_iv:.1f}%")
    print(f"  Avg Put IV:   {avg_put_iv:.1f}%")
    print(f"  IV30:         {data['iv30']:.1f}%")
    print(f"  Skew (P-C):   {avg_put_iv - avg_call_iv:+.1f}%")
    status = "EXPENSIVE" if avg_call_iv > data['iv30'] * 1.1 else "CHEAP" if avg_call_iv < data['iv30'] * 0.9 else "FAIR"
    print(f"  Assessment:   {status}")
    print()
    print("  Columns:")
    print("    iv       = per-contract implied volatility")
    print("    ivVsIV30 = contract IV minus overall IV30 (negative = cheap vs market)")
    print("    delta    = CBOE-provided delta")
    print("    probITM  = probability of expiring in-the-money")
    print("    probProfit = probability of profit if bought at mid price")
    print()
    print(f"  IV Rank builds over time — run daily to accumulate history.")
    print(f"  Current IV history: {data['iv_history_days']} day(s) stored in iv_history.db")


if __name__ == "__main__":
    main()
