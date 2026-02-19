#!/usr/bin/env python3
"""
Trade Journal
=============
Personal trade journal for equity/options swing trading.
Tracks entries, partial exits (scaling out), open positions,
and provides comprehensive performance analytics.

Usage:
  python trade_journal.py              # start on port 5070
  python trade_journal.py --port 5080  # custom port
"""

import argparse
import json
import os
import sqlite3
import threading
import webbrowser
from datetime import date, datetime

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.db")

DEFAULT_SETUP_TAGS = [
    "Wedge Pop", "EMA Flip", "Base Breakout", "High Tight Flag",
    "Momentum", "Earnings", "Other",
]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            asset_type  TEXT    NOT NULL DEFAULT 'stock',
            direction   TEXT    NOT NULL DEFAULT 'long',
            strike      REAL,
            expiry      TEXT,
            entry_price REAL    NOT NULL,
            quantity    REAL    NOT NULL,
            entry_date  TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'open',
            setup_tag   TEXT,
            entry_thesis TEXT,
            exit_notes  TEXT,
            account_size REAL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS exits (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL,
            exit_price  REAL    NOT NULL,
            exit_quantity REAL  NOT NULL,
            exit_date   TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS accounts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL UNIQUE,
            starting_size REAL    NOT NULL,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        );
    """)

    # Add missing columns to positions if needed
    cols = [r[1] for r in conn.execute("PRAGMA table_info(positions)").fetchall()]
    if "account_id" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN account_id INTEGER")
    if "option_type" not in cols:
        conn.execute("ALTER TABLE positions ADD COLUMN option_type TEXT")

    # Migrate existing positions without an account_id
    orphans = conn.execute(
        "SELECT id, account_size FROM positions WHERE account_id IS NULL AND account_size > 0"
    ).fetchall()
    if orphans:
        # Use the most common account_size as the starting balance for the Default account
        from collections import Counter
        sizes = [r["account_size"] for r in orphans]
        most_common_size = Counter(sizes).most_common(1)[0][0]
        conn.execute(
            "INSERT OR IGNORE INTO accounts (name, starting_size) VALUES (?, ?)",
            ("Default", most_common_size),
        )
        acct = conn.execute("SELECT id FROM accounts WHERE name='Default'").fetchone()
        conn.execute(
            "UPDATE positions SET account_id=? WHERE account_id IS NULL",
            (acct["id"],),
        )

    # Seed defaults
    cur = conn.execute("SELECT value FROM settings WHERE key='setup_tags'")
    if cur.fetchone() is None:
        conn.execute("INSERT INTO settings (key, value) VALUES ('setup_tags', ?)",
                     (json.dumps(DEFAULT_SETUP_TAGS),))
    # Clean up legacy account_size setting
    conn.execute("DELETE FROM settings WHERE key='account_size'")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Compute helpers
# ---------------------------------------------------------------------------

def enrich_position(row, exits_rows):
    """Add computed fields to a position dict."""
    p = dict(row)
    exits = [dict(e) for e in exits_rows]
    p["exits"] = exits

    # Options contracts = 100 shares per contract
    multiplier = 100 if p["asset_type"] == "option" else 1

    total_exit_qty = sum(e["exit_quantity"] for e in exits)
    p["exited_quantity"] = total_exit_qty
    p["remaining_quantity"] = p["quantity"] - total_exit_qty

    if total_exit_qty > 0:
        avg_exit = sum(e["exit_price"] * e["exit_quantity"] for e in exits) / total_exit_qty
        p["avg_exit_price"] = round(avg_exit, 4)

        if p["direction"] == "long":
            pct_pnl = (avg_exit - p["entry_price"]) / p["entry_price"] * 100
            dollar_pnl = (avg_exit - p["entry_price"]) * total_exit_qty * multiplier
        else:
            pct_pnl = (p["entry_price"] - avg_exit) / p["entry_price"] * 100
            dollar_pnl = (p["entry_price"] - avg_exit) * total_exit_qty * multiplier

        p["pct_pnl"] = round(pct_pnl, 2)
        p["dollar_pnl"] = round(dollar_pnl, 2)

        acct = p.get("account_size") or 0
        p["account_pnl"] = round(dollar_pnl / acct * 100, 2) if acct > 0 else 0

        # Days held: entry_date to latest exit_date
        try:
            entry_dt = datetime.strptime(p["entry_date"], "%Y-%m-%d")
            latest_exit = max(e["exit_date"] for e in exits)
            exit_dt = datetime.strptime(latest_exit, "%Y-%m-%d")
            p["days_held"] = (exit_dt - entry_dt).days
        except (ValueError, TypeError):
            p["days_held"] = 0
    else:
        p["avg_exit_price"] = None
        p["pct_pnl"] = 0
        p["dollar_pnl"] = 0
        p["account_pnl"] = 0
        p["days_held"] = 0

    # Position size: nominal $ and % of account
    p["position_size"] = round(p["entry_price"] * p["quantity"] * multiplier, 2)
    acct = p.get("account_size") or 0
    p["position_size_pct"] = round(p["position_size"] / acct * 100, 1) if acct > 0 else 0

    return p


def compute_stats(positions):
    """Compute aggregate stats from a list of enriched closed positions."""
    if not positions:
        return {
            "total_trades": 0, "winners": 0, "losers": 0, "breakeven": 0,
            "win_rate": 0, "avg_winner_pct": 0, "avg_loser_pct": 0,
            "avg_pnl_pct": 0, "total_dollar_pnl": 0, "avg_account_pnl": 0,
            "avg_days_held": 0, "current_streak": 0, "max_win_streak": 0,
            "max_loss_streak": 0, "profit_factor": 0,
        }

    winners = [p for p in positions if p["pct_pnl"] > 0]
    losers = [p for p in positions if p["pct_pnl"] < 0]
    breakeven = [p for p in positions if p["pct_pnl"] == 0]
    total = len(positions)

    avg_winner = sum(p["account_pnl"] for p in winners) / len(winners) if winners else 0
    avg_loser = sum(p["account_pnl"] for p in losers) / len(losers) if losers else 0

    total_dollar = sum(p["dollar_pnl"] for p in positions)
    avg_acct = sum(p["account_pnl"] for p in positions) / total

    gross_wins = sum(p["dollar_pnl"] for p in winners)
    gross_losses = abs(sum(p["dollar_pnl"] for p in losers))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else (
        float("inf") if gross_wins > 0 else 0
    )

    avg_days = sum(p["days_held"] for p in positions) / total

    # Streaks — sort by latest exit date
    sorted_pos = sorted(positions, key=lambda p: max(
        (e["exit_date"] for e in p["exits"]), default=p["entry_date"]
    ))
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    streak = 0
    last_type = None
    for p in sorted_pos:
        is_win = p["pct_pnl"] > 0
        if last_type is None:
            streak = 1
            last_type = is_win
        elif is_win == last_type:
            streak += 1
        else:
            streak = 1
            last_type = is_win
        if is_win:
            max_win_streak = max(max_win_streak, streak)
        else:
            max_loss_streak = max(max_loss_streak, streak)
    current_streak = streak if last_type else -streak

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "breakeven": len(breakeven),
        "win_rate": round(len(winners) / total * 100, 1),
        "avg_winner_pct": round(avg_winner, 2),
        "avg_loser_pct": round(avg_loser, 2),
        "avg_pnl_pct": round(avg_acct, 2),
        "total_dollar_pnl": round(total_dollar, 2),
        "avg_account_pnl": round(avg_acct, 2),
        "avg_days_held": round(avg_days, 1),
        "current_streak": current_streak,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999,
    }


def _get_positions(conn, status=None, account_id=None):
    """Fetch positions with optional status/account filters, enriched with exits."""
    sql = "SELECT * FROM positions WHERE 1=1"
    params = []
    if status:
        sql += " AND status=?"
        params.append(status)
    if account_id:
        sql += " AND account_id=?"
        params.append(int(account_id))
    sql += " ORDER BY entry_date DESC"
    rows = conn.execute(sql, params).fetchall()
    result = []
    for row in rows:
        exits = conn.execute("SELECT * FROM exits WHERE position_id=? ORDER BY exit_date",
                             (row["id"],)).fetchall()
        result.append(enrich_position(row, exits))
    return result


def _account_current_size(conn, account_id):
    """starting_size + sum(dollar_pnl) of closed positions for this account."""
    acct = conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
    if not acct:
        return 0
    closed = _get_positions(conn, status="closed", account_id=account_id)
    realized = sum(p["dollar_pnl"] for p in closed)
    return round(acct["starting_size"] + realized, 2)


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_positions(conn, ids):
    """Merge open positions into the first one — weighted avg price, summed qty.
    Returns (primary_id, error_string). Exits are re-attributed to the survivor.
    """
    positions = []
    for pid in ids:
        row = conn.execute(
            "SELECT * FROM positions WHERE id=? AND status='open'", (pid,)
        ).fetchone()
        if row:
            positions.append(dict(row))
    if len(positions) < 2:
        return None, "Need at least 2 open positions to merge"

    total_qty    = sum(p["quantity"] for p in positions)
    avg_price    = sum(p["entry_price"] * p["quantity"] for p in positions) / total_qty
    earliest_date = min(p["entry_date"] for p in positions)
    primary_id   = positions[0]["id"]

    conn.execute(
        "UPDATE positions SET entry_price=?, quantity=?, entry_date=? WHERE id=?",
        (round(avg_price, 4), total_qty, earliest_date, primary_id),
    )
    for pos in positions[1:]:
        conn.execute(
            "UPDATE exits SET position_id=? WHERE position_id=?",
            (primary_id, pos["id"]),
        )
        conn.execute("DELETE FROM positions WHERE id=?", (pos["id"],))
    conn.commit()
    return primary_id, None


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return _HTML


@app.route("/api/positions")
def api_positions():
    status = request.args.get("status", "all")
    account_id = request.args.get("account_id")
    conn = get_db()
    positions = _get_positions(conn, status=None if status == "all" else status, account_id=account_id)
    conn.close()
    return jsonify(positions)


@app.route("/api/position", methods=["POST"])
def api_create_position():
    data = request.get_json()
    account_id = data.get("account_id")
    if not account_id:
        return jsonify({"error": "account_id is required"}), 400
    conn = get_db()
    # Stamp current account balance at time of entry
    account_size = _account_current_size(conn, int(account_id))
    cur = conn.execute("""
        INSERT INTO positions (ticker, asset_type, direction, option_type, strike, expiry,
            entry_price, quantity, entry_date, setup_tag, entry_thesis, account_size, account_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["ticker"].upper().strip(),
        data.get("asset_type", "stock"),
        data.get("direction", "long"),
        data.get("option_type"),
        data.get("strike"),
        data.get("expiry"),
        float(data["entry_price"]),
        float(data["quantity"]),
        data.get("entry_date", date.today().isoformat()),
        data.get("setup_tag"),
        data.get("entry_thesis"),
        account_size,
        int(account_id),
    ))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return jsonify({"id": pid, "status": "created"}), 201


@app.route("/api/position/<int:pid>", methods=["PUT"])
def api_update_position(pid):
    data = request.get_json()
    conn = get_db()
    fields = []
    values = []
    for col in ("ticker", "asset_type", "direction", "option_type", "strike", "expiry",
                "entry_price", "quantity", "entry_date", "setup_tag",
                "entry_thesis", "exit_notes", "status"):
        if col in data:
            fields.append(f"{col}=?")
            values.append(data[col])
    if not fields:
        conn.close()
        return jsonify({"error": "No fields to update"}), 400
    values.append(pid)
    conn.execute(f"UPDATE positions SET {', '.join(fields)} WHERE id=?", values)
    conn.commit()
    conn.close()
    return jsonify({"status": "updated"})


@app.route("/api/position/<int:pid>", methods=["DELETE"])
def api_delete_position(pid):
    conn = get_db()
    conn.execute("DELETE FROM positions WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})


@app.route("/api/exit", methods=["POST"])
def api_create_exit():
    data = request.get_json()
    pid = int(data["position_id"])
    exit_qty = float(data["exit_quantity"])
    exit_price = float(data["exit_price"])
    exit_date = data.get("exit_date", date.today().isoformat())

    conn = get_db()
    pos = conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
    if not pos:
        conn.close()
        return jsonify({"error": "Position not found"}), 404

    # Check remaining qty
    exited = conn.execute("SELECT COALESCE(SUM(exit_quantity),0) as total FROM exits WHERE position_id=?",
                          (pid,)).fetchone()["total"]
    remaining = pos["quantity"] - exited
    if exit_qty > remaining + 0.0001:
        conn.close()
        return jsonify({"error": f"Exit qty {exit_qty} exceeds remaining {remaining}"}), 400

    conn.execute("INSERT INTO exits (position_id, exit_price, exit_quantity, exit_date) VALUES (?,?,?,?)",
                 (pid, exit_price, exit_qty, exit_date))

    # Auto-close if fully exited
    new_exited = exited + exit_qty
    if new_exited >= pos["quantity"] - 0.0001:
        conn.execute("UPDATE positions SET status='closed' WHERE id=?", (pid,))

    conn.commit()
    conn.close()
    return jsonify({"status": "exit_logged"}), 201


@app.route("/api/exit/<int:eid>", methods=["DELETE"])
def api_delete_exit(eid):
    conn = get_db()
    ex = conn.execute("SELECT * FROM exits WHERE id=?", (eid,)).fetchone()
    if not ex:
        conn.close()
        return jsonify({"error": "Exit not found"}), 404
    pid = ex["position_id"]
    conn.execute("DELETE FROM exits WHERE id=?", (eid,))
    # Re-check if position should reopen
    pos = conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
    if pos:
        exited = conn.execute("SELECT COALESCE(SUM(exit_quantity),0) as total FROM exits WHERE position_id=?",
                              (pid,)).fetchone()["total"]
        if exited < pos["quantity"] - 0.0001:
            conn.execute("UPDATE positions SET status='open' WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "exit_deleted"})


@app.route("/api/stats")
def api_stats():
    account_id = request.args.get("account_id")
    conn = get_db()
    positions = _get_positions(conn, status="closed", account_id=account_id)
    conn.close()

    overall = compute_stats(positions)

    # Per-setup breakdown
    by_setup = {}
    for p in positions:
        tag = p.get("setup_tag") or "Untagged"
        by_setup.setdefault(tag, []).append(p)
    setup_stats = {tag: compute_stats(ps) for tag, ps in by_setup.items()}

    return jsonify({"overall": overall, "by_setup": setup_stats})


@app.route("/api/stats/direction")
def api_stats_direction():
    account_id = request.args.get("account_id")
    conn = get_db()
    positions = _get_positions(conn, status="closed", account_id=account_id)
    conn.close()

    longs = [p for p in positions if p["direction"] == "long"]
    shorts = [p for p in positions if p["direction"] == "short"]
    return jsonify({"long": compute_stats(longs), "short": compute_stats(shorts)})


@app.route("/api/monthly-breakdown")
def api_monthly_breakdown():
    account_id = request.args.get("account_id")
    conn = get_db()
    positions = _get_positions(conn, status="closed", account_id=account_id)
    conn.close()

    by_month = {}
    for p in positions:
        if p["exits"]:
            last_exit = max(e["exit_date"] for e in p["exits"])
            month_key = last_exit[:7]  # YYYY-MM
        else:
            month_key = p["entry_date"][:7]
        by_month.setdefault(month_key, []).append(p)

    result = []
    for month in sorted(by_month.keys()):
        ps = by_month[month]
        stats = compute_stats(ps)
        stats["month"] = month
        result.append(stats)
    return jsonify(result)


@app.route("/api/equity-curve")
def api_equity_curve():
    """Return cumulative account P&L % series for closed trades, sorted by exit date."""
    period = request.args.get("period", "all")  # day, wtd, mtd, ytd, all
    account_id = request.args.get("account_id")
    conn = get_db()
    positions = _get_positions(conn, status="closed", account_id=account_id)
    conn.close()

    # Each trade's data point is at its last exit date
    points = []
    for p in positions:
        if not p["exits"]:
            continue
        last_exit = max(e["exit_date"] for e in p["exits"])
        points.append({"date": last_exit, "account_pnl": p["account_pnl"]})

    points.sort(key=lambda x: x["date"])

    # Filter by period
    today_str = date.today().isoformat()
    if period == "day":
        points = [pt for pt in points if pt["date"] == today_str]
    elif period == "wtd":
        # Week-to-date: Monday of current week
        d = date.today()
        monday = d.isoformat() if d.weekday() == 0 else (
            date.fromordinal(d.toordinal() - d.weekday()).isoformat()
        )
        points = [pt for pt in points if pt["date"] >= monday]
    elif period == "mtd":
        month_start = today_str[:8] + "01"
        points = [pt for pt in points if pt["date"] >= month_start]
    elif period == "ytd":
        year_start = today_str[:5] + "01-01"
        points = [pt for pt in points if pt["date"] >= year_start]

    # Build cumulative series
    cumulative = []
    running = 0.0
    for pt in points:
        running += pt["account_pnl"]
        cumulative.append({"date": pt["date"], "cumulative_pnl": round(running, 2)})

    return jsonify(cumulative)


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    conn = get_db()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    settings = {}
    for r in rows:
        if r["key"] == "setup_tags":
            settings[r["key"]] = json.loads(r["value"])
        else:
            settings[r["key"]] = r["value"]
    return jsonify(settings)


@app.route("/api/settings", methods=["PUT"])
def api_update_settings():
    data = request.get_json()
    conn = get_db()
    for key, value in data.items():
        val = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, val))
    conn.commit()
    conn.close()
    return jsonify({"status": "updated"})


@app.route("/api/accounts")
def api_list_accounts():
    conn = get_db()
    rows = conn.execute("SELECT * FROM accounts ORDER BY created_at").fetchall()
    accounts = []
    for row in rows:
        a = dict(row)
        a["current_size"] = _account_current_size(conn, row["id"])
        accounts.append(a)
    conn.close()
    return jsonify(accounts)


@app.route("/api/accounts", methods=["POST"])
def api_create_account():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    starting_size = data.get("starting_size")
    if not name or starting_size is None:
        return jsonify({"error": "name and starting_size required"}), 400
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO accounts (name, starting_size) VALUES (?, ?)",
            (name, float(starting_size)),
        )
        conn.commit()
        aid = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Account name already exists"}), 409
    conn.close()
    return jsonify({"id": aid, "status": "created"}), 201


@app.route("/api/positions/merge", methods=["POST"])
def api_merge_positions():
    data = request.get_json()
    ids  = data.get("ids", [])
    if len(ids) < 2:
        return jsonify({"error": "Need at least 2 positions"}), 400
    conn = get_db()
    primary_id, err = merge_positions(conn, ids)
    conn.close()
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"status": "merged", "id": primary_id})


@app.route("/api/accounts/<int:aid>", methods=["DELETE"])
def api_delete_account(aid):
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM positions WHERE account_id=?", (aid,)
    ).fetchone()["cnt"]
    if count > 0:
        conn.close()
        return jsonify({"error": f"Cannot delete — {count} position(s) reference this account"}), 409
    conn.execute("DELETE FROM accounts WHERE id=?", (aid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Journal</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --orange: #db6d28;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); min-height: 100vh; }
header { background: var(--surface); border-bottom: 1px solid var(--border);
         padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
header h1 { font-size: 18px; font-weight: 600; }
.tabs { display: flex; gap: 0; margin: 0 24px; border-bottom: 1px solid var(--border); }
.tab { padding: 10px 20px; cursor: pointer; color: var(--muted); font-size: 13px;
       font-weight: 600; border-bottom: 2px solid transparent; transition: all 0.2s;
       user-select: none; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
             padding: 14px 16px; }
.stat-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 6px; }
.stat-card .value { font-size: 22px; font-weight: 700; }
.stat-card .sub { font-size: 11px; color: var(--muted); margin-top: 4px; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 10px 12px; color: var(--muted); font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
     cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { color: var(--text); }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
tr:hover { background: rgba(88,166,255,0.04); }
.positive { color: var(--green); }
.negative { color: var(--red); }
.muted { color: var(--muted); }
.small { font-size: 11px; color: var(--muted); }
.ticker-cell { font-weight: 700; color: var(--accent); }
.badge { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px;
         border-radius: 4px; letter-spacing: 0.3px; }
.badge-long { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-short { background: rgba(248,81,73,0.15); color: var(--red); }
.badge-stock { background: rgba(88,166,255,0.15); color: var(--accent); }
.badge-option { background: rgba(210,153,34,0.15); color: var(--yellow); }
.badge-call { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-put { background: rgba(248,81,73,0.15); color: var(--red); }

/* Expand row */
.expand-row { display: none; }
.expand-row.active { display: table-row; }
.expand-row td { padding: 12px 16px; background: rgba(22,27,34,0.8); }
.expand-content { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.expand-section h4 { font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; }
.expand-section p { font-size: 13px; white-space: pre-wrap; }
.expand-section p img { max-width: 100%; border-radius: 4px; margin-top: 8px; display: block; }
.exit-list { list-style: none; }
.exit-list li { padding: 4px 0; display: flex; align-items: center; gap: 8px; font-size: 12px; }
.exit-list .del-exit { background: none; border: none; color: var(--red); cursor: pointer;
                       font-size: 11px; opacity: 0.6; }
.exit-list .del-exit:hover { opacity: 1; }

/* Buttons */
.btn { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface);
       color: var(--text); font-size: 12px; cursor: pointer; font-weight: 600; transition: all 0.15s; }
.btn:hover { border-color: var(--muted); }
.btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.btn-primary:hover { background: #4a9aef; }
.btn-danger { color: var(--red); }
.btn-danger:hover { background: rgba(248,81,73,0.1); border-color: var(--red); }
.btn-sm { padding: 4px 10px; font-size: 11px; }
.btn-group { display: flex; gap: 6px; }

/* Toggle button */
.toggle-btn { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
              background: var(--surface); color: var(--muted); font-size: 12px; cursor: pointer;
              font-weight: 600; transition: all 0.15s; }
.toggle-btn:hover { color: var(--text); border-color: var(--muted); }
.toggle-btn.active { color: var(--accent); border-color: var(--accent); background: rgba(88,166,255,0.1); }

/* FAB */
.fab { position: fixed; bottom: 28px; right: 28px; width: 52px; height: 52px; border-radius: 50%;
       background: var(--accent); color: #fff; border: none; font-size: 28px; cursor: pointer;
       box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: flex; align-items: center;
       justify-content: center; z-index: 50; transition: transform 0.15s; }
.fab:hover { transform: scale(1.1); }

/* Modal */
.modal-overlay { display: none; position: fixed; inset: 0; z-index: 100;
                 background: rgba(0,0,0,0.7); align-items: center; justify-content: center; }
.modal-overlay.active { display: flex; }
.modal { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
         width: 90vw; max-width: 560px; max-height: 90vh; overflow-y: auto; }
.modal-header { display: flex; align-items: center; justify-content: space-between;
                padding: 14px 20px; border-bottom: 1px solid var(--border); }
.modal-header h2 { font-size: 16px; font-weight: 700; }
.modal-close { background: none; border: none; color: var(--muted); font-size: 22px;
               cursor: pointer; padding: 0 4px; }
.modal-close:hover { color: var(--text); }
.modal-body { padding: 16px 20px; }
.form-group { margin-bottom: 14px; }
.form-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px;
                    text-transform: uppercase; letter-spacing: 0.4px; }
.form-group input, .form-group select, .form-group textarea {
  width: 100%; padding: 8px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 13px; font-family: inherit; }
.form-group input:focus, .form-group select:focus, .form-group textarea:focus {
  outline: none; border-color: var(--accent); }
.form-group textarea { resize: vertical; min-height: 60px; }
.rich-editor { min-height: 80px; padding: 8px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 13px; font-family: inherit; cursor: text;
  white-space: pre-wrap; word-wrap: break-word; outline: none; }
.rich-editor:focus { border-color: var(--accent); }
.rich-editor img { max-width: 100%; border-radius: 4px; margin-top: 8px; display: block; }
.form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.form-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.option-fields { display: none; }
.option-fields.active { display: grid; }
.modal-footer { padding: 12px 20px; border-top: 1px solid var(--border);
                display: flex; justify-content: flex-end; gap: 8px; }

/* Inline exit form */
.exit-form { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.exit-form input { width: 80px; padding: 4px 8px; background: var(--bg); border: 1px solid var(--border);
                   border-radius: 4px; color: var(--text); font-size: 12px; }
.exit-form input:focus { outline: none; border-color: var(--accent); }

/* Monthly bars */
.bar-chart { margin-top: 16px; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; font-size: 13px; }
.bar-label { width: 70px; color: var(--muted); font-size: 12px; text-align: right; flex-shrink: 0; }
.bar-track { flex: 1; height: 22px; background: var(--surface); border-radius: 4px; overflow: hidden;
             position: relative; }
.bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center;
            padding-left: 8px; font-size: 11px; font-weight: 600; min-width: fit-content; }
.bar-value { margin-left: 8px; font-size: 12px; font-weight: 600; white-space: nowrap; }

/* Settings */
.settings-section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
                    padding: 16px 20px; margin-bottom: 16px; }
.settings-section h3 { font-size: 14px; margin-bottom: 12px; }
.tag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.tag-item { display: flex; align-items: center; gap: 4px; background: var(--bg);
            border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px; font-size: 12px; }
.tag-item .del-tag { background: none; border: none; color: var(--red); cursor: pointer;
                     font-size: 14px; opacity: 0.5; padding: 0 2px; }
.tag-item .del-tag:hover { opacity: 1; }
.inline-add { display: flex; gap: 6px; }
.inline-add input { flex: 1; padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
                    border-radius: 6px; color: var(--text); font-size: 13px; }

/* Empty state */
.empty { color: var(--muted); padding: 40px; text-align: center; font-size: 14px; }

/* Direction breakdown panel */
.dir-panel { margin-top: 16px; display: none; }
.dir-panel.active { display: block; }
.dir-panel h3 { font-size: 14px; color: var(--muted); margin-bottom: 10px; }

/* Filter bar */
.filter-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.filter-bar select, .filter-bar input {
  padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 12px; }
</style>
</head>
<body>

<header>
  <h1>Trade Journal</h1>
  <select id="account-selector" style="padding:6px 10px;background:var(--bg);border:1px solid var(--border);
          border-radius:6px;color:var(--text);font-size:13px;min-width:180px" onchange="switchAccount()">
    <option value="">All Accounts</option>
  </select>
</header>

<div class="tabs">
  <div class="tab active" data-tab="dashboard">Dashboard</div>
  <div class="tab" data-tab="open">Open Positions</div>
  <div class="tab" data-tab="closed">Closed Trades</div>
  <div class="tab" data-tab="monthly">Monthly</div>
  <div class="tab" data-tab="settings">Settings</div>
</div>

<main>
  <!-- Dashboard -->
  <div class="tab-content active" id="tab-dashboard">
    <div id="dash-stats" class="stat-grid"></div>
    <div style="margin-bottom:12px;">
      <button class="toggle-btn" id="dir-toggle" onclick="toggleDirBreakdown()">Long / Short Breakdown</button>
    </div>
    <div class="dir-panel" id="dir-panel">
      <h3>Long</h3>
      <div id="dir-long-stats" class="stat-grid"></div>
      <h3 style="margin-top:12px">Short</h3>
      <div id="dir-short-stats" class="stat-grid"></div>
    </div>
    <div id="equity-curve-section" style="margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <h3 style="font-size:14px;color:var(--muted);margin:0">Equity Curve</h3>
        <div class="btn-group" id="eq-period-btns">
          <button class="toggle-btn btn-sm" data-period="day">Day</button>
          <button class="toggle-btn btn-sm" data-period="wtd">Week</button>
          <button class="toggle-btn btn-sm" data-period="mtd">Month</button>
          <button class="toggle-btn btn-sm" data-period="ytd">YTD</button>
          <button class="toggle-btn btn-sm active" data-period="all">All</button>
        </div>
      </div>
      <div id="equity-chart" style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;height:220px;position:relative"></div>
    </div>
    <div style="margin-bottom:12px;">
      <button class="toggle-btn" id="dir-toggle" onclick="toggleDirBreakdown()">Long / Short Breakdown</button>
    </div>
    <div class="dir-panel" id="dir-panel">
      <h3>Long</h3>
      <div id="dir-long-stats" class="stat-grid"></div>
      <h3 style="margin-top:12px">Short</h3>
      <div id="dir-short-stats" class="stat-grid"></div>
    </div>
    <h3 style="font-size:14px;margin-bottom:10px;color:var(--muted)">Per-Setup Breakdown</h3>
    <table id="setup-table">
      <thead><tr>
        <th>Setup</th><th>Trades</th><th>Win Rate</th><th>Avg W%</th><th>Avg L%</th>
        <th>Avg P&L%</th><th>$ Total</th><th>Profit Factor</th>
      </tr></thead>
      <tbody id="setup-tbody"></tbody>
    </table>
  </div>

  <!-- Open Positions -->
  <div class="tab-content" id="tab-open">
    <div id="open-content"></div>
  </div>

  <!-- Closed Trades -->
  <div class="tab-content" id="tab-closed">
    <div class="filter-bar">
      <select id="filter-setup" onchange="renderClosed()"><option value="">All Setups</option></select>
      <select id="filter-dir" onchange="renderClosed()">
        <option value="">All Directions</option>
        <option value="long">Long</option>
        <option value="short">Short</option>
      </select>
      <select id="filter-type" onchange="renderClosed()">
        <option value="">All Types</option>
        <option value="stock">Stock</option>
        <option value="option">Option</option>
      </select>
      <input type="text" id="filter-ticker" placeholder="Ticker..." onkeyup="renderClosed()" style="width:100px">
    </div>
    <div id="closed-content"></div>
  </div>

  <!-- Monthly -->
  <div class="tab-content" id="tab-monthly">
    <div id="monthly-content"></div>
  </div>

  <!-- Settings -->
  <div class="tab-content" id="tab-settings">
    <div class="settings-section">
      <h3>Accounts</h3>
      <table id="accounts-table" style="margin-bottom:12px">
        <thead><tr><th>Name</th><th>Starting Size</th><th>Current Size</th><th></th></tr></thead>
        <tbody id="accounts-tbody"></tbody>
      </table>
      <div class="inline-add" style="gap:8px">
        <input type="text" id="new-acct-name" placeholder="Account name..." style="flex:2">
        <input type="number" id="new-acct-size" placeholder="Starting size..." style="flex:1">
        <button class="btn btn-primary" onclick="createAccount()">Create</button>
      </div>
    </div>
    <div class="settings-section">
      <h3>Setup Tags</h3>
      <div class="tag-list" id="tag-list"></div>
      <div class="inline-add">
        <input type="text" id="new-tag-input" placeholder="New tag name...">
        <button class="btn" onclick="addTag()">Add</button>
      </div>
    </div>
    <div class="settings-section">
      <h3>Export</h3>
      <button class="btn" onclick="exportCSV()">Download Closed Trades CSV</button>
    </div>
  </div>
</main>

<!-- FAB -->
<button class="fab" onclick="openNewModal()" title="New Entry">+</button>

<!-- New Entry Modal -->
<div class="modal-overlay" id="entry-modal">
  <div class="modal">
    <div class="modal-header">
      <h2 id="modal-title">New Entry</h2>
      <button class="modal-close" onclick="closeEntryModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Account</label>
        <select id="f-account" required></select>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Ticker</label>
          <input type="text" id="f-ticker" placeholder="AAPL" autocomplete="off">
        </div>
        <div class="form-group">
          <label>Type</label>
          <select id="f-type" onchange="toggleOptionFields()">
            <option value="stock">Stock</option>
            <option value="option">Option</option>
          </select>
        </div>
      </div>
      <div class="form-row-3 option-fields" id="option-fields">
        <div class="form-group">
          <label>Call / Put</label>
          <select id="f-option-type">
            <option value="call">Call</option>
            <option value="put">Put</option>
          </select>
        </div>
        <div class="form-group">
          <label>Strike</label>
          <input type="number" id="f-strike" step="0.5">
        </div>
        <div class="form-group">
          <label>Expiry</label>
          <input type="date" id="f-expiry">
        </div>
      </div>
      <div class="form-row-3">
        <div class="form-group">
          <label>Direction</label>
          <select id="f-direction">
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </div>
        <div class="form-group">
          <label>Entry Price</label>
          <input type="number" id="f-price" step="0.01">
        </div>
        <div class="form-group">
          <label>Quantity</label>
          <input type="number" id="f-qty" step="1">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Entry Date</label>
          <input type="date" id="f-date">
        </div>
        <div class="form-group">
          <label>Setup Tag</label>
          <select id="f-tag"></select>
        </div>
      </div>
      <div class="form-group">
        <label>Entry Thesis</label>
        <div contenteditable="true" id="f-thesis" class="rich-editor"></div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="closeEntryModal()">Cancel</button>
      <button class="btn btn-primary" id="modal-submit-btn" onclick="submitEntry()">Create</button>
    </div>
  </div>
</div>

<!-- Edit Entry Modal -->
<div class="modal-overlay" id="edit-modal">
  <div class="modal">
    <div class="modal-header">
      <h2>Edit Position</h2>
      <button class="modal-close" onclick="closeEditModal()">&times;</button>
    </div>
    <div class="modal-body">
      <input type="hidden" id="e-id">
      <div class="form-group">
        <label>Setup Tag</label>
        <select id="e-tag"></select>
      </div>
      <div class="form-group">
        <label>Entry Thesis</label>
        <div contenteditable="true" id="e-thesis" class="rich-editor"></div>
      </div>
      <div class="form-group">
        <label>Exit Notes</label>
        <div contenteditable="true" id="e-notes" class="rich-editor"></div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn" onclick="closeEditModal()">Cancel</button>
      <button class="btn btn-primary" onclick="submitEdit()">Save</button>
    </div>
  </div>
</div>

<script>
// ── Rich editor helpers ──
function sanitizeRichContent(html) {
  var t = (html || '').trim();
  if (!t || t === '<br>' || t === '<br/>') return null;
  return t;
}

function initRichEditors() {
  document.querySelectorAll('.rich-editor').forEach(function(editor) {
    editor.addEventListener('paste', function(e) {
      var items = (e.clipboardData || e.originalEvent.clipboardData).items;
      var hasImage = false;
      for (var i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          hasImage = true;
          e.preventDefault();
          var blob = items[i].getAsFile();
          var reader = new FileReader();
          reader.onload = function(ev) {
            var img = document.createElement('img');
            img.src = ev.target.result;
            img.style.maxWidth = '100%';
            img.style.borderRadius = '4px';
            img.style.marginTop = '8px';
            img.style.display = 'block';
            document.execCommand('insertHTML', false, img.outerHTML);
          };
          reader.readAsDataURL(blob);
          return;
        }
      }
      if (!hasImage) {
        e.preventDefault();
        var text = (e.clipboardData || e.originalEvent.clipboardData).getData('text/plain');
        document.execCommand('insertText', false, text);
      }
    });
  });
}

// ── State ──
var STATE = { settings: {}, positions: [], setupTags: [], accounts: [], activeAccountId: '' };
var SORT = { closed: { col: null, asc: true } };

// ── API helpers ──
async function api(url, opts) {
  var res = await fetch(url, opts);
  return res.json();
}
function apiGet(url) { return api(url); }
function apiPost(url, body) {
  return api(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
}
function apiPut(url, body) {
  return api(url, { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
}
function apiDelete(url) { return api(url, { method: 'DELETE' }); }

// ── Account param helper ──
function acctParam(prefix) {
  if (!STATE.activeAccountId) return '';
  var sep = prefix.indexOf('?') >= 0 ? '&' : '?';
  return sep + 'account_id=' + STATE.activeAccountId;
}

// ── Init ──
async function init() {
  STATE.settings = await apiGet('/api/settings');
  STATE.setupTags = STATE.settings.setup_tags || [];
  STATE.accounts = await apiGet('/api/accounts');
  renderAccountSelector();
  loadData();
  renderSettings();
}

function renderAccountSelector() {
  var sel = document.getElementById('account-selector');
  var html = '<option value="">All Accounts</option>';
  STATE.accounts.forEach(function(a) {
    html += '<option value="' + a.id + '"' + (STATE.activeAccountId == a.id ? ' selected' : '') + '>' +
      a.name + ' ($' + a.current_size.toLocaleString() + ')</option>';
  });
  sel.innerHTML = html;
}

function switchAccount() {
  STATE.activeAccountId = document.getElementById('account-selector').value;
  loadData();
}

async function loadData() {
  var url = '/api/positions?status=all' + acctParam('/api/positions?status=all');
  STATE.positions = await apiGet(url);
  var tab = document.querySelector('.tab.active').dataset.tab;
  if (tab === 'dashboard') renderDashboard();
  else if (tab === 'open') renderOpen();
  else if (tab === 'closed') renderClosed();
  else if (tab === 'monthly') renderMonthly();
}

// ── Tabs ──
document.querySelectorAll('.tab').forEach(function(t) {
  t.addEventListener('click', function() {
    document.querySelectorAll('.tab').forEach(function(x) { x.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(x) { x.classList.remove('active'); });
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    var tab = t.dataset.tab;
    if (tab === 'dashboard') renderDashboard();
    else if (tab === 'open') renderOpen();
    else if (tab === 'closed') renderClosed();
    else if (tab === 'monthly') renderMonthly();
    else if (tab === 'settings') renderSettings();
  });
});

// ── Format helpers ──
function pnlClass(v) { return v > 0 ? 'positive' : v < 0 ? 'negative' : 'muted'; }
function pnlSign(v) { return v > 0 ? '+' : ''; }
function fmt$(v) { return '$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtPct(v) { return pnlSign(v) + v.toFixed(2) + '%'; }
function today() { return new Date().toISOString().slice(0, 10); }

// ── Stat cards ──
function renderStatCards(container, stats) {
  var cards = [
    { label: 'Total Trades', value: stats.total_trades, sub: stats.winners + 'W / ' + stats.losers + 'L' },
    { label: 'Win Rate', value: stats.win_rate + '%', cls: stats.win_rate >= 50 ? 'positive' : 'negative' },
    { label: 'Avg Winner', value: fmtPct(stats.avg_winner_pct), cls: 'positive' },
    { label: 'Avg Loser', value: fmtPct(stats.avg_loser_pct), cls: 'negative' },
    { label: 'Avg P&L', value: fmtPct(stats.avg_account_pnl), cls: pnlClass(stats.avg_account_pnl) },
    { label: 'Total P&L', value: (stats.total_dollar_pnl >= 0 ? '+' : '-') + fmt$(stats.total_dollar_pnl),
      cls: pnlClass(stats.total_dollar_pnl) },
    { label: 'Avg Hold', value: stats.avg_days_held + 'd' },
    { label: 'Profit Factor', value: stats.profit_factor },
    { label: 'Streak', value: (stats.current_streak > 0 ? '+' : '') + stats.current_streak,
      sub: 'Max W:' + stats.max_win_streak + ' / L:' + stats.max_loss_streak,
      cls: stats.current_streak > 0 ? 'positive' : stats.current_streak < 0 ? 'negative' : '' },
  ];
  container.innerHTML = cards.map(function(c) {
    return '<div class="stat-card"><div class="label">' + c.label + '</div>' +
      '<div class="value ' + (c.cls || '') + '">' + c.value + '</div>' +
      (c.sub ? '<div class="sub">' + c.sub + '</div>' : '') + '</div>';
  }).join('');
}

// ── Dashboard ──
var eqPeriod = 'all';

async function renderDashboard() {
  var data = await apiGet('/api/stats' + acctParam('/api/stats'));
  renderStatCards(document.getElementById('dash-stats'), data.overall);
  renderEquityCurve();

  var tbody = document.getElementById('setup-tbody');
  var setups = data.by_setup;
  var keys = Object.keys(setups).sort();
  if (!keys.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted" style="text-align:center">No closed trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = keys.map(function(tag) {
    var s = setups[tag];
    return '<tr><td style="font-weight:600">' + tag + '</td><td>' + s.total_trades + '</td>' +
      '<td class="' + (s.win_rate >= 50 ? 'positive' : 'negative') + '">' + s.win_rate + '%</td>' +
      '<td class="positive">' + fmtPct(s.avg_winner_pct) + '</td>' +
      '<td class="negative">' + fmtPct(s.avg_loser_pct) + '</td>' +
      '<td class="' + pnlClass(s.avg_account_pnl) + '">' + fmtPct(s.avg_account_pnl) + '</td>' +
      '<td class="' + pnlClass(s.total_dollar_pnl) + '">' + (s.total_dollar_pnl >= 0 ? '+' : '-') + fmt$(s.total_dollar_pnl) + '</td>' +
      '<td>' + s.profit_factor + '</td></tr>';
  }).join('');
}

// ── Equity Curve ──
document.getElementById('eq-period-btns').addEventListener('click', function(e) {
  var btn = e.target.closest('.toggle-btn');
  if (!btn) return;
  eqPeriod = btn.dataset.period;
  document.querySelectorAll('#eq-period-btns .toggle-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.period === eqPeriod);
  });
  renderEquityCurve();
});

async function renderEquityCurve() {
  var url = '/api/equity-curve?period=' + eqPeriod + acctParam('/api/equity-curve?period=' + eqPeriod);
  var data = await apiGet(url);
  var container = document.getElementById('equity-chart');

  if (!data.length) {
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-size:13px">No closed trades in this period</div>';
    return;
  }

  // Chart dimensions
  var W = container.clientWidth - 32;
  var H = 188;
  var padL = 55, padR = 56, padT = 20, padB = 28;
  var cw = W - padL - padR;
  var ch = H - padT - padB;

  var vals = data.map(function(d) { return d.cumulative_pnl; });
  var minV = Math.min(0, Math.min.apply(null, vals));
  var maxV = Math.max(0, Math.max.apply(null, vals));
  if (minV === maxV) { minV -= 1; maxV += 1; }
  var range = maxV - minV;

  function xPos(i) { return padL + (data.length === 1 ? cw / 2 : i / (data.length - 1) * cw); }
  function yPos(v) { return padT + (1 - (v - minV) / range) * ch; }

  // Build SVG
  var zeroY = yPos(0);
  var pathParts = [];
  var areaParts = [];
  data.forEach(function(d, i) {
    var x = xPos(i), y = yPos(d.cumulative_pnl);
    pathParts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1));
    areaParts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1));
  });
  // Close area to zero line
  areaParts.push('L' + xPos(data.length - 1).toFixed(1) + ',' + zeroY.toFixed(1));
  areaParts.push('L' + xPos(0).toFixed(1) + ',' + zeroY.toFixed(1) + 'Z');

  var lastVal = vals[vals.length - 1];
  var lineColor = lastVal >= 0 ? '#3fb950' : '#f85149';
  var fillColor = lastVal >= 0 ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)';

  var svg = '<svg width="' + W + '" height="' + H + '" style="display:block">';

  // Grid lines + labels (5 horizontal lines)
  var steps = 4;
  for (var s = 0; s <= steps; s++) {
    var v = minV + (range * s / steps);
    var gy = yPos(v);
    svg += '<line x1="' + padL + '" y1="' + gy.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + gy.toFixed(1) +
      '" stroke="#1e252e" stroke-width="1"/>';
    svg += '<text x="' + (padL - 6) + '" y="' + (gy + 4).toFixed(1) +
      '" fill="#8b949e" font-size="10" text-anchor="end">' + v.toFixed(1) + '%</text>';
  }

  // Zero line
  if (minV < 0 && maxV > 0) {
    svg += '<line x1="' + padL + '" y1="' + zeroY.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + zeroY.toFixed(1) +
      '" stroke="#30363d" stroke-width="1" stroke-dasharray="4,3"/>';
  }

  // X-axis labels (show up to 8 dates)
  var labelStep = Math.max(1, Math.floor(data.length / 8));
  for (var i = 0; i < data.length; i += labelStep) {
    var lbl = data[i].date.slice(5); // MM-DD
    svg += '<text x="' + xPos(i).toFixed(1) + '" y="' + (H - 4) +
      '" fill="#8b949e" font-size="10" text-anchor="middle">' + lbl + '</text>';
  }

  // Area fill + line
  svg += '<path d="' + areaParts.join('') + '" fill="' + fillColor + '"/>';
  svg += '<path d="' + pathParts.join('') + '" fill="none" stroke="' + lineColor + '" stroke-width="2"/>';

  // Dots on data points (if few enough)
  if (data.length <= 30) {
    data.forEach(function(d, i) {
      svg += '<circle cx="' + xPos(i).toFixed(1) + '" cy="' + yPos(d.cumulative_pnl).toFixed(1) +
        '" r="3" fill="' + lineColor + '"/>';
    });
  }

  // End label
  var lastX = xPos(data.length - 1);
  var lastY = yPos(lastVal);
  svg += '<text x="' + (lastX + 6).toFixed(1) + '" y="' + (lastY + 4).toFixed(1) +
    '" fill="' + lineColor + '" font-size="11" font-weight="700">' + (lastVal >= 0 ? '+' : '') + lastVal.toFixed(2) + '%</text>';

  svg += '</svg>';
  container.innerHTML = svg;
}

async function toggleDirBreakdown() {
  var panel = document.getElementById('dir-panel');
  var btn = document.getElementById('dir-toggle');
  var showing = panel.classList.toggle('active');
  btn.classList.toggle('active', showing);
  if (showing) {
    var data = await apiGet('/api/stats/direction' + acctParam('/api/stats/direction'));
    renderStatCards(document.getElementById('dir-long-stats'), data.long);
    renderStatCards(document.getElementById('dir-short-stats'), data.short);
  }
}

// ── Direction badge helper ──
function dirBadge(p) {
  if (p.asset_type === 'option' && p.option_type) {
    return '<span class="badge badge-' + p.option_type + '">' + p.option_type.toUpperCase() + '</span>' +
           ' <span class="badge badge-' + p.direction + '" style="font-size:9px;padding:1px 5px">' + p.direction.toUpperCase() + '</span>';
  }
  return '<span class="badge badge-' + p.direction + '">' + p.direction.toUpperCase() + '</span>';
}

// ── Open Positions ──
function posKey(p) {
  var k = p.ticker + '|' + p.asset_type + '|' + p.direction + '|' + (p.account_id || '');
  if (p.asset_type === 'option') k += '|' + (p.option_type||'') + '|' + (p.strike||'') + '|' + (p.expiry||'');
  return k;
}

async function mergeGroup(ids) {
  if (!confirm('Merge ' + ids.length + ' positions into one using a weighted average entry price?')) return;
  var res = await apiPost('/api/positions/merge', { ids: ids });
  if (res.error) { alert('Merge failed: ' + res.error); return; }
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts  = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
}

function renderOpen() {
  var positions = STATE.positions.filter(function(p) { return p.status === 'open'; });
  var el = document.getElementById('open-content');
  if (!positions.length) {
    el.innerHTML = '<div class="empty">No open positions</div>';
    return;
  }

  // Detect duplicate groups (same ticker / type / direction / account)
  var groups = {};
  positions.forEach(function(p) {
    var k = posKey(p);
    if (!groups[k]) groups[k] = [];
    groups[k].push(p.id);
  });
  var dupeGroups = {}; // key -> [ids] only for groups with 2+
  Object.keys(groups).forEach(function(k) { if (groups[k].length > 1) dupeGroups[k] = groups[k]; });
  var hasDupes = Object.keys(dupeGroups).length > 0;

  var html = '';
  if (hasDupes) {
    html += '<div style="background:rgba(210,153,34,0.1);border:1px solid var(--yellow);border-radius:6px;' +
            'padding:10px 14px;margin-bottom:12px;font-size:13px;display:flex;align-items:center;gap:12px">' +
            '<span style="color:var(--yellow)">&#9888;</span>' +
            '<span style="color:var(--text)">Duplicate open positions detected — use the <strong>Merge</strong> button to combine them.</span>' +
            '</div>';
  }

  html += '<table><thead><tr>' +
    '<th>Ticker</th><th>Type</th><th>Dir</th><th>Avg Entry</th><th>Qty</th><th>Remaining</th>' +
    '<th>Size</th><th>Date</th><th>Setup</th><th>Log Exit</th><th></th>' +
    '</tr></thead><tbody>';
  positions.forEach(function(p) {
    var optInfo = p.asset_type === 'option'
      ? ' ' + (p.option_type ? p.option_type.toUpperCase() + ' ' : '') + (p.strike || '') + ' ' + (p.expiry || '')
      : '';
    var k = posKey(p);
    var isDupe = !!dupeGroups[k];
    var dupeIds = isDupe ? JSON.stringify(dupeGroups[k]) : '[]';
    html += '<tr' + (isDupe ? ' style="background:rgba(210,153,34,0.04)"' : '') + '>' +
      '<td class="ticker-cell">' + p.ticker + '<span class="small">' + optInfo + '</span></td>' +
      '<td><span class="badge badge-' + p.asset_type + '">' + p.asset_type.toUpperCase() + '</span></td>' +
      '<td>' + dirBadge(p) + '</td>' +
      '<td>$' + p.entry_price.toFixed(2) + '</td>' +
      '<td>' + p.quantity + '</td>' +
      '<td>' + p.remaining_quantity + '</td>' +
      '<td>$' + p.position_size.toLocaleString() + '<div class="small">' + p.position_size_pct.toFixed(1) + '% of acct</div></td>' +
      '<td>' + p.entry_date + '</td>' +
      '<td>' + (p.setup_tag || '<span class="muted">-</span>') + '</td>' +
      '<td><div class="exit-form">' +
        '<input type="number" step="0.01" placeholder="Price" id="ex-price-' + p.id + '">' +
        '<input type="number" step="1" placeholder="Qty" id="ex-qty-' + p.id + '" value="' + p.remaining_quantity + '">' +
        '<input type="date" id="ex-date-' + p.id + '" value="' + today() + '" style="width:120px">' +
        '<button class="btn btn-sm btn-primary" onclick="logExit(' + p.id + ')">Exit</button>' +
      '</div></td>' +
      '<td class="btn-group">' +
        (isDupe ? '<button class="btn btn-sm" style="color:var(--yellow);border-color:var(--yellow)" onclick="mergeGroup(' + dupeIds + ')" title="Merge duplicate positions">Merge</button>' : '') +
        '<button class="btn btn-sm" onclick="toggleAddRow(' + p.id + ')" title="Add to position" style="font-weight:700">+</button>' +
        '<button class="btn btn-sm" onclick="openEditModal(' + p.id + ')" title="Edit">&#9998;</button>' +
        '<button class="btn btn-sm btn-danger" onclick="deletePosition(' + p.id + ')" title="Delete">&#10005;</button>' +
      '</td></tr>';
    // Add-to-position inline form (hidden)
    html += '<tr id="add-row-' + p.id + '" style="display:none"><td colspan="11" style="padding:4px 12px 8px 24px;background:rgba(88,166,255,0.04)">' +
      '<div class="exit-form">' +
        '<span class="small" style="color:var(--muted);white-space:nowrap">Add shares:</span>' +
        '<input type="number" step="0.01" placeholder="Price" id="add-price-' + p.id + '">' +
        '<input type="number" step="1" placeholder="Qty" id="add-qty-' + p.id + '">' +
        '<input type="date" id="add-date-' + p.id + '" value="' + today() + '" style="width:120px">' +
        '<button class="btn btn-sm btn-primary" onclick="addToPosition(' + p.id + ')">Add</button>' +
        '<button class="btn btn-sm" onclick="toggleAddRow(' + p.id + ')">Cancel</button>' +
      '</div>' +
    '</td></tr>';
    // Show existing partial exits
    if (p.exits.length) {
      html += '<tr><td colspan="11" style="padding:4px 12px 8px 40px;border-bottom:1px solid var(--border)">' +
        '<span class="small">Exits: </span>';
      p.exits.forEach(function(ex) {
        html += '<span style="font-size:12px;margin-right:12px">' +
          ex.exit_quantity + ' @ $' + ex.exit_price.toFixed(2) + ' (' + ex.exit_date + ')' +
          ' <button class="exit-list del-exit" onclick="deleteExit(' + ex.id + ')" title="Undo">&#8634;</button>' +
          '</span>';
      });
      html += '</td></tr>';
    }
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

async function logExit(pid) {
  var price = parseFloat(document.getElementById('ex-price-' + pid).value);
  var qty = parseFloat(document.getElementById('ex-qty-' + pid).value);
  var dt = document.getElementById('ex-date-' + pid).value;
  if (!price || !qty) return;
  await apiPost('/api/exit', { position_id: pid, exit_price: price, exit_quantity: qty, exit_date: dt });
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
}

async function deleteExit(eid) {
  await apiDelete('/api/exit/' + eid);
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
}

async function deletePosition(pid) {
  if (!confirm('Delete this position and all its exits?')) return;
  await apiDelete('/api/position/' + pid);
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
  renderClosed();
}

function toggleAddRow(pid) {
  var row = document.getElementById('add-row-' + pid);
  if (!row) return;
  row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
}

async function addToPosition(pid) {
  var pos = STATE.positions.find(function(p) { return p.id === pid; });
  if (!pos) return;
  var addPrice = parseFloat(document.getElementById('add-price-' + pid).value);
  var addQty = parseFloat(document.getElementById('add-qty-' + pid).value);
  if (!addPrice || !addQty || addQty <= 0) return;
  var newQty = pos.quantity + addQty;
  var newAvg = (pos.entry_price * pos.quantity + addPrice * addQty) / newQty;
  await apiPut('/api/position/' + pid, { entry_price: parseFloat(newAvg.toFixed(4)), quantity: newQty });
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
}

// ── Closed Trades ──
function renderClosed() {
  var positions = STATE.positions.filter(function(p) { return p.status === 'closed'; });

  // Filters
  var fSetup = document.getElementById('filter-setup').value;
  var fDir = document.getElementById('filter-dir').value;
  var fType = document.getElementById('filter-type').value;
  var fTicker = document.getElementById('filter-ticker').value.toUpperCase().trim();

  if (fSetup) positions = positions.filter(function(p) { return p.setup_tag === fSetup; });
  if (fDir) positions = positions.filter(function(p) { return p.direction === fDir; });
  if (fType) positions = positions.filter(function(p) { return p.asset_type === fType; });
  if (fTicker) positions = positions.filter(function(p) { return p.ticker.indexOf(fTicker) >= 0; });

  // Sorting
  if (SORT.closed.col) {
    var col = SORT.closed.col;
    var asc = SORT.closed.asc;
    positions.sort(function(a, b) {
      var va = a[col], vb = b[col];
      if (va == null) va = '';
      if (vb == null) vb = '';
      if (typeof va === 'string') return asc ? va.localeCompare(vb) : vb.localeCompare(va);
      return asc ? va - vb : vb - va;
    });
  }

  // Populate setup filter options
  var setupSel = document.getElementById('filter-setup');
  var currentVal = setupSel.value;
  var allSetups = {};
  STATE.positions.forEach(function(p) { if (p.setup_tag) allSetups[p.setup_tag] = 1; });
  var opts = '<option value="">All Setups</option>';
  Object.keys(allSetups).sort().forEach(function(s) {
    opts += '<option value="' + s + '"' + (s === currentVal ? ' selected' : '') + '>' + s + '</option>';
  });
  setupSel.innerHTML = opts;

  var el = document.getElementById('closed-content');
  if (!positions.length) {
    el.innerHTML = '<div class="empty">No closed trades' + (fSetup || fDir || fType || fTicker ? ' matching filters' : '') + '</div>';
    return;
  }

  var arrow = function(col) {
    if (SORT.closed.col !== col) return '';
    return SORT.closed.asc ? ' &#9650;' : ' &#9660;';
  };

  var html = '<table><thead><tr>' +
    '<th onclick="sortClosed(&#39;ticker&#39;)">Ticker' + arrow('ticker') + '</th>' +
    '<th onclick="sortClosed(&#39;direction&#39;)">Dir' + arrow('direction') + '</th>' +
    '<th onclick="sortClosed(&#39;entry_price&#39;)">Entry' + arrow('entry_price') + '</th>' +
    '<th onclick="sortClosed(&#39;avg_exit_price&#39;)">Avg Exit' + arrow('avg_exit_price') + '</th>' +
    '<th onclick="sortClosed(&#39;quantity&#39;)">Qty' + arrow('quantity') + '</th>' +
    '<th onclick="sortClosed(&#39;position_size&#39;)">Size' + arrow('position_size') + '</th>' +
    '<th onclick="sortClosed(&#39;account_pnl&#39;)">P&L %' + arrow('account_pnl') + '</th>' +
    '<th onclick="sortClosed(&#39;dollar_pnl&#39;)">$ P&L' + arrow('dollar_pnl') + '</th>' +
    '<th onclick="sortClosed(&#39;days_held&#39;)">Days' + arrow('days_held') + '</th>' +
    '<th onclick="sortClosed(&#39;setup_tag&#39;)">Setup' + arrow('setup_tag') + '</th>' +
    '<th onclick="sortClosed(&#39;entry_date&#39;)">Date' + arrow('entry_date') + '</th>' +
    '<th></th>' +
    '</tr></thead><tbody>';

  positions.forEach(function(p) {
    var optInfo = p.asset_type === 'option'
      ? ' ' + (p.option_type ? p.option_type.toUpperCase() + ' ' : '') + (p.strike || '') + ' ' + (p.expiry || '')
      : '';
    html += '<tr style="cursor:pointer" onclick="toggleExpand(' + p.id + ', event)">' +
      '<td class="ticker-cell">' + p.ticker +
        '<span class="small">' + optInfo + '</span>' +
        (p.asset_type === 'option' ? ' <span class="badge badge-option">OPT</span>' : '') + '</td>' +
      '<td>' + dirBadge(p) + '</td>' +
      '<td>$' + p.entry_price.toFixed(2) + '</td>' +
      '<td>' + (p.avg_exit_price != null ? '$' + p.avg_exit_price.toFixed(2) : '-') + '</td>' +
      '<td>' + p.quantity + '</td>' +
      '<td>$' + p.position_size.toLocaleString() + '<div class="small">' + p.position_size_pct.toFixed(1) + '% of acct</div></td>' +
      '<td class="' + pnlClass(p.account_pnl) + '" style="font-weight:700">' + fmtPct(p.account_pnl) +
        '<div class="small">' + fmtPct(p.pct_pnl) + ' raw</div></td>' +
      '<td class="' + pnlClass(p.dollar_pnl) + '">' + pnlSign(p.dollar_pnl) + fmt$(p.dollar_pnl) + '</td>' +
      '<td>' + p.days_held + '</td>' +
      '<td>' + (p.setup_tag || '<span class="muted">-</span>') + '</td>' +
      '<td>' + p.entry_date + '</td>' +
      '<td class="btn-group">' +
        '<button class="btn btn-sm" onclick="event.stopPropagation();openEditModal(' + p.id + ')" title="Edit">&#9998;</button>' +
        '<button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deletePosition(' + p.id + ')" title="Delete">&#10005;</button>' +
      '</td></tr>';
    // Expand row
    html += '<tr class="expand-row" id="expand-' + p.id + '"><td colspan="11"><div class="expand-content">' +
      '<div class="expand-section"><h4>Thesis</h4><p>' + (p.entry_thesis || '<span class="muted">No thesis recorded</span>') + '</p></div>' +
      '<div class="expand-section"><h4>Exit Notes</h4><p>' + (p.exit_notes || '<span class="muted">No notes</span>') + '</p></div>' +
      '<div class="expand-section"><h4>Partial Exits (' + p.exits.length + ')</h4><ul class="exit-list">';
    p.exits.forEach(function(ex) {
      html += '<li>' + ex.exit_quantity + ' @ $' + ex.exit_price.toFixed(2) + ' on ' + ex.exit_date + '</li>';
    });
    html += '</ul></div>' +
      '<div class="expand-section"><h4>Account Size</h4><p>' + (p.account_size ? '$' + parseFloat(p.account_size).toLocaleString() : '-') + '</p></div>' +
      '</div></td></tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function toggleExpand(id, event) {
  if (event && (event.target.tagName === 'BUTTON' || event.target.closest('button'))) return;
  var row = document.getElementById('expand-' + id);
  if (row) row.classList.toggle('active');
}

function sortClosed(col) {
  if (SORT.closed.col === col) {
    SORT.closed.asc = !SORT.closed.asc;
  } else {
    SORT.closed.col = col;
    SORT.closed.asc = true;
  }
  renderClosed();
}

// ── Monthly ──
async function renderMonthly() {
  var data = await apiGet('/api/monthly-breakdown' + acctParam('/api/monthly-breakdown'));
  var el = document.getElementById('monthly-content');
  if (!data.length) {
    el.innerHTML = '<div class="empty">No monthly data yet</div>';
    return;
  }

  var maxAbs = Math.max.apply(null, data.map(function(d) { return Math.abs(d.avg_account_pnl * d.total_trades); }));
  if (maxAbs === 0) maxAbs = 1;

  var html = '<table><thead><tr>' +
    '<th>Month</th><th>Trades</th><th>Win Rate</th><th>Avg P&L%</th><th>Total $</th><th>PF</th>' +
    '</tr></thead><tbody>';
  data.forEach(function(m) {
    html += '<tr>' +
      '<td style="font-weight:600">' + m.month + '</td>' +
      '<td>' + m.total_trades + '</td>' +
      '<td class="' + (m.win_rate >= 50 ? 'positive' : 'negative') + '">' + m.win_rate + '%</td>' +
      '<td class="' + pnlClass(m.avg_account_pnl) + '">' + fmtPct(m.avg_account_pnl) + '</td>' +
      '<td class="' + pnlClass(m.total_dollar_pnl) + '">' + pnlSign(m.total_dollar_pnl) + fmt$(m.total_dollar_pnl) + '</td>' +
      '<td>' + m.profit_factor + '</td></tr>';
  });
  html += '</tbody></table>';

  // Bar chart
  html += '<div class="bar-chart">';
  data.forEach(function(m) {
    var totalPnl = m.avg_account_pnl * m.total_trades;
    var pct = Math.abs(totalPnl) / maxAbs * 100;
    var color = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
    html += '<div class="bar-row">' +
      '<span class="bar-label">' + m.month + '</span>' +
      '<div class="bar-track"><div class="bar-fill" style="width:' + Math.max(pct, 2) + '%;background:' + color + '"></div></div>' +
      '<span class="bar-value ' + pnlClass(m.total_dollar_pnl) + '">' + pnlSign(m.total_dollar_pnl) + fmt$(m.total_dollar_pnl) + '</span>' +
      '</div>';
  });
  html += '</div>';

  el.innerHTML = html;
}

// ── Settings ──
async function renderSettings() {
  STATE.settings = await apiGet('/api/settings');
  STATE.setupTags = STATE.settings.setup_tags || [];
  STATE.accounts = await apiGet('/api/accounts');
  renderAccountSelector();

  // Accounts table
  var tbody = document.getElementById('accounts-tbody');
  if (!STATE.accounts.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="muted" style="text-align:center">No accounts — create one below</td></tr>';
  } else {
    tbody.innerHTML = STATE.accounts.map(function(a) {
      return '<tr><td style="font-weight:600">' + a.name + '</td>' +
        '<td>$' + a.starting_size.toLocaleString() + '</td>' +
        '<td>$' + a.current_size.toLocaleString() + '</td>' +
        '<td><button class="btn btn-sm btn-danger" onclick="deleteAccount(' + a.id + ')">Delete</button></td></tr>';
    }).join('');
  }

  var tagList = document.getElementById('tag-list');
  tagList.innerHTML = STATE.setupTags.map(function(tag) {
    return '<div class="tag-item">' + tag +
      '<button class="del-tag" onclick="removeTag(&#39;' + tag.replace(/&/g,'&amp;').replace(/'/g,'&#39;') + '&#39;)">&times;</button></div>';
  }).join('');
}

async function createAccount() {
  var name = document.getElementById('new-acct-name').value.trim();
  var size = document.getElementById('new-acct-size').value;
  if (!name || !size) return;
  var res = await apiPost('/api/accounts', { name: name, starting_size: parseFloat(size) });
  if (res.error) { alert(res.error); return; }
  document.getElementById('new-acct-name').value = '';
  document.getElementById('new-acct-size').value = '';
  renderSettings();
}

async function deleteAccount(aid) {
  if (!confirm('Delete this account?')) return;
  var res = await apiDelete('/api/accounts/' + aid);
  if (res.error) { alert(res.error); return; }
  renderSettings();
}

async function addTag() {
  var input = document.getElementById('new-tag-input');
  var tag = input.value.trim();
  if (!tag || STATE.setupTags.indexOf(tag) >= 0) return;
  STATE.setupTags.push(tag);
  await apiPut('/api/settings', { setup_tags: STATE.setupTags });
  input.value = '';
  renderSettings();
}

async function removeTag(tag) {
  STATE.setupTags = STATE.setupTags.filter(function(t) { return t !== tag; });
  await apiPut('/api/settings', { setup_tags: STATE.setupTags });
  renderSettings();
}

function populateTagDropdown(selectId) {
  var sel = document.getElementById(selectId);
  sel.innerHTML = '<option value="">-- Select --</option>';
  STATE.setupTags.forEach(function(tag) {
    sel.innerHTML += '<option value="' + tag + '">' + tag + '</option>';
  });
}

// ── New Entry Modal ──
var editingId = null;

function populateAccountDropdown(selectId) {
  var sel = document.getElementById(selectId);
  sel.innerHTML = '<option value="">-- Select Account --</option>';
  STATE.accounts.forEach(function(a) {
    sel.innerHTML += '<option value="' + a.id + '">' + a.name + ' ($' + a.current_size.toLocaleString() + ')</option>';
  });
}

function openNewModal() {
  editingId = null;
  document.getElementById('modal-title').textContent = 'New Entry';
  document.getElementById('modal-submit-btn').textContent = 'Create';
  document.getElementById('f-ticker').value = '';
  document.getElementById('f-type').value = 'stock';
  document.getElementById('f-direction').value = 'long';
  document.getElementById('f-price').value = '';
  document.getElementById('f-qty').value = '';
  document.getElementById('f-date').value = today();
  document.getElementById('f-option-type').value = 'call';
  document.getElementById('f-strike').value = '';
  document.getElementById('f-expiry').value = '';
  document.getElementById('f-thesis').innerHTML = '';
  toggleOptionFields();
  populateTagDropdown('f-tag');
  document.getElementById('f-tag').value = '';
  populateAccountDropdown('f-account');
  // Pre-select the active header account (or first if "All")
  var acctSel = document.getElementById('f-account');
  if (STATE.activeAccountId) {
    acctSel.value = STATE.activeAccountId;
  } else if (STATE.accounts.length) {
    acctSel.value = STATE.accounts[0].id;
  }
  document.getElementById('entry-modal').classList.add('active');
  document.getElementById('f-ticker').focus();
}

function closeEntryModal() {
  document.getElementById('entry-modal').classList.remove('active');
}

function toggleOptionFields() {
  var isOpt = document.getElementById('f-type').value === 'option';
  document.getElementById('option-fields').classList.toggle('active', isOpt);
}

async function submitEntry() {
  var accountId = document.getElementById('f-account').value;
  if (!accountId) { alert('Please select an account'); return; }

  var ticker    = document.getElementById('f-ticker').value.trim().toUpperCase();
  var assetType = document.getElementById('f-type').value;
  var direction = document.getElementById('f-direction').value;
  var price     = parseFloat(document.getElementById('f-price').value);
  var qty       = parseFloat(document.getElementById('f-qty').value);
  var entryDate = document.getElementById('f-date').value;
  if (!ticker || !price || !qty) return;

  var optType = null, strike = null, expiry = null;
  if (assetType === 'option') {
    optType = document.getElementById('f-option-type').value;
    strike  = document.getElementById('f-strike').value  ? parseFloat(document.getElementById('f-strike').value)  : null;
    expiry  = document.getElementById('f-expiry').value  || null;
  }

  // Look for an existing open position with the same identity
  var existing = STATE.positions.find(function(p) {
    if (p.status !== 'open')                       return false;
    if (p.ticker !== ticker)                        return false;
    if (p.asset_type !== assetType)                 return false;
    if (p.direction !== direction)                  return false;
    if (parseInt(p.account_id) !== parseInt(accountId)) return false;
    if (assetType === 'option') {
      if (p.option_type !== optType)               return false;
      if (parseFloat(p.strike) !== parseFloat(strike)) return false;
      if (p.expiry !== expiry)                     return false;
    }
    return true;
  });

  if (existing) {
    // Add to existing position — weighted-average entry price
    var newQty = existing.quantity + qty;
    var newAvg = (existing.entry_price * existing.quantity + price * qty) / newQty;
    await apiPut('/api/position/' + existing.id, {
      entry_price: parseFloat(newAvg.toFixed(4)),
      quantity:    newQty,
    });
  } else {
    var body = {
      ticker:       ticker,
      asset_type:   assetType,
      direction:    direction,
      entry_price:  price,
      quantity:     qty,
      entry_date:   entryDate,
      setup_tag:    document.getElementById('f-tag').value || null,
      entry_thesis: sanitizeRichContent(document.getElementById('f-thesis').innerHTML),
      account_id:   parseInt(accountId),
    };
    if (assetType === 'option') {
      body.option_type = optType;
      body.strike      = strike;
      body.expiry      = expiry;
    }
    await apiPost('/api/position', body);
  }

  closeEntryModal();
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  // Switch to open tab
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
  document.querySelector('[data-tab="open"]').classList.add('active');
  document.getElementById('tab-open').classList.add('active');
  renderOpen();
}

// ── Edit Modal ──
function openEditModal(pid) {
  var pos = STATE.positions.find(function(p) { return p.id === pid; });
  if (!pos) return;
  document.getElementById('e-id').value = pid;
  populateTagDropdown('e-tag');
  document.getElementById('e-tag').value = pos.setup_tag || '';
  document.getElementById('e-thesis').innerHTML = pos.entry_thesis || '';
  document.getElementById('e-notes').innerHTML = pos.exit_notes || '';
  document.getElementById('edit-modal').classList.add('active');
}

function closeEditModal() {
  document.getElementById('edit-modal').classList.remove('active');
}

async function submitEdit() {
  var pid = document.getElementById('e-id').value;
  var body = {
    setup_tag: document.getElementById('e-tag').value || null,
    entry_thesis: sanitizeRichContent(document.getElementById('e-thesis').innerHTML),
    exit_notes: sanitizeRichContent(document.getElementById('e-notes').innerHTML),
  };
  await apiPut('/api/position/' + pid, body);
  closeEditModal();
  STATE.positions = await apiGet('/api/positions?status=all' + acctParam('/api/positions?status=all'));
  STATE.accounts = await apiGet('/api/accounts'); renderAccountSelector();
  renderOpen();
  renderClosed();
}

// ── CSV Export ──
function exportCSV() {
  var closed = STATE.positions.filter(function(p) { return p.status === 'closed'; });
  if (!closed.length) { alert('No closed trades to export'); return; }
  var cols = ['ticker','asset_type','direction','strike','expiry','entry_price','quantity',
              'entry_date','setup_tag','avg_exit_price','pct_pnl','account_pnl','dollar_pnl',
              'days_held','entry_thesis','exit_notes','account_size'];
  var csv = cols.join(',') + '\\n';
  closed.forEach(function(p) {
    csv += cols.map(function(c) {
      var v = p[c];
      if (v == null) return '';
      v = String(v);
      if (v.indexOf(',') >= 0 || v.indexOf('"') >= 0 || v.indexOf('\\n') >= 0) {
        v = '"' + v.replace(/"/g, '""') + '"';
      }
      return v;
    }).join(',') + '\\n';
  });
  var blob = new Blob([csv], { type: 'text/csv' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'trade_journal_export.csv';
  a.click();
}

// ── Modal close on overlay / escape ──
document.querySelectorAll('.modal-overlay').forEach(function(overlay) {
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) {
      overlay.classList.remove('active');
    }
  });
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.active').forEach(function(m) {
      m.classList.remove('active');
    });
  }
});

// ── Boot ──
initRichEditors();
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trade Journal")
    parser.add_argument("--port", type=int, default=5070, help="Port (default: 5070)")
    args = parser.parse_args()

    init_db()

    print(f"[*] Starting Trade Journal at http://127.0.0.1:{args.port}")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
