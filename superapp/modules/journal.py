"""Trade Journal blueprint — delegates all logic to Screeners/trade_journal.py."""

from flask import Blueprint, jsonify, request
import trade_journal as tj

LABEL   = "Trade Journal"
ICON    = "📒"
SECTION = "Portfolio"
ORDER   = 1

PREFIX = "/journal"
bp = Blueprint("journal", __name__, url_prefix=PREFIX)

# Rewrite embedded HTML so JS fetch calls hit /journal/api/... not /api/...
_HTML = tj._HTML.replace("'/api/", f"'{PREFIX}/api/").replace('"/api/', f'"{PREFIX}/api/')


@bp.route("/")
def index():
    return _HTML


# ── Positions ──────────────────────────────────────────────────────────────────

@bp.route("/api/positions")
def api_positions():
    status     = request.args.get("status", "all")
    account_id = request.args.get("account_id")
    conn = tj.get_db()
    positions = tj._get_positions(
        conn,
        status=None if status == "all" else status,
        account_id=account_id,
    )
    conn.close()
    return jsonify(positions)


@bp.route("/api/position", methods=["POST"])
def api_create_position():
    data       = request.get_json()
    account_id = data.get("account_id")
    if not account_id:
        return jsonify({"error": "account_id is required"}), 400
    conn         = tj.get_db()
    account_size = tj._account_current_size(conn, int(account_id))
    cur = conn.execute(
        """INSERT INTO positions
           (ticker, asset_type, direction, option_type, strike, expiry,
            entry_price, quantity, entry_date, setup_tag, entry_thesis,
            account_size, account_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["ticker"].upper().strip(),
            data.get("asset_type", "stock"),
            data.get("direction", "long"),
            data.get("option_type"),
            data.get("strike"),
            data.get("expiry"),
            float(data["entry_price"]),
            float(data["quantity"]),
            data.get("entry_date", tj.date.today().isoformat()),
            data.get("setup_tag"),
            data.get("entry_thesis"),
            account_size,
            int(account_id),
        ),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return jsonify({"id": pid, "status": "created"}), 201


@bp.route("/api/position/<int:pid>", methods=["PUT"])
def api_update_position(pid):
    data = request.get_json()
    conn = tj.get_db()
    fields, values = [], []
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


@bp.route("/api/position/<int:pid>", methods=["DELETE"])
def api_delete_position(pid):
    conn = tj.get_db()
    conn.execute("DELETE FROM positions WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted"})


# ── Exits ──────────────────────────────────────────────────────────────────────

@bp.route("/api/exit", methods=["POST"])
def api_create_exit():
    data       = request.get_json()
    pid        = int(data["position_id"])
    exit_qty   = float(data["exit_quantity"])
    exit_price = float(data["exit_price"])
    exit_date  = data.get("exit_date", tj.date.today().isoformat())
    conn = tj.get_db()
    pos  = conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
    if not pos:
        conn.close()
        return jsonify({"error": "Position not found"}), 404
    exited = conn.execute(
        "SELECT COALESCE(SUM(exit_quantity),0) as total FROM exits WHERE position_id=?",
        (pid,),
    ).fetchone()["total"]
    remaining = pos["quantity"] - exited
    if exit_qty > remaining + 0.0001:
        conn.close()
        return jsonify({"error": f"Exit qty {exit_qty} exceeds remaining {remaining}"}), 400
    conn.execute(
        "INSERT INTO exits (position_id, exit_price, exit_quantity, exit_date) VALUES (?,?,?,?)",
        (pid, exit_price, exit_qty, exit_date),
    )
    new_exited = exited + exit_qty
    if new_exited >= pos["quantity"] - 0.0001:
        conn.execute("UPDATE positions SET status='closed' WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "exit_logged"}), 201


@bp.route("/api/exit/<int:eid>", methods=["DELETE"])
def api_delete_exit(eid):
    conn = tj.get_db()
    ex   = conn.execute("SELECT * FROM exits WHERE id=?", (eid,)).fetchone()
    if not ex:
        conn.close()
        return jsonify({"error": "Exit not found"}), 404
    pid = ex["position_id"]
    conn.execute("DELETE FROM exits WHERE id=?", (eid,))
    pos = conn.execute("SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
    if pos:
        exited = conn.execute(
            "SELECT COALESCE(SUM(exit_quantity),0) as total FROM exits WHERE position_id=?",
            (pid,),
        ).fetchone()["total"]
        if exited < pos["quantity"] - 0.0001:
            conn.execute("UPDATE positions SET status='open' WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return jsonify({"status": "exit_deleted"})


# ── Stats & analytics ──────────────────────────────────────────────────────────

@bp.route("/api/stats")
def api_stats():
    account_id = request.args.get("account_id")
    conn       = tj.get_db()
    positions  = tj._get_positions(conn, status="closed", account_id=account_id)
    conn.close()
    overall    = tj.compute_stats(positions)
    by_setup   = {}
    for p in positions:
        tag = p.get("setup_tag") or "Untagged"
        by_setup.setdefault(tag, []).append(p)
    setup_stats = {tag: tj.compute_stats(ps) for tag, ps in by_setup.items()}
    return jsonify({"overall": overall, "by_setup": setup_stats})


@bp.route("/api/stats/direction")
def api_stats_direction():
    account_id = request.args.get("account_id")
    conn       = tj.get_db()
    positions  = tj._get_positions(conn, status="closed", account_id=account_id)
    conn.close()
    longs  = [p for p in positions if p["direction"] == "long"]
    shorts = [p for p in positions if p["direction"] == "short"]
    return jsonify({"long": tj.compute_stats(longs), "short": tj.compute_stats(shorts)})


@bp.route("/api/monthly-breakdown")
def api_monthly_breakdown():
    account_id = request.args.get("account_id")
    conn       = tj.get_db()
    positions  = tj._get_positions(conn, status="closed", account_id=account_id)
    conn.close()
    by_month = {}
    for p in positions:
        if p["exits"]:
            month_key = max(e["exit_date"] for e in p["exits"])[:7]
        else:
            month_key = p["entry_date"][:7]
        by_month.setdefault(month_key, []).append(p)
    result = []
    for month in sorted(by_month):
        s = tj.compute_stats(by_month[month])
        s["month"] = month
        result.append(s)
    return jsonify(result)


@bp.route("/api/equity-curve")
def api_equity_curve():
    period     = request.args.get("period", "all")
    account_id = request.args.get("account_id")
    conn       = tj.get_db()
    positions  = tj._get_positions(conn, status="closed", account_id=account_id)
    conn.close()
    points = []
    for p in positions:
        if not p["exits"]:
            continue
        last_exit = max(e["exit_date"] for e in p["exits"])
        points.append({"date": last_exit, "account_pnl": p["account_pnl"]})
    points.sort(key=lambda x: x["date"])
    today_str = tj.date.today().isoformat()
    if period == "day":
        points = [pt for pt in points if pt["date"] == today_str]
    elif period == "wtd":
        d      = tj.date.today()
        monday = tj.date.fromordinal(d.toordinal() - d.weekday()).isoformat()
        points = [pt for pt in points if pt["date"] >= monday]
    elif period == "mtd":
        points = [pt for pt in points if pt["date"] >= today_str[:8] + "01"]
    elif period == "ytd":
        points = [pt for pt in points if pt["date"] >= today_str[:5] + "01-01"]
    cumulative, running = [], 0.0
    for pt in points:
        running += pt["account_pnl"]
        cumulative.append({"date": pt["date"], "cumulative_pnl": round(running, 2)})
    return jsonify(cumulative)


# ── Settings ───────────────────────────────────────────────────────────────────

@bp.route("/api/settings", methods=["GET"])
def api_get_settings():
    import json as _json
    conn     = tj.get_db()
    rows     = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    settings = {}
    for r in rows:
        settings[r["key"]] = _json.loads(r["value"]) if r["key"] == "setup_tags" else r["value"]
    return jsonify(settings)


@bp.route("/api/settings", methods=["PUT"])
def api_update_settings():
    import json as _json
    data = request.get_json()
    conn = tj.get_db()
    for key, value in data.items():
        val = _json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, val))
    conn.commit()
    conn.close()
    return jsonify({"status": "updated"})


# ── Accounts ───────────────────────────────────────────────────────────────────

@bp.route("/api/accounts")
def api_list_accounts():
    conn     = tj.get_db()
    rows     = conn.execute("SELECT * FROM accounts ORDER BY created_at").fetchall()
    accounts = []
    for row in rows:
        a               = dict(row)
        a["current_size"] = tj._account_current_size(conn, row["id"])
        accounts.append(a)
    conn.close()
    return jsonify(accounts)


@bp.route("/api/accounts", methods=["POST"])
def api_create_account():
    import sqlite3 as _sqlite3
    data         = request.get_json()
    name         = (data.get("name") or "").strip()
    starting_size = data.get("starting_size")
    if not name or starting_size is None:
        return jsonify({"error": "name and starting_size required"}), 400
    conn = tj.get_db()
    try:
        cur = conn.execute(
            "INSERT INTO accounts (name, starting_size) VALUES (?, ?)",
            (name, float(starting_size)),
        )
        conn.commit()
        aid = cur.lastrowid
    except _sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Account name already exists"}), 409
    conn.close()
    return jsonify({"id": aid, "status": "created"}), 201


@bp.route("/api/positions/merge", methods=["POST"])
def api_merge_positions():
    data = request.get_json()
    ids  = data.get("ids", [])
    if len(ids) < 2:
        return jsonify({"error": "Need at least 2 positions"}), 400
    conn = tj.get_db()
    primary_id, err = tj.merge_positions(conn, ids)
    conn.close()
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"status": "merged", "id": primary_id})


@bp.route("/api/accounts/<int:aid>", methods=["DELETE"])
def api_delete_account(aid):
    conn  = tj.get_db()
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


# ── Init DB on first import ────────────────────────────────────────────────────
tj.init_db()
