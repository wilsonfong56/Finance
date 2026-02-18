"""
Combined Screener blueprint
===========================
Wraps combined_screener.py in a proper web form: user picks universe/side/
screeners, a background thread runs the scan, and the results page reuses
combined_screener.build_html() so charts/modals work as before.
"""

import threading
import time
import uuid
from collections import defaultdict

import yfinance as yf
from flask import Blueprint, jsonify, request

import combined_screener as cs

LABEL   = "Stock Screener"
ICON    = "🔍"
SECTION = "Screeners"
ORDER   = 1

PREFIX = "/screeners"
bp = Blueprint("screeners", __name__, url_prefix=PREFIX)

# ── Discover available screeners at startup ────────────────────────────────────
_SCREENERS = cs.discover_screeners()   # {mod_name: {name, run_scan, data_params}}

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: dict = {}   # job_id -> {status, message, html, elapsed, started}

_FORM_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Screener</title>
<style>
  :root { --bg:#0d1117; --surface:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
          --green:#3fb950; --red:#f85149; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--text); }
  header { background:var(--surface); border-bottom:1px solid var(--border);
           padding:12px 24px; }
  header h1 { font-size:18px; font-weight:600; }
  main { padding:32px 24px; max-width:620px; margin:0 auto; }

  .section-label { font-size:11px; font-weight:600; color:var(--muted);
                   text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px; }
  .form-group { margin-bottom:24px; }
  .option-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(148px,1fr)); gap:8px; }
  .opt { padding:10px 14px; background:var(--surface); border:1px solid var(--border);
         border-radius:8px; color:var(--muted); cursor:pointer; font-size:13px;
         font-weight:500; text-align:left; transition:all 0.15s;
         border:none; width:100%; }
  .opt:hover { color:var(--text); background:rgba(255,255,255,0.04); }
  .opt.on { border:1px solid var(--accent); color:var(--accent);
            background:rgba(88,166,255,0.08); }
  .run-btn { width:100%; padding:13px; background:var(--accent); color:#fff;
             border:none; border-radius:8px; font-size:14px; font-weight:600;
             cursor:pointer; margin-top:4px; transition:background 0.15s; }
  .run-btn:hover { background:#4a9aef; }
  .run-btn:disabled { opacity:0.5; cursor:not-allowed; }

  /* Progress */
  #progress { display:none; text-align:center; margin-top:32px; }
  .spinner { width:36px; height:36px; border:3px solid var(--border);
             border-top-color:var(--accent); border-radius:50%;
             animation:spin 0.8s linear infinite; margin:0 auto 14px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #status-msg { color:var(--muted); font-size:13px; margin-bottom:4px; }
  #elapsed    { font-size:12px; color:var(--muted); opacity:0.6; }
  .err-msg    { color:var(--red); padding:32px; text-align:center; }
</style>
</head>
<body>
<header><h1>Stock Screener</h1></header>
<main>
<div id="form-wrap">
  <div class="form-group">
    <div class="section-label">Universe</div>
    <div class="option-grid" id="universe-grid">
      <button class="opt on" data-group="universe" data-value="spy">S&amp;P 500</button>
      <button class="opt"    data-group="universe" data-value="qqq">Nasdaq 100</button>
      <button class="opt"    data-group="universe" data-value="iwm">Russell 2000</button>
      <button class="opt"    data-group="universe" data-value="dia">Dow Jones 30</button>
      <button class="opt"    data-group="universe" data-value="all">All (slow)</button>
    </div>
  </div>

  <div class="form-group">
    <div class="section-label">Direction</div>
    <div class="option-grid">
      <button class="opt on" data-group="side" data-value="both">Both</button>
      <button class="opt"    data-group="side" data-value="bull">Bullish</button>
      <button class="opt"    data-group="side" data-value="bear">Bearish</button>
    </div>
  </div>

  <div class="form-group">
    <div class="section-label">Screeners</div>
    <div class="option-grid" id="screener-grid"></div>
  </div>

  <button class="run-btn" id="run-btn" onclick="startScan()">Run Scan</button>
</div>

<div id="progress">
  <div class="spinner"></div>
  <div id="status-msg">Starting scan…</div>
  <div id="elapsed"></div>
</div>
</main>

<script>
const SCREENERS = __SCREENERS_JSON__;
let selectedUniverse = 'spy', selectedSide = 'both';
let selectedScreeners = new Set(Object.keys(SCREENERS));

// Build screener toggle buttons
const grid = document.getElementById('screener-grid');
Object.entries(SCREENERS).forEach(([key, name]) => {
  const btn = document.createElement('button');
  btn.className = 'opt on';
  btn.dataset.value = key;
  btn.textContent = name;
  btn.onclick = () => {
    if (selectedScreeners.has(key)) { selectedScreeners.delete(key); btn.classList.remove('on'); }
    else                            { selectedScreeners.add(key);    btn.classList.add('on');    }
  };
  grid.appendChild(btn);
});

// Universe / side toggles (single-select)
document.querySelectorAll('.opt[data-group]').forEach(btn => {
  btn.addEventListener('click', () => {
    const g = btn.dataset.group;
    document.querySelectorAll(`[data-group="${g}"]`).forEach(b => b.classList.remove('on'));
    btn.classList.add('on');
    if (g === 'universe') selectedUniverse = btn.dataset.value;
    else if (g === 'side') selectedSide = btn.dataset.value;
  });
});

async function startScan() {
  if (!selectedScreeners.size) { alert('Select at least one screener.'); return; }
  document.getElementById('form-wrap').style.display  = 'none';
  document.getElementById('progress').style.display   = 'block';
  document.getElementById('run-btn').disabled = true;

  const res = await fetch('/screeners/api/scan', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      universe: selectedUniverse,
      side:     selectedSide,
      screeners: [...selectedScreeners],
    }),
  });
  const {job_id} = await res.json();

  const t0    = Date.now();
  const poll  = setInterval(async () => {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
    document.getElementById('elapsed').textContent = elapsed + 's elapsed';

    const s = await (await fetch('/screeners/api/status/' + job_id)).json();
    document.getElementById('status-msg').textContent = s.message || '…';

    if (s.status === 'done') {
      clearInterval(poll);
      window.location.href = '/screeners/results/' + job_id;
    } else if (s.status === 'error') {
      clearInterval(poll);
      document.getElementById('progress').innerHTML =
        '<div class="err-msg">Error: ' + s.message + '</div>';
    }
  }, 2000);
}
</script>
</body>
</html>
"""


def _build_form_html():
    import json
    screener_map = {k: v["name"] for k, v in _SCREENERS.items()}
    return _FORM_HTML.replace("__SCREENERS_JSON__", json.dumps(screener_map))


_CACHED_FORM = _build_form_html()


# ── Scan worker ────────────────────────────────────────────────────────────────

def _run_scan(job_id: str, universe: str, side: str, screener_keys: list):
    job = _jobs[job_id]
    try:
        # Fetch tickers
        job["message"] = f"Fetching {universe.upper()} universe…"
        fetch_fn = cs.FETCH_FUNCTIONS.get(universe, cs.fetch_sp500)
        if universe == "all":
            job["message"] = "Fetching all universes…"
            tickers = cs.fetch_all_universes()
        else:
            tickers = fetch_fn()
        job["message"] = f"Got {len(tickers)} tickers. Downloading data…"

        # Filter screeners
        screeners = {k: v for k, v in _SCREENERS.items() if k in screener_keys}
        if not screeners:
            raise ValueError("No valid screeners selected")

        # Group by DATA_PARAMS to download each combo once
        data_groups: dict = defaultdict(list)
        for mod_name, info in screeners.items():
            key = (info["data_params"]["period"], info["data_params"]["interval"])
            data_groups[key].append(mod_name)

        downloaded = {}
        for (period, interval), mod_names in data_groups.items():
            job["message"] = f"Downloading {interval} data ({period})…"
            data = cs.download_data(tickers, period=period, interval=interval)
            if data is not None:
                downloaded[(period, interval)] = data

        # Run each screener
        job["message"] = "Running screeners…"
        t0 = time.time()
        all_data: dict = {}
        ticker_sets: list = []

        for mod_name, info in screeners.items():
            dp  = info["data_params"]
            data = downloaded.get((dp["period"], dp["interval"]))
            if data is None:
                ticker_sets.append(set())
                all_data[mod_name] = {"display_name": info["name"], "results": [],
                                      "chart_data": {}, "scanned": 0}
                continue
            try:
                results, chart_data, scanned = info["run_scan"](tickers, side=side, data=data)
            except Exception as ex:
                print(f"[screeners] {info['name']} error: {ex}")
                results, chart_data, scanned = [], {}, 0
            ticker_sets.append({r["ticker"] for r in results})
            all_data[mod_name] = {"display_name": info["name"], "results": results,
                                  "chart_data": chart_data, "scanned": scanned}

        # Intersection
        common_tickers = ticker_sets[0] if ticker_sets else set()
        for s in ticker_sets[1:]:
            common_tickers &= s

        for mod_name in all_data:
            all_data[mod_name]["results"]    = [r for r in all_data[mod_name]["results"]
                                                if r["ticker"] in common_tickers]
            all_data[mod_name]["chart_data"] = {k: v for k, v in all_data[mod_name]["chart_data"].items()
                                                if k in common_tickers}

        elapsed = time.time() - t0

        # Build HTML — then rewrite the chart API path to our prefix
        html = cs.build_html(all_data, elapsed, len(common_tickers), len(screeners))
        html = html.replace("fetch('/api/chart/", f"fetch('{PREFIX}/api/chart/")

        job["status"]  = "done"
        job["html"]    = html
        job["elapsed"] = round(elapsed, 1)
        job["message"] = f"Done — {len(common_tickers)} ticker(s) matched all screeners"

    except Exception as ex:
        job["status"]  = "error"
        job["message"] = str(ex)


# ── Routes ─────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    return _CACHED_FORM


@bp.route("/api/scan", methods=["POST"])
def api_scan():
    body          = request.get_json()
    universe      = body.get("universe", "spy")
    side          = body.get("side", "both")
    screener_keys = body.get("screeners", list(_SCREENERS.keys()))

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":  "running",
        "message": "Starting…",
        "html":    None,
        "elapsed": 0,
        "started": time.time(),
    }
    t = threading.Thread(
        target=_run_scan,
        args=(job_id, universe, side, screener_keys),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})


@bp.route("/api/status/<job_id>")
def api_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    return jsonify({
        "status":  job["status"],
        "message": job["message"],
        "elapsed": round(time.time() - job["started"], 1),
    })


@bp.route("/results/<job_id>")
def results(job_id):
    job = _jobs.get(job_id)
    if not job:
        return "Job not found", 404
    if job["status"] == "running":
        return "Scan still in progress", 202
    if job["status"] == "error":
        return f"Scan error: {job['message']}", 500
    return job["html"]


@bp.route("/api/chart/<ticker>")
def api_chart(ticker):
    interval = request.args.get("interval", "1d")
    rng      = request.args.get("range", "6mo")
    try:
        df = yf.download(ticker, period=rng, interval=interval, progress=False)
        if df.empty:
            return jsonify({"error": "No data"})
        records = []
        for dt, row in df.iterrows():
            t = dt.strftime("%Y-%m-%d") if interval in ("1d", "1wk", "1mo") else int(dt.timestamp())
            def _f(col):
                v = row[col]
                return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
            records.append({
                "time": t, "open": _f("Open"), "high": _f("High"),
                "low":  _f("Low"), "close": _f("Close"), "volume": _f("Volume"),
            })
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": str(e)})
