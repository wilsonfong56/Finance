"""Options Analyzer blueprint — wraps options_analyzer.py in a web UI."""

import io
from contextlib import redirect_stdout

from flask import Blueprint, jsonify, request
import options_analyzer as oa

LABEL   = "Options Analyzer"
ICON    = "📊"
SECTION = "Markets"
ORDER   = 1

PREFIX = "/options"
bp = Blueprint("options", __name__, url_prefix=PREFIX)

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Options Analyzer</title>
<style>
  :root { --bg:#0d1117; --surface:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
          --green:#3fb950; --red:#f85149; --yellow:#d29922; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--text); }
  header { background:var(--surface); border-bottom:1px solid var(--border);
           padding:12px 24px; display:flex; align-items:center; gap:16px; }
  header h1 { font-size:18px; font-weight:600; }
  .search-form { display:flex; gap:8px; align-items:center; }
  .search-form input {
    padding:7px 12px; background:var(--bg); border:1px solid var(--border);
    border-radius:6px; color:var(--text); font-size:13px; width:120px;
    text-transform:uppercase; }
  .search-form input:focus { outline:none; border-color:var(--accent); }
  .search-form button {
    padding:7px 16px; background:var(--accent); border:none; border-radius:6px;
    color:#fff; font-size:13px; font-weight:600; cursor:pointer; }
  .search-form button:hover { background:#4a9aef; }
  .search-form button:disabled { opacity:0.5; cursor:not-allowed; }
  main { padding:20px 24px; max-width:1600px; margin:0 auto; }

  /* Summary cards */
  .summary { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:8px;
          padding:12px 16px; min-width:120px; }
  .card .label { font-size:11px; color:var(--muted); text-transform:uppercase;
                 letter-spacing:0.5px; margin-bottom:4px; }
  .card .value { font-size:20px; font-weight:700; }
  .card .sub   { font-size:11px; color:var(--muted); margin-top:2px; }

  /* Section headers */
  .section-hdr { font-size:13px; font-weight:700; color:var(--muted);
                 text-transform:uppercase; letter-spacing:0.5px;
                 border-bottom:1px solid var(--border); padding-bottom:8px;
                 margin:20px 0 10px; }

  /* Table */
  .tbl-wrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:12px; min-width:700px; }
  th { text-align:right; padding:8px 10px; color:var(--muted); font-size:10px;
       text-transform:uppercase; border-bottom:1px solid var(--border); cursor:pointer;
       white-space:nowrap; user-select:none; }
  th:first-child, th:nth-child(2) { text-align:left; }
  th:hover { color:var(--text); }
  td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--border); }
  td:first-child, td:nth-child(2) { text-align:left; }
  tr:hover { background:rgba(88,166,255,0.04); }
  .call { color:var(--green); font-weight:600; }
  .put  { color:var(--red);   font-weight:600; }
  .muted { color:var(--muted); }

  /* Misc */
  .spinner { display:flex; align-items:center; justify-content:center; height:200px; }
  .spin { width:28px; height:28px; border:3px solid var(--border); border-top-color:var(--accent);
          border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .error { color:var(--red); padding:40px; text-align:center; }
  .welcome { color:var(--muted); padding:60px; text-align:center; font-size:14px; }
  .exps { font-size:12px; color:var(--muted); margin-bottom:4px; }
</style>
</head>
<body>
<header>
  <h1>Options Analyzer</h1>
  <form class="search-form" onsubmit="analyze(event)">
    <input id="ticker-input" placeholder="AAPL" maxlength="10">
    <button id="search-btn" type="submit">Analyze</button>
  </form>
</header>
<main id="main">
  <div class="welcome">Enter a ticker above to analyze its options chain.</div>
</main>

<script>
let _data = null;
let _sortCol = 'iv', _sortAsc = true;

async function analyze(e) {
  if (e) e.preventDefault();
  const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
  if (!ticker) return;
  const btn = document.getElementById('search-btn');
  btn.disabled = true; btn.textContent = 'Loading...';
  document.getElementById('main').innerHTML =
    '<div class="spinner"><div class="spin"></div></div>';
  try {
    const res  = await fetch('/options/api/analyze/' + ticker);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    _data = data;
    render();
  } catch(err) {
    document.getElementById('main').innerHTML =
      '<div class="error">' + err.message + '</div>';
  } finally {
    btn.disabled = false; btn.textContent = 'Analyze';
  }
}

function fmt(v, dec=2) {
  if (v === null || v === undefined) return '—';
  return typeof v === 'number' ? v.toFixed(dec) : v;
}

function ivColor(iv, iv30) {
  if (!iv30) return '';
  if (iv < iv30 * 0.9)  return 'color:var(--green)';
  if (iv > iv30 * 1.1) return 'color:var(--red)';
  return '';
}

function rankColor(r) {
  if (r === null) return '';
  if (r < 30) return 'color:var(--green)';
  if (r > 70) return 'color:var(--red)';
  return 'color:var(--yellow)';
}

function buildTable(rows, cols, d) {
  const thead = cols.map(c =>
    `<th onclick="sortTable('${c.key}')">${c.label} ${_sortCol===c.key ? (_sortAsc?'↑':'↓') : ''}</th>`
  ).join('');
  const sorted = [...rows].sort((a, b) => {
    const va = a[_sortCol] ?? 0, vb = b[_sortCol] ?? 0;
    return _sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });
  const tbody = sorted.slice(0,25).map(r => {
    const cells = cols.map(c => {
      const v = r[c.key];
      if (c.key === 'type') return `<td class="${v}">${v}</td>`;
      let style = '';
      if (c.key === 'iv')      style = ivColor(v, d.iv30);
      if (c.key === 'probITM' || c.key === 'probProfit') style = v > 50 ? 'color:var(--green)' : 'color:var(--red)';
      const disp = c.dec !== undefined ? fmt(v, c.dec) : (v === null ? '—' : v);
      return `<td style="${style}">${disp}</td>`;
    }).join('');
    return '<tr>' + cells + '</tr>';
  }).join('');
  return `<div class="tbl-wrap"><table>
    <thead><tr>${thead}</tr></thead>
    <tbody>${tbody}</tbody>
  </table></div>`;
}

function sortTable(col) {
  if (_sortCol === col) _sortAsc = !_sortAsc;
  else { _sortCol = col; _sortAsc = true; }
  render();
}

const COLS = [
  {key:'type',       label:'Type'},
  {key:'strike',     label:'Strike',   dec:2},
  {key:'expiration', label:'Exp'},
  {key:'dte',        label:'DTE',      dec:0},
  {key:'bid',        label:'Bid',      dec:2},
  {key:'ask',        label:'Ask',      dec:2},
  {key:'mid',        label:'Mid',      dec:2},
  {key:'volume',     label:'Vol',      dec:0},
  {key:'oi',         label:'OI',       dec:0},
  {key:'iv',         label:'IV %',     dec:1},
  {key:'ivVsIV30',   label:'vs IV30',  dec:1},
  {key:'delta',      label:'Delta',    dec:3},
  {key:'probITM',    label:'P(ITM)',   dec:1},
  {key:'probProfit', label:'P(Profit)',dec:1},
];

function render() {
  const d = _data;
  const iv30    = d.iv30 ? d.iv30.toFixed(1) + '%' : 'N/A';
  const ivRank  = d.iv_rank    !== null ? d.iv_rank.toFixed(1) + '%'    : 'N/A';
  const ivPctl  = d.iv_percentile !== null ? d.iv_percentile.toFixed(1) + '%' : 'N/A';

  const cheapest = [...d.options]
    .filter(r => r.iv > 0)
    .sort((a,b) => a.ivVsIV30 - b.ivVsIV30);

  const liquid = [...d.options]
    .filter(r => r.volume >= 200 && r.oi >= 500)
    .sort((a,b) => b.volume - a.volume);

  const unusual = [...d.options]
    .filter(r => r.volume >= 1000 && r.oi > 0 && (r.volume / r.oi) >= 3)
    .sort((a,b) => b.volume - a.volume);

  document.getElementById('main').innerHTML = `
    <div class="summary">
      <div class="card">
        <div class="label">Price</div>
        <div class="value">$${d.price.toFixed(2)}</div>
      </div>
      <div class="card">
        <div class="label">IV30</div>
        <div class="value">${iv30}</div>
        <div class="sub">chg: ${d.iv30_change !== null ? (d.iv30_change > 0 ? '+' : '') + d.iv30_change.toFixed(1) + '%' : '—'}</div>
      </div>
      <div class="card">
        <div class="label">IV Rank</div>
        <div class="value" style="${rankColor(d.iv_rank)}">${ivRank}</div>
        <div class="sub">${d.iv_history_days} days of history</div>
      </div>
      <div class="card">
        <div class="label">IV Percentile</div>
        <div class="value" style="${rankColor(d.iv_percentile)}">${ivPctl}</div>
      </div>
    </div>
    <div class="exps">Expirations: ${d.expirations.join(' · ')}</div>

    <div class="section-hdr">Cheapest Options (lowest IV vs IV30)</div>
    ${buildTable(cheapest, COLS, d)}

    <div class="section-hdr">Most Liquid</div>
    ${buildTable(liquid, COLS.filter(c => c.key !== 'ivVsIV30' && c.key !== 'probProfit'), d)}

    <div class="section-hdr">Unusual Activity (volume ≥ 3× OI)</div>
    ${buildTable(unusual, COLS.filter(c => c.key !== 'ivVsIV30' && c.key !== 'probProfit'), d)}
  `;
}
</script>
</body>
</html>
"""


@bp.route("/")
def index():
    return _HTML


@bp.route("/api/analyze/<ticker>")
def api_analyze(ticker):
    ticker = ticker.upper()
    try:
        with redirect_stdout(io.StringIO()):
            data = oa.analyze_options(ticker)
        if not data:
            return jsonify({"error": f"No data found for {ticker}"}), 404
        # Serialize the DataFrame
        opts = data["options"]
        data_out = {
            "ticker":          data["ticker"],
            "price":           data["price"],
            "iv30":            data["iv30"],
            "iv30_change":     data.get("iv30_change", 0),
            "iv_rank":         data["iv_rank"],
            "iv_percentile":   data["iv_percentile"],
            "iv_history_days": data["iv_history_days"],
            "expirations":     data["expirations"],
            "options":         opts.to_dict(orient="records"),
        }
        return jsonify(data_out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
