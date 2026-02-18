"""Position Sizer blueprint — wraps position_sizer.py in a web dashboard."""

from flask import Blueprint, jsonify
import position_sizer as ps

LABEL   = "Position Sizer"
ICON    = "⚖️"
SECTION = "Research"
ORDER   = 2

PREFIX = "/sizer"
bp = Blueprint("sizer", __name__, url_prefix=PREFIX)

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Position Sizer</title>
<style>
  :root { --bg:#0d1117; --surface:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
          --green:#3fb950; --red:#f85149; --yellow:#d29922; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--text); }
  header { background:var(--surface); border-bottom:1px solid var(--border);
           padding:12px 24px; display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:18px; font-weight:600; }
  .refresh-btn { padding:6px 14px; background:var(--surface); border:1px solid var(--border);
                 border-radius:6px; color:var(--muted); font-size:12px; cursor:pointer; }
  .refresh-btn:hover { color:var(--text); border-color:var(--muted); }
  main { padding:24px; max-width:900px; margin:0 auto; }

  /* Multiplier hero */
  .hero { text-align:center; padding:32px 0 24px; }
  .mult-label { font-size:12px; color:var(--muted); text-transform:uppercase;
                letter-spacing:0.8px; margin-bottom:8px; }
  .mult-value { font-size:72px; font-weight:800; line-height:1; }
  .mult-sub   { font-size:13px; color:var(--muted); margin-top:8px; }

  /* Gauge bar */
  .gauge-wrap { margin:16px auto; max-width:400px; }
  .gauge-track { height:10px; background:var(--border); border-radius:5px; overflow:hidden; }
  .gauge-fill  { height:100%; border-radius:5px; transition:width 0.6s ease; }
  .gauge-labels { display:flex; justify-content:space-between;
                  font-size:11px; color:var(--muted); margin-top:4px; }

  /* Cards */
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; margin-bottom:24px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
  .card .label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
  .card .value { font-size:22px; font-weight:700; }
  .card .sub   { font-size:11px; color:var(--muted); margin-top:3px; }

  /* Score bars */
  .section-title { font-size:13px; font-weight:600; color:var(--muted);
                   text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; }
  .score-row { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
  .score-name  { width:130px; font-size:12px; color:var(--muted); flex-shrink:0; }
  .score-track { flex:1; height:8px; background:var(--border); border-radius:4px; overflow:hidden; }
  .score-fill  { height:100%; border-radius:4px; background:var(--accent); transition:width 0.5s ease; }
  .score-val   { width:36px; text-align:right; font-size:12px; font-weight:600; }

  /* Indicators table */
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:16px; }
  th { text-align:left; padding:8px 12px; color:var(--muted); font-size:11px;
       text-transform:uppercase; border-bottom:1px solid var(--border); }
  td { padding:8px 12px; border-bottom:1px solid var(--border); }

  .spinner { display:flex; align-items:center; justify-content:center; height:200px; }
  .spin { width:28px; height:28px; border:3px solid var(--border); border-top-color:var(--accent);
          border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .error { color:var(--red); padding:40px; text-align:center; }

  .green { color:var(--green); }
  .red   { color:var(--red); }
  .yellow { color:var(--yellow); }
</style>
</head>
<body>
<header>
  <h1>Position Sizer</h1>
  <button class="refresh-btn" onclick="load()">Refresh</button>
</header>
<main id="main">
  <div class="spinner"><div class="spin"></div></div>
</main>

<script>
async function load() {
  const main = document.getElementById('main');
  main.innerHTML = '<div class="spinner"><div class="spin"></div></div>';
  try {
    const res  = await fetch('/sizer/api/regime');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    render(data);
  } catch(e) {
    main.innerHTML = '<div class="error">Failed to load: ' + e.message + '</div>';
  }
}

function multColor(m) {
  if (m >= 1.5) return 'var(--green)';
  if (m >= 1.0) return 'var(--accent)';
  if (m >= 0.75) return 'var(--yellow)';
  return 'var(--red)';
}

function render(d) {
  const m = d.multiplier;
  const raw = d.raw_score;
  // gauge: 0.5x→0%, 2.0x→100%
  const gaugePct = Math.round(((m - 0.5) / 1.5) * 100);
  const gaugeColor = multColor(m);

  const scores = d.scores;
  const inds   = d.indicators;

  const scoreRows = Object.entries(scores).map(([k, v]) => {
    const pct = Math.round(v * 100);
    return `
      <div class="score-row">
        <div class="score-name">${k.replace('score_','').replace('_',' ')}</div>
        <div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>
        <div class="score-val">${v.toFixed(2)}</div>
      </div>`;
  }).join('');

  const indRows = Object.entries(inds).map(([k, v]) => `
    <tr><td>${k.replace(/_/g,' ')}</td><td style="text-align:right">${v.toFixed(3)}</td></tr>
  `).join('');

  document.getElementById('main').innerHTML = `
    <div class="hero">
      <div class="mult-label">Market Regime Risk Multiplier</div>
      <div class="mult-value" style="color:${gaugeColor}">${m.toFixed(2)}x</div>
      <div class="mult-sub">Scale your base position size by this multiplier</div>
      <div class="gauge-wrap">
        <div class="gauge-track">
          <div class="gauge-fill" style="width:${gaugePct}%;background:${gaugeColor}"></div>
        </div>
        <div class="gauge-labels"><span>0.5x (Reduce)</span><span>1.0x (Neutral)</span><span>2.0x (Increase)</span></div>
      </div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="label">Raw Composite</div>
        <div class="value">${raw.toFixed(3)}</div>
        <div class="sub">Weighted average of all signals (0–1)</div>
      </div>
      <div class="card">
        <div class="label">EMA Trend</div>
        <div class="value ${inds.EMA_20 > inds.EMA_50 ? 'green' : 'red'}">
          ${inds.EMA_20 > inds.EMA_50 ? 'Bullish' : 'Bearish'}
        </div>
        <div class="sub">20 EMA ${inds.EMA_20 > inds.EMA_50 ? 'above' : 'below'} 50 EMA</div>
      </div>
      <div class="card">
        <div class="label">SPY RSI (14)</div>
        <div class="value ${inds.RSI > 70 ? 'red' : inds.RSI < 40 ? 'red' : 'green'}">
          ${inds.RSI.toFixed(1)}
        </div>
        <div class="sub">Sweet spot: 55–70</div>
      </div>
      <div class="card">
        <div class="label">Rel Volume</div>
        <div class="value ${inds.RVOL >= 1.5 ? 'green' : 'yellow'}">${inds.RVOL.toFixed(2)}x</div>
        <div class="sub">vs 20-day average</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
      <div>
        <div class="section-title">Component Scores</div>
        ${scoreRows}
      </div>
      <div>
        <div class="section-title">SPY Indicators</div>
        <table>
          <tr><th>Indicator</th><th style="text-align:right">Value</th></tr>
          ${indRows}
        </table>
      </div>
    </div>
  `;
}

load();
</script>
</body>
</html>
"""


@bp.route("/")
def index():
    return _HTML


@bp.route("/api/regime")
def api_regime():
    try:
        regime = ps.get_market_regime()
        return jsonify({
            "multiplier": round(regime.multiplier, 4),
            "raw_score":  round(regime.raw_score, 4),
            "indicators": {k: round(v, 4) for k, v in regime.indicators.items()},
            "scores":     {k: round(v, 4) for k, v in regime.scores.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
