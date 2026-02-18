"""Market News blueprint — wraps market_news.py in a web UI."""

from flask import Blueprint, jsonify
import market_news as mn

LABEL   = "Market News"
ICON    = "📰"
SECTION = "Research"
ORDER   = 1

PREFIX = "/news"
bp = Blueprint("news", __name__, url_prefix=PREFIX)

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market News</title>
<style>
  :root { --bg:#0d1117; --surface:#161b22; --border:#30363d;
          --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff;
          --green:#3fb950; --red:#f85149; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--text); min-height:100vh; }
  header { background:var(--surface); border-bottom:1px solid var(--border);
           padding:12px 24px; display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:18px; font-weight:600; }
  .refresh-btn { padding:6px 14px; background:var(--surface); border:1px solid var(--border);
                 border-radius:6px; color:var(--muted); font-size:12px; cursor:pointer; }
  .refresh-btn:hover { color:var(--text); border-color:var(--muted); }
  main { padding:24px; max-width:860px; margin:0 auto; }
  .story { background:var(--surface); border:1px solid var(--border); border-radius:8px;
           padding:16px 20px; margin-bottom:10px; transition:border-color 0.15s; }
  .story:hover { border-color:var(--accent); }
  .story a { text-decoration:none; color:inherit; display:block; }
  .story-title { font-size:15px; font-weight:600; line-height:1.4; color:var(--text); margin-bottom:6px; }
  .story-meta  { font-size:12px; color:var(--muted); display:flex; gap:12px; align-items:center; }
  .story-source { color:var(--accent); }
  .dot { color:var(--border); }
  .story-rank { display:inline-block; width:22px; height:22px; border-radius:50%;
                background:rgba(88,166,255,0.15); color:var(--accent);
                font-size:11px; font-weight:700; text-align:center; line-height:22px;
                flex-shrink:0; margin-right:4px; }
  .spinner { display:flex; align-items:center; justify-content:center; height:200px; }
  .spin { width:28px; height:28px; border:3px solid var(--border); border-top-color:var(--accent);
          border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .error { color:var(--red); padding:40px; text-align:center; }
  .sub { font-size:12px; color:var(--muted); margin-top:8px; }
  #count { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <div>
    <h1>Market News</h1>
    <div class="sub" id="count"></div>
  </div>
  <button class="refresh-btn" onclick="load(true)">Refresh</button>
</header>
<main id="main">
  <div class="spinner"><div class="spin"></div></div>
</main>

<script>
let _stories = [];

async function load(force) {
  const main = document.getElementById('main');
  main.innerHTML = '<div class="spinner"><div class="spin"></div></div>';
  try {
    const res  = await fetch('/news/api/stories?n=20' + (force ? '&force=1' : ''));
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    _stories = data;
    render();
  } catch(e) {
    main.innerHTML = '<div class="error">Failed to load news: ' + e.message + '</div>';
  }
}

function render() {
  const main = document.getElementById('main');
  if (!_stories.length) {
    main.innerHTML = '<div class="error">No stories found.</div>';
    return;
  }
  document.getElementById('count').textContent =
    _stories.length + ' stories · auto-refreshes every 5 min';
  main.innerHTML = _stories.map((s, i) => `
    <div class="story">
      <a href="${s.url}" target="_blank" rel="noopener">
        <div class="story-title">
          <span class="story-rank">${i+1}</span>${s.title}
        </div>
        <div class="story-meta">
          <span class="story-source">${s.source}</span>
          <span class="dot">·</span>
          <span>${s.age || 'unknown'}</span>
          ${s.relevance ? '<span class="dot">·</span><span>relevance ' + (s.relevance*100).toFixed(0) + '%</span>' : ''}
        </div>
      </a>
    </div>
  `).join('');
}

load();
setInterval(() => load(), 5 * 60 * 1000);
</script>
</body>
</html>
"""


@bp.route("/")
def index():
    return _HTML


@bp.route("/api/stories")
def api_stories():
    from flask import request as req
    n = int(req.args.get("n", 20))
    try:
        stories = mn.get_top_stories(n)
        out = []
        for s in stories:
            out.append({
                "title":     s.title,
                "source":    s.source,
                "url":       s.url,
                "age":       s.age_label,
                "relevance": round(s.relevance, 3),
                "origin":    s.origin,
            })
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
