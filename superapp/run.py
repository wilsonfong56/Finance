#!/usr/bin/env python3
"""
Finance Super App
=================
Single Flask server (port 5000) that hosts every trading tool in one
sidebar-nav + iframe shell.  Each tool is a Blueprint discovered from
superapp/modules/*.py — any file that exports a `bp` Flask Blueprint
gets registered automatically.

Usage:
  python superapp/run.py
"""

import importlib.util
import os
import sys

# ── Path setup ───────────────────────────────────────────────────────────────
_SELF    = os.path.dirname(os.path.abspath(__file__))   # superapp/
_ROOT    = os.path.dirname(_SELF)                        # Finance/
_LIBDIR  = os.path.join(_SELF, "lib")                   # superapp/lib/
_SCRDIR  = os.path.join(_ROOT, "Screeners")             # Finance/Screeners/
_JRNDIR  = os.path.join(_ROOT, "Journal")               # Finance/Journal/

for _p in [_LIBDIR, _SCRDIR, _JRNDIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load env before any module that calls os.environ ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

# ── Flask app ─────────────────────────────────────────────────────────────────
from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder=os.path.join(_SELF, "templates"))
CORS(app)

# ── Auto-discover blueprints from modules/ ────────────────────────────────────
_MOD_DIR   = os.path.join(_SELF, "modules")
NAV_ITEMS  = []   # [{id, label, icon, url, section, order}]

for _fname in sorted(os.listdir(_MOD_DIR)):
    if _fname.startswith("_") or not _fname.endswith(".py"):
        continue
    _mod_id  = _fname[:-3]
    _spec    = importlib.util.spec_from_file_location(
        f"superapp_mod_{_mod_id}",
        os.path.join(_MOD_DIR, _fname),
    )
    _mod = importlib.util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_mod)
    except Exception as _e:
        print(f"[superapp] failed to load module {_fname}: {_e}")
        continue

    if not hasattr(_mod, "bp"):
        continue

    app.register_blueprint(_mod.bp)
    NAV_ITEMS.append({
        "id":      _mod_id,
        "label":   getattr(_mod, "LABEL",   _mod_id.replace("_", " ").title()),
        "icon":    getattr(_mod, "ICON",    ""),
        "url":     f"/{_mod_id}/",
        "section": getattr(_mod, "SECTION", "Tools"),
        "order":   getattr(_mod, "ORDER",   99),
    })
    print(f"[superapp] registered /{_mod_id}/  ({_mod.LABEL if hasattr(_mod, 'LABEL') else _mod_id})")

NAV_ITEMS.sort(key=lambda x: (x["section"], x["order"]))

# ── Shell ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    first_url = NAV_ITEMS[0]["url"] if NAV_ITEMS else "/"
    return render_template("shell.html", nav_items=NAV_ITEMS, default_url=first_url)


if __name__ == "__main__":
    print("[superapp] Starting Finance Suite on http://localhost:5000")
    app.run(port=5000, debug=False, threaded=True)
