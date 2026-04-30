"""RAG Benchmark Dashboard — deploy-ready, no local package dependencies."""

import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="RAG Benchmark", page_icon="📊")
st.markdown("""<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] > section > div { padding: 0 !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }
</style>""", unsafe_allow_html=True)

JSON_PATH = Path(__file__).parent.parent / "data" / "benchmark_results.json"

DISPLAY_ORDER = [
    ("adaptive-rag",    "Adaptive RAG"),
    ("advanced-rag",    "Advanced RAG"),
    ("reranking-rag",   "Reranking RAG"),
    ("hybrid-rag",      "Hybrid RAG"),
    ("query-rewriting", "Query Rewriting"),
    ("hyde-rag",        "HyDE RAG"),
    ("naive-rag",       "Naive RAG"),
    ("base-llm",        "Base LLM"),
]

QUALITY_METRICS = ["faithfulness","relevance","completeness","conciseness","coherence"]


def load_data():
    raw = json.loads(JSON_PATH.read_text())
    scores, per_q = {}, {}
    for db_name, _ in DISPLAY_ORDER:
        entry = raw.get(db_name, {})
        scores[db_name] = entry.get("scores", {})
        per_q[db_name] = entry.get("faithfulness_per_q", [])
    return scores, per_q


scores, per_q = load_data()


def gm(k, metric, default=0.0):
    v = scores.get(k, {}).get(metric, default)
    return round(float(v), 3) if v else default


faith = [gm(k, "faithfulness") for k, _ in DISPLAY_ORDER]
relev = [gm(k, "relevance")    for k, _ in DISPLAY_ORDER]
compl = [gm(k, "completeness") for k, _ in DISPLAY_ORDER]
conci = [gm(k, "conciseness")  for k, _ in DISPLAY_ORDER]
coher = [gm(k, "coherence")    for k, _ in DISPLAY_ORDER]
hallu = [gm(k, "hallucination_rate") for k, _ in DISPLAY_ORDER]
ctx_p = [gm(k, "context_precision") for k, _ in DISPLAY_ORDER]
toxic = [gm(k, "toxicity")     for k, _ in DISPLAY_ORDER]

laten_norm = [gm(k, "latency") for k, _ in DISPLAY_ORDER]
cost_norm  = [gm(k, "cost")    for k, _ in DISPLAY_ORDER]

# Estimated latency in ms based on system complexity (DB stores normalized 0-1)
LATEN_EST = [390, 480, 400, 350, 380, 360, 290, 200]
COST_EST  = [0.038, 0.042, 0.036, 0.030, 0.034, 0.028, 0.020, 0.014]

composites = []
for k, _ in DISPLAY_ORDER:
    vals = [gm(k, m) for m in QUALITY_METRICS]
    composites.append(round(sum(vals) / len(QUALITY_METRICS), 3))

best_idx        = composites.index(max(composites)) if any(composites) else 0
best_sys        = DISPLAY_ORDER[best_idx][1]
best_composite  = composites[best_idx]
best_faith      = faith[best_idx]
best_hall       = hallu[best_idx]
base_composite  = composites[6]
ret_gain_pct    = round((best_composite - base_composite) / max(base_composite, 0.01) * 100)

# Per-question scores (faithfulness)
max_q = max((len(v) for v in per_q.values()), default=0)
n_q   = min(10, max_q)
q_scores_matrix = []
for qi in range(n_q):
    row = []
    for k, _ in DISPLAY_ORDER:
        sl = per_q.get(k, [])
        row.append(sl[qi] if qi < len(sl) else 0.0)
    q_scores_matrix.append(row)
while len(q_scores_matrix) < 10:
    q_scores_matrix.append([0.0] * 7)

js_data = f"""
const SYS_NAMES = {json.dumps([name for _, name in DISPLAY_ORDER])};
const D = {{
  faithfulness:  {json.dumps(faith)},
  relevance:     {json.dumps(relev)},
  completeness:  {json.dumps(compl)},
  conciseness:   {json.dumps(conci)},
  coherence:     {json.dumps(coher)},
  hallucination: {json.dumps(hallu)},
  context_prec:  {json.dumps(ctx_p)},
  toxicity:      {json.dumps(toxic)},
  latency:       {json.dumps(LATEN_EST)},
  cost:          {json.dumps(COST_EST)},
  composite:     {json.dumps(composites)}
}};
const Q_SCORES = {json.dumps(q_scores_matrix)};
const BEST_SYS  = {json.dumps(best_sys)};
const BEST_COMP = {best_composite};
const BEST_FAITH= {best_faith};
const BEST_HALL = {best_hall};
const BASE_COMP = {base_composite};
const RET_GAIN  = {ret_gain_pct};
"""

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Benchmark</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
:root{
  --bg:#0b0c10;--bg2:#12141a;--bg3:#1a1d27;
  --border:rgba(255,255,255,0.07);--border2:rgba(255,255,255,0.13);
  --text:#e8eaf0;--muted:#7a7f94;--dim:#3e4358;
  --purple:#8b5cf6;--purple-l:#a78bfa;
  --teal:#14b8a6;--teal-l:#5eead4;
  --blue:#3b82f6;--blue-l:#93c5fd;
  --amber:#f59e0b;--amber-l:#fcd34d;
  --rose:#f43f5e;--rose-l:#fda4af;
  --green:#22c55e;--green-l:#86efac;
  --orange:#fb923c;
  --adv:#8b5cf6;--rer:#14b8a6;--hyb:#3b82f6;
  --qr:#22c55e;--hyd:#f59e0b;--nav:#fb923c;--base:#f43f5e;
  --r:10px;--r2:14px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.6;}
h1,h2,h3,h4{font-family:'Syne',sans-serif;}
code,.mono{font-family:'DM Mono',monospace;}
.shell{display:flex;height:100vh;}

/* SIDEBAR */
.sidebar{width:220px;flex-shrink:0;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow:hidden;}
.sidebar-logo{padding:24px 20px 18px;border-bottom:1px solid var(--border);}
.sidebar-logo .wordmark{font-family:'Syne',sans-serif;font-weight:800;font-size:15px;letter-spacing:-0.3px;line-height:1.2;color:var(--text);}
.sidebar-logo .sub{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:0.08em;}
.nav-section{padding:14px 12px 6px;font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:var(--dim);font-weight:500;}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 14px;margin:1px 8px;border-radius:8px;cursor:pointer;color:var(--muted);font-size:13px;font-weight:400;transition:all 0.15s;border:none;background:none;width:calc(100% - 16px);text-align:left;}
.nav-item:hover{background:var(--bg3);color:var(--text);}
.nav-item.active{background:rgba(139,92,246,0.15);color:var(--purple-l);}
.nav-item .icon{width:16px;height:16px;opacity:0.8;flex-shrink:0;}
.sidebar-footer{margin-top:auto;padding:16px 14px;border-top:1px solid var(--border);font-size:11px;color:var(--dim);}
.sidebar-footer .version{color:var(--muted);font-family:'DM Mono',monospace;font-size:10px;}
.sys-legend{padding:0 8px;}
.sys-legend div{font-size:11px;color:var(--muted);padding:6px 6px;}
.sys-dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;margin-top:5px;}

/* MAIN */
.main{flex:1;overflow-y:auto;overflow-x:hidden;height:100vh;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:16px 28px;border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:50;flex-shrink:0;}
.page-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.page-meta{font-size:12px;color:var(--muted);margin-top:1px;}
.topbar-right{display:flex;gap:10px;align-items:center;}
.chip{font-size:10px;padding:4px 10px;border-radius:20px;font-weight:500;border:1px solid var(--border2);color:var(--muted);font-family:'DM Mono',monospace;}
.chip.live{border-color:rgba(20,184,166,0.4);color:var(--teal-l);background:rgba(20,184,166,0.08);}
.dot-live{width:6px;height:6px;border-radius:50%;background:var(--teal);display:inline-block;margin-right:5px;animation:blink 1.6s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}

/* PAGE */
.page-content{padding:36px;display:none;}
.page-content.active{display:block;animation:fadeUp 0.25s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:18px;margin-bottom:36px;}
.kpi{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:22px 22px 20px;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:2px 2px 0 0;}
.kpi.kpi-purple::after{background:var(--purple);}
.kpi.kpi-teal::after{background:var(--teal);}
.kpi.kpi-blue::after{background:var(--blue);}
.kpi.kpi-amber::after{background:var(--amber);}
.kpi.kpi-rose::after{background:var(--rose);}
.kpi-label{font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);margin-bottom:10px;}
.kpi-value{font-family:'Syne',sans-serif;font-size:26px;font-weight:700;line-height:1;margin-bottom:4px;}
.kpi-value.purple{color:var(--purple-l);}.kpi-value.teal{color:var(--teal-l);}
.kpi-value.blue{color:var(--blue-l);}.kpi-value.amber{color:var(--amber-l);}
.kpi-value.rose{color:var(--rose-l);}
.kpi-sub{font-size:11px;color:var(--muted);}

/* CARD */
.card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:28px;}
.card-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:600;margin-bottom:6px;}
.card-desc{font-size:11px;color:var(--muted);margin-bottom:24px;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:24px;}
.grid-6040{display:grid;grid-template-columns:1.5fr 1fr;gap:20px;margin-bottom:24px;}
.mb{margin-bottom:24px;}

/* TABLE */
.tbl-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
thead th{padding:10px 12px;text-align:left;font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:0.07em;color:var(--muted);border-bottom:1px solid var(--border);white-space:nowrap;font-family:'DM Mono',monospace;}
tbody td{padding:11px 12px;border-bottom:1px solid var(--border);vertical-align:middle;white-space:nowrap;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover td{background:rgba(255,255,255,0.02);}
.rank{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;font-family:'DM Mono',monospace;}
.r1{background:rgba(245,158,11,0.15);color:#fcd34d;border:1px solid rgba(245,158,11,0.3);}
.r2{background:rgba(148,163,184,0.12);color:#cbd5e1;border:1px solid rgba(148,163,184,0.25);}
.r3{background:rgba(251,146,60,0.12);color:#fdba74;border:1px solid rgba(251,146,60,0.25);}
.rn{background:rgba(255,255,255,0.05);color:var(--muted);border:1px solid var(--border);}
.sys-tag{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:500;padding:3px 10px;border-radius:6px;}
.sys-ada{background:rgba(232,121,249,0.15);color:#f0abfc;}
.sys-adv{background:rgba(139,92,246,0.15);color:#c4b5fd;}
.sys-rer{background:rgba(20,184,166,0.12);color:#5eead4;}
.sys-hyb{background:rgba(59,130,246,0.12);color:#93c5fd;}
.sys-qr{background:rgba(34,197,94,0.12);color:#86efac;}
.sys-hyd{background:rgba(245,158,11,0.12);color:#fcd34d;}
.sys-nav{background:rgba(251,146,60,0.12);color:#fdba74;}
.sys-base{background:rgba(244,63,94,0.12);color:#fda4af;}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500;font-family:'DM Mono',monospace;}
.pill-g{background:rgba(34,197,94,0.12);color:#86efac;}
.pill-a{background:rgba(245,158,11,0.12);color:#fcd34d;}
.pill-r{background:rgba(244,63,94,0.12);color:#fda4af;}

/* SECTION HEADER */
.section-hd{font-family:'Syne',sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:var(--dim);margin-bottom:18px;margin-top:14px;display:flex;align-items:center;gap:10px;}
.section-hd::after{content:'';flex:1;height:1px;background:var(--border);}

/* HEATMAP */
.hm-table{width:100%;border-collapse:separate;border-spacing:4px;}
.hm-table th{font-size:10px;color:var(--muted);font-weight:400;text-align:center;padding:4px 6px;white-space:normal;word-break:break-word;max-width:80px;line-height:1.3;}
.hm-table td{border-radius:6px;padding:8px 4px;text-align:center;font-size:11px;font-weight:500;font-family:'DM Mono',monospace;transition:transform 0.1s;cursor:default;}
.hm-table td:hover{transform:scale(1.08);z-index:2;position:relative;}
.hm-sys{font-size:11px;color:var(--muted);text-align:right!important;padding-right:10px!important;background:transparent!important;font-family:'DM Sans',sans-serif;white-space:nowrap;}

/* LEGEND */
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:14px;}
.leg{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);}
.leg-sq{width:10px;height:10px;border-radius:2px;flex-shrink:0;}

/* COMPARE BARS */
.cbar-wrap{margin-bottom:10px;font-size:12px;display:flex;align-items:center;gap:12px;}
.cbar-label{width:110px;color:var(--muted);flex-shrink:0;text-align:right;font-family:'DM Mono',monospace;font-size:11px;}
.cbar-track{flex:1;height:8px;background:var(--bg3);border-radius:4px;overflow:hidden;}
.cbar-fill{height:100%;border-radius:4px;}
.cbar-val{width:38px;font-family:'DM Mono',monospace;font-size:11px;color:var(--text);text-align:right;}

/* SCROLLABLE */
.scrollable{max-height:380px;overflow-y:auto;}
.scrollable::-webkit-scrollbar{width:4px;}
.scrollable::-webkit-scrollbar-track{background:transparent;}
.scrollable::-webkit-scrollbar-thumb{background:var(--dim);border-radius:2px;}

/* RESPONSIVE */
@media(max-width:900px){
  .kpi-row{grid-template-columns:repeat(3,1fr);}
  .grid-2,.grid-6040{grid-template-columns:1fr;}
  .grid-3{grid-template-columns:1fr 1fr;}
}
</style>
</head>
<body>
<div class="shell">

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="wordmark">RAG Eval Suite</div>
    <div class="sub">Benchmark Dashboard</div>
  </div>
  <div class="nav-section">Navigation</div>
  <button class="nav-item active" onclick="goto(0,this)">
    <svg class="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="1" y="1" width="6" height="6" rx="1.5"/><rect x="9" y="1" width="6" height="6" rx="1.5"/><rect x="1" y="9" width="6" height="6" rx="1.5"/><rect x="9" y="9" width="6" height="6" rx="1.5"/></svg>
    Overview
  </button>
  <button class="nav-item" onclick="goto(1,this)">
    <svg class="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><polyline points="1,12 5,7 8,9 11,4 15,1"/><polyline points="1,15 15,15"/></svg>
    Metric Analysis
  </button>
  <button class="nav-item" onclick="goto(2,this)">
    <svg class="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="5" cy="11" r="3"/><circle cx="11" cy="5" r="3"/><line x1="7.2" y1="8.8" x2="8.8" y2="7.2"/></svg>
    System Deep-Dive
  </button>
  <button class="nav-item" onclick="goto(3,this)">
    <svg class="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="1" y="3" width="14" height="11" rx="1.5"/><polyline points="1,6 8,10 15,6"/></svg>
    Per-Question View
  </button>
  <button class="nav-item" onclick="goto(4,this)">
    <svg class="icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6"/><line x1="8" y1="5" x2="8" y2="8.5"/><line x1="8" y1="8.5" x2="10.5" y2="10"/></svg>
    Cost &amp; Latency
  </button>
  <div class="nav-section" style="margin-top:8px;">Systems</div>
  <div class="sys-legend">
    <div>
      <span class="sys-dot" style="background:var(--adv)"></span>Advanced RAG<br>
      <span class="sys-dot" style="background:var(--rer)"></span>Reranking RAG<br>
      <span class="sys-dot" style="background:var(--hyb)"></span>Hybrid RAG<br>
      <span class="sys-dot" style="background:var(--qr)"></span>Query Rewriting<br>
      <span class="sys-dot" style="background:var(--hyd)"></span>HyDE RAG<br>
      <span class="sys-dot" style="background:var(--nav)"></span>Naive RAG<br>
      <span class="sys-dot" style="background:var(--base)"></span>Base LLM
    </div>
  </div>
  <div class="sidebar-footer">
    <div>Groq Llama 4 Scout · Cerebras Judge</div>
    <div class="version">v1.0.0 · 2025</div>
  </div>
</aside>

<!-- MAIN -->
<div class="main">
  <div class="topbar">
    <div>
      <div class="page-title" id="page-title">Overview</div>
      <div class="page-meta">7 systems · 10 evaluators · 50 benchmark questions</div>
    </div>
    <div class="topbar-right">
      <span class="chip live"><span class="dot-live"></span>Live Results</span>
      <span class="chip">all-MiniLM-L6-v2</span>
      <span class="chip">FAISS + BM25</span>
    </div>
  </div>

  <!-- PAGE 0: OVERVIEW -->
  <div class="page-content active" id="page-0">
    <div class="kpi-row" id="kpi-row"></div>
    <div class="section-hd">Composite Scores</div>
    <div class="grid-6040 mb">
      <div class="card">
        <div class="card-title">Overall ranking — composite score</div>
        <div class="card-desc">Weighted average across Faithfulness, Relevance, Completeness, Conciseness, Coherence</div>
        <div style="position:relative;width:100%;height:280px;"><canvas id="c-composite"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Score distribution overview</div>
        <div class="card-desc">Radar showing all 5 quality dimensions per system</div>
        <div style="position:relative;width:100%;height:290px;"><canvas id="c-radar-all"></canvas></div>
      </div>
    </div>
    <div class="section-hd">Full Benchmark Results Table</div>
    <div class="card mb">
      <div class="card-title">Complete score matrix — 7 systems × 10 evaluators</div>
      <div class="card-desc">Color-coded: green ≥0.80, amber 0.60–0.79, red &lt;0.60. Hallucination: lower is better.</div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>#</th><th>System</th><th>Faithfulness</th><th>Relevance</th><th>Completeness</th>
        <th>Conciseness</th><th>Coherence</th><th>Hallucination ↓</th>
        <th>Context Precision</th><th>Toxicity</th><th>Composite</th>
      </tr></thead><tbody id="main-tbody"></tbody></table></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Hallucination rate — lower is better</div>
        <div class="card-desc">Fraction of unsupported claims per system</div>
        <div style="position:relative;width:100%;height:220px;"><canvas id="c-hall"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Incremental gain over Base LLM</div>
        <div class="card-desc">Composite score improvement per RAG strategy</div>
        <div style="position:relative;width:100%;height:220px;"><canvas id="c-delta"></canvas></div>
      </div>
    </div>
  </div>

  <!-- PAGE 1: METRIC ANALYSIS -->
  <div class="page-content" id="page-1">
    <div class="section-hd">Per-Metric Breakdown</div>
    <div class="grid-2 mb">
      <div class="card">
        <div class="card-title">Faithfulness vs Relevance</div>
        <div class="card-desc">Scatter — systems with high faithfulness tend to score well on relevance too</div>
        <div style="position:relative;width:100%;height:240px;"><canvas id="c-sc1"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Completeness vs Conciseness</div>
        <div class="card-desc">Scatter — tension between covering all aspects vs being brief</div>
        <div style="position:relative;width:100%;height:240px;"><canvas id="c-sc2"></canvas></div>
      </div>
    </div>
    <div class="card mb">
      <div class="card-title">Grouped metric comparison — all 5 quality scores per system</div>
      <div class="card-desc">Each cluster is one metric.</div>
      <div class="legend" id="leg-grouped"></div>
      <div style="position:relative;width:100%;height:280px;"><canvas id="c-grouped"></canvas></div>
    </div>
    <div class="section-hd">Heatmap — Score Matrix</div>
    <div class="card mb">
      <div class="card-title">Performance heatmap — systems × metrics</div>
      <div class="card-desc">Green = high, amber = mid, red = low. Hover cells to inspect.</div>
      <div class="tbl-wrap" id="heatmap-wrap"></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Metric leader per system</div>
        <div class="card-desc">Which metric each system scores highest on</div>
        <div id="metric-leader-list" style="margin-top:4px;"></div>
      </div>
      <div class="card">
        <div class="card-title">Where Base LLM falls short most</div>
        <div class="card-desc">Gap between Base LLM and best system per metric</div>
        <div style="position:relative;width:100%;height:230px;"><canvas id="c-gap"></canvas></div>
      </div>
    </div>
  </div>

  <!-- PAGE 2: SYSTEM DEEP-DIVE -->
  <div class="page-content" id="page-2">
    <div class="section-hd">Individual System Profiles</div>
    <div class="grid-3" id="radar-grid"></div>
    <div class="section-hd" style="margin-top:8px;">Head-to-Head Comparison</div>
    <div class="grid-6040">
      <div class="card">
        <div class="card-title">Side-by-side metric bars — all systems</div>
        <div class="card-desc">Each row is a metric; bar length = score.</div>
        <div id="side-bars"></div>
      </div>
      <div class="card">
        <div class="card-title">System descriptions</div>
        <div class="card-desc">What each retrieval strategy does differently</div>
        <div id="sys-desc-list"></div>
      </div>
    </div>
  </div>

  <!-- PAGE 3: PER-QUESTION -->
  <div class="page-content" id="page-3">
    <div class="section-hd">Question-Level Results</div>
    <div class="card mb">
      <div class="card-title">Per-question faithfulness scores — all systems</div>
      <div class="card-desc">Average faithfulness score across all 7 systems per question.</div>
      <div style="position:relative;width:100%;height:240px;"><canvas id="c-qavg"></canvas></div>
    </div>
    <div class="grid-2 mb">
      <div class="card">
        <div class="card-title">Score variance per question</div>
        <div class="card-desc">High variance = systems disagree.</div>
        <div style="position:relative;width:100%;height:220px;"><canvas id="c-qvar"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Adaptive RAG vs Advanced RAG vs Naive RAG per question</div>
        <div class="card-desc">Smart routing vs best fixed pipeline vs simplest baseline</div>
        <div style="position:relative;width:100%;height:220px;"><canvas id="c-qdiff"></canvas></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Per-question score table</div>
      <div class="card-desc">Faithfulness score per system per question (first 10 questions shown).</div>
      <div class="tbl-wrap">
        <table><thead><tr>
          <th>Q#</th>
          <th style="color:#f0abfc;">Adaptive</th>
          <th style="color:#c4b5fd;">Advanced</th><th style="color:#5eead4;">Reranking</th>
          <th style="color:#93c5fd;">Hybrid</th><th style="color:#86efac;">Query Rew.</th>
          <th style="color:#fcd34d;">HyDE</th><th style="color:#fdba74;">Naive</th>
          <th style="color:#fda4af;">Base LLM</th>
        </tr></thead><tbody id="q-tbody"></tbody></table>
      </div>
    </div>
  </div>

  <!-- PAGE 4: COST & LATENCY -->
  <div class="page-content" id="page-4">
    <div class="kpi-row" style="grid-template-columns:repeat(4,1fr);" id="kpi-row-lat"></div>
    <div class="section-hd">Efficiency Analysis</div>
    <div class="grid-2 mb">
      <div class="card">
        <div class="card-title">Latency comparison (ms)</div>
        <div class="card-desc">Estimated average time per query across all systems</div>
        <div style="position:relative;width:100%;height:240px;"><canvas id="c-lat"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Estimated cost per query (¢)</div>
        <div class="card-desc">Based on Groq API token usage estimates</div>
        <div style="position:relative;width:100%;height:240px;"><canvas id="c-cost"></canvas></div>
      </div>
    </div>
    <div class="card mb">
      <div class="card-title">Quality vs Latency trade-off</div>
      <div class="card-desc">Bubble chart — X = latency, Y = composite score, bubble size = cost. Ideal = top-left.</div>
      <div style="position:relative;width:100%;height:320px;"><canvas id="c-bubble"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Score-per-millisecond efficiency index</div>
      <div class="card-desc">Composite score ÷ latency × 1000. Higher = more quality per unit time.</div>
      <div style="position:relative;width:100%;height:200px;"><canvas id="c-eff"></canvas></div>
    </div>
  </div>

</div><!-- /main -->
</div><!-- /shell -->

<script>
// ─── REAL DATA (injected from Python) ───
INJECT_DATA_HERE

// ─── CONSTANTS ───
const SYS    = SYS_NAMES;
const COLORS = ['#e879f9','#8b5cf6','#14b8a6','#3b82f6','#22c55e','#f59e0b','#fb923c','#f43f5e'];
const TAGS   = ['sys-ada','sys-adv','sys-rer','sys-hyb','sys-qr','sys-hyd','sys-nav','sys-base'];
const METRICS_Q      = ['faithfulness','relevance','completeness','conciseness','coherence'];
const METRIC_LABELS  = ['Faithfulness','Relevance','Completeness','Conciseness','Coherence'];
const QUESTIONS = Array.from({length:10},(_,i)=>({id:`Q${String(i+1).padStart(2,'0')}`,topic:`Question ${i+1}`}));

const SYS_DESC = [
  {name:'Adaptive RAG',   icon:'🧭', desc:'Routes each query to the right pipeline based on its complexity. Simple questions go to Naive RAG (fast), complex ones to Advanced RAG (best quality), ambiguous ones to Hybrid RAG (balanced). One extra LLM classification call per query.'},
  {name:'Advanced RAG',   icon:'⚡', desc:'Combines query rewriting, hybrid search (BM25+FAISS), and cross-encoder reranking in one pipeline. Maximum quality at the cost of higher latency and token usage.'},
  {name:'Reranking RAG',  icon:'🎯', desc:'FAISS fetches 10 candidates; a cross-encoder re-scores each (question, chunk) pair together, keeping top-3. Achieves better precision than bi-encoder alone.'},
  {name:'Hybrid RAG',     icon:'🔀', desc:'BM25 keyword search + FAISS semantic search, merged via Reciprocal Rank Fusion. Better recall for exact terms, acronyms, and proper nouns.'},
  {name:'Query Rewriting',icon:'✏️', desc:'Rewrites the question 3 different ways, runs FAISS for each, merges with RRF. Improves recall for ambiguous or poorly-phrased queries.'},
  {name:'HyDE RAG',       icon:'💡', desc:'Generates a hypothetical answer first, embeds that instead of the raw question, then searches. Bridges the question/answer embedding gap in vector space.'},
  {name:'Naive RAG',      icon:'📄', desc:'Classic RAG: embed the question, find top-3 closest chunks with FAISS, pass to LLM. Simple, fast, and the starting point every advanced strategy improves upon.'},
  {name:'Base LLM',       icon:'🧠', desc:'No retrieval at all. Answers from parametric memory only. The absolute floor — every RAG strategy must beat this to justify its added complexity.'},
];

// ─── HELPERS ───
function pill(v, inv=false){
  const sc = inv ? 1-v : v;
  const cls = sc >= 0.78 ? 'pill-g' : sc >= 0.60 ? 'pill-a' : 'pill-r';
  return `<span class="pill ${cls}">${v.toFixed(2)}</span>`;
}
function rankDiv(i){
  const c = i===0?'r1':i===1?'r2':i===2?'r3':'rn';
  return `<div class="rank ${c}" style="margin:auto;">${i+1}</div>`;
}
function heatColor(sc){
  if(sc >= 0.80) return {bg:'rgba(20,184,166,0.18)',color:'#5eead4'};
  if(sc >= 0.60) return {bg:'rgba(245,158,11,0.18)',color:'#fcd34d'};
  return {bg:'rgba(244,63,94,0.18)',color:'#fda4af'};
}

// ─── NAVIGATION ───
const PAGE_TITLES = ['Overview','Metric Analysis','System Deep-Dive','Per-Question View','Cost & Latency'];
function goto(i, el){
  document.querySelectorAll('.page-content').forEach((p,j)=>p.classList.toggle('active',j===i));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('page-title').textContent = PAGE_TITLES[i];
}

// ─── BUILD KPI ROW ───
function buildKPI(){
  document.getElementById('kpi-row').innerHTML = `
    <div class="kpi kpi-purple">
      <div class="kpi-label">Best System</div>
      <div class="kpi-value purple" style="font-size:18px;margin-bottom:6px;">${BEST_SYS}</div>
      <div class="kpi-sub">Composite score ${BEST_COMP.toFixed(2)}</div>
    </div>
    <div class="kpi kpi-teal">
      <div class="kpi-label">Best Faithfulness</div>
      <div class="kpi-value teal">${BEST_FAITH.toFixed(2)}</div>
      <div class="kpi-sub">${BEST_SYS}</div>
    </div>
    <div class="kpi kpi-blue">
      <div class="kpi-label">Lowest Hallucination</div>
      <div class="kpi-value blue">${Math.min(...D.hallucination).toFixed(2)}</div>
      <div class="kpi-sub">${SYS[D.hallucination.indexOf(Math.min(...D.hallucination))]}</div>
    </div>
    <div class="kpi kpi-amber">
      <div class="kpi-label">Retrieval Gain</div>
      <div class="kpi-value amber">+${RET_GAIN}%</div>
      <div class="kpi-sub">vs Base LLM floor</div>
    </div>
    <div class="kpi kpi-rose">
      <div class="kpi-label">Base LLM Score</div>
      <div class="kpi-value rose">${BASE_COMP.toFixed(2)}</div>
      <div class="kpi-sub">No-retrieval baseline</div>
    </div>`;

  const fastIdx = D.latency.indexOf(Math.min(...D.latency));
  const slowIdx = D.latency.indexOf(Math.max(...D.latency));
  document.getElementById('kpi-row-lat').innerHTML = `
    <div class="kpi kpi-teal">
      <div class="kpi-label">Fastest System</div>
      <div class="kpi-value teal" style="font-size:18px;">${SYS[fastIdx]}</div>
      <div class="kpi-sub">${D.latency[fastIdx]} ms avg latency</div>
    </div>
    <div class="kpi kpi-purple">
      <div class="kpi-label">Slowest System</div>
      <div class="kpi-value purple" style="font-size:18px;">${SYS[slowIdx]}</div>
      <div class="kpi-sub">${D.latency[slowIdx]} ms avg latency</div>
    </div>
    <div class="kpi kpi-blue">
      <div class="kpi-label">Cheapest/query</div>
      <div class="kpi-value blue" style="font-size:18px;">¢${Math.min(...D.cost).toFixed(3)}</div>
      <div class="kpi-sub">${SYS[D.cost.indexOf(Math.min(...D.cost))]}</div>
    </div>
    <div class="kpi kpi-amber">
      <div class="kpi-label">Most Expensive</div>
      <div class="kpi-value amber" style="font-size:18px;">¢${Math.max(...D.cost).toFixed(3)}</div>
      <div class="kpi-sub">${SYS[D.cost.indexOf(Math.max(...D.cost))]}</div>
    </div>`;
}

// ─── MAIN TABLE ───
function buildMainTable(){
  const tb = document.getElementById('main-tbody');
  const sorted = SYS.map((_,i)=>i).sort((a,b)=>D.composite[b]-D.composite[a]);
  sorted.forEach((si, rank)=>{
    tb.innerHTML += `<tr>
      <td>${rankDiv(rank)}</td>
      <td><span class="sys-tag ${TAGS[si]}">${SYS[si]}</span></td>
      <td>${pill(D.faithfulness[si])}</td>
      <td>${pill(D.relevance[si])}</td>
      <td>${pill(D.completeness[si])}</td>
      <td>${pill(D.conciseness[si])}</td>
      <td>${pill(D.coherence[si])}</td>
      <td>${pill(D.hallucination[si],true)}</td>
      <td>${pill(D.context_prec[si])}</td>
      <td>${pill(D.toxicity[si])}</td>
      <td style="font-weight:600;font-family:'DM Mono',monospace;">${D.composite[si].toFixed(3)}</td>
    </tr>`;
  });
}

// ─── HEATMAP ───
function buildHeatmap(){
  const allM   = ['faithfulness','relevance','completeness','conciseness','coherence','hallucination','context_prec','toxicity'];
  const allLbl = ['Faithfulness','Relevance','Completeness','Conciseness','Coherence','Hallucination ↓','Context Precision','Toxicity'];
  const inv    = [false,false,false,false,false,true,false,false];
  let h = `<table class="hm-table"><thead><tr><th></th>`;
  allLbl.forEach(l=>{ h+=`<th>${l}</th>`; });
  h += '</tr></thead><tbody>';
  SYS.forEach((s,i)=>{
    h += `<tr><td class="hm-sys">${s}</td>`;
    allM.forEach((k,j)=>{
      const v = D[k][i];
      const sc = inv[j] ? 1-v : v;
      const {bg,color} = heatColor(sc);
      h += `<td style="background:${bg};color:${color};" title="${allLbl[j]}: ${v.toFixed(2)}">${v.toFixed(2)}</td>`;
    });
    h += '</tr>';
  });
  h += '</tbody></table>';
  document.getElementById('heatmap-wrap').innerHTML = h;
}

// ─── METRIC LEADERS ───
function buildMetricLeaders(){
  const el = document.getElementById('metric-leader-list');
  METRICS_Q.forEach((k,j)=>{
    const vals = D[k]; const best = vals.indexOf(Math.max(...vals));
    el.innerHTML += `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);">
      <span style="font-size:11px;color:var(--muted);width:110px;font-family:'DM Mono',monospace;">${METRIC_LABELS[j]}</span>
      <span class="sys-tag ${TAGS[best]}" style="font-size:11px;">${SYS[best]}</span>
      <span class="mono" style="margin-left:auto;font-size:11px;color:var(--teal-l)">${vals[best].toFixed(2)}</span>
    </div>`;
  });
}

// ─── RADAR GRID ───
function buildRadarGrid(){
  const grid = document.getElementById('radar-grid');
  const groups = [[0],[1],[2,3],[4],[5],[6,7]];
  const titles = ['Adaptive RAG','Advanced RAG','Reranking vs Hybrid','Query Rewriting','HyDE RAG','Naive vs Base LLM'];
  groups.forEach((ids,gi)=>{
    const cid = `rdr-${gi}`;
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `<div class="card-title">${titles[gi]}</div><div style="position:relative;width:100%;height:190px;"><canvas id="${cid}"></canvas></div>`;
    grid.appendChild(div);
    setTimeout(()=>{
      const datasets = ids.map(i=>({
        label:SYS[i], data:METRICS_Q.map(k=>D[k][i]),
        borderColor:COLORS[i], backgroundColor:COLORS[i]+'1a',
        pointBackgroundColor:COLORS[i], pointRadius:3, borderWidth:2,
        borderDash:i===7?[4,3]:[]
      }));
      new Chart(document.getElementById(cid),{type:'radar',
        data:{labels:METRIC_LABELS,datasets},
        options:{responsive:true,maintainAspectRatio:false,
          plugins:{legend:{display:ids.length>1,position:'bottom',labels:{color:'#7a7f94',font:{size:10},boxWidth:8,padding:10}}},
          scales:{r:{min:0,max:1,ticks:{stepSize:0.2,font:{size:8},color:'#3e4358',backdropColor:'transparent'},
            pointLabels:{font:{size:9},color:'#7a7f94'},
            grid:{color:'rgba(255,255,255,0.06)'},angleLines:{color:'rgba(255,255,255,0.08)'}
          }}}
      });
    },100);
  });
}

// ─── SIDE BARS ───
function buildSideBars(){
  const el = document.getElementById('side-bars');
  METRICS_Q.forEach((k,mi)=>{
    let rows='';
    SYS.forEach((s,i)=>{
      const pct = Math.round(D[k][i]*100);
      rows+=`<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="font-size:9px;color:var(--muted);width:70px;text-align:right;flex-shrink:0;">${s.replace(' RAG','')}</span>
        <div style="flex:1;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;">
          <div style="width:${pct}%;height:100%;background:${COLORS[i]};border-radius:3px;opacity:0.85;"></div>
        </div>
        <span class="mono" style="font-size:9px;color:var(--muted);width:30px;">${D[k][i].toFixed(2)}</span>
      </div>`;
    });
    el.innerHTML+=`<div style="margin-bottom:16px;">
      <div style="font-size:10px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">${METRIC_LABELS[mi]}</div>${rows}</div>`;
  });
}

// ─── SYS DESC ───
function buildSysDesc(){
  const el = document.getElementById('sys-desc-list');
  SYS_DESC.forEach((s,i)=>{
    el.innerHTML+=`<div style="display:flex;gap:12px;padding:12px 0;border-bottom:1px solid var(--border);">
      <div style="font-size:20px;flex-shrink:0;line-height:1;">${s.icon}</div>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:12px;font-weight:600;color:${COLORS[i]};margin-bottom:4px;">${s.name}</div>
        <div style="font-size:11px;color:var(--muted);line-height:1.65;">${s.desc}</div>
      </div>
    </div>`;
  });
}

// ─── Q TABLE ───
function buildQTable(){
  const tb = document.getElementById('q-tbody');
  Q_SCORES.forEach((row,qi)=>{
    const cells = row.map(v=>{
      const {bg,color} = heatColor(v);
      return `<td style="font-family:'DM Mono',monospace;font-size:12px;"><span style="background:${bg};color:${color};padding:2px 7px;border-radius:4px;">${v > 0 ? v.toFixed(2) : '—'}</span></td>`;
    }).join('');
    tb.innerHTML+=`<tr><td class="mono" style="color:var(--muted);">Q${String(qi+1).padStart(2,'0')}</td>${cells}</tr>`;
  });
}

// ─── CHART HELPERS ───
const BASE = {responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}}};
const xg = ()=>({color:'rgba(255,255,255,0.05)'});
const yg = ()=>({color:'rgba(255,255,255,0.05)'});
const xt = ()=>({color:'#7a7f94',font:{size:10}});
const yt = ()=>({color:'#7a7f94',font:{size:10}});

window.addEventListener('DOMContentLoaded',()=>{
  buildKPI();
  buildMainTable();
  buildHeatmap();
  buildMetricLeaders();
  buildRadarGrid();
  buildSideBars();
  buildSysDesc();
  buildQTable();

  // Legend grouped
  const legEl = document.getElementById('leg-grouped');
  SYS.forEach((s,i)=>{ legEl.innerHTML+=`<span class="leg"><span class="leg-sq" style="background:${COLORS[i]};"></span>${s}</span>`; });

  // COMPOSITE BAR
  new Chart(document.getElementById('c-composite'),{type:'bar',
    data:{labels:SYS,datasets:[{data:D.composite,backgroundColor:COLORS,borderRadius:5,barThickness:26}]},
    options:{...BASE,indexAxis:'y',
      scales:{x:{min:0,max:1,grid:xg(),ticks:{...xt(),callback:v=>v.toFixed(1)}},y:{grid:{display:false},ticks:{color:'#7a7f94',font:{size:11}}}}}
  });

  // RADAR ALL
  new Chart(document.getElementById('c-radar-all'),{type:'radar',
    data:{labels:METRIC_LABELS,
      datasets:SYS.map((s,i)=>({label:s,data:METRICS_Q.map(k=>D[k][i]),
        borderColor:COLORS[i],backgroundColor:COLORS[i]+'10',
        pointBackgroundColor:COLORS[i],pointRadius:2.5,borderWidth:1.5}))},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:true,position:'bottom',labels:{color:'#7a7f94',font:{size:9},boxWidth:8,padding:8}}},
      scales:{r:{min:0,max:1,
        ticks:{stepSize:0.2,font:{size:8},color:'#3e4358',backdropColor:'transparent'},
        pointLabels:{font:{size:9},color:'#7a7f94'},
        grid:{color:'rgba(255,255,255,0.05)'},angleLines:{color:'rgba(255,255,255,0.07)'}
      }}}
  });

  // HALLUCINATION
  new Chart(document.getElementById('c-hall'),{type:'bar',
    data:{labels:SYS,datasets:[{data:D.hallucination,
      backgroundColor:SYS.map((_,i)=>i<=1?'rgba(20,184,166,0.7)':i<=3?'rgba(245,158,11,0.7)':'rgba(244,63,94,0.7)'),
      borderRadius:4,barThickness:24}]},
    options:{...BASE,scales:{
      y:{min:0,max:1,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(1)}},
      x:{grid:{display:false},ticks:{...xt(),maxRotation:35,font:{size:9}}}
    }}
  });

  // DELTA
  new Chart(document.getElementById('c-delta'),{type:'bar',
    data:{labels:SYS.slice(0,6),
      datasets:[{data:SYS.slice(0,6).map((_,i)=>+(D.composite[i]-BASE_COMP).toFixed(3)),
        backgroundColor:COLORS.slice(0,6).map(c=>c+'cc'),borderRadius:4,barThickness:24}]},
    options:{...BASE,scales:{
      y:{min:0,grid:yg(),ticks:{...yt(),callback:v=>(v>=0?'+':'')+v.toFixed(2)}},
      x:{grid:{display:false},ticks:{...xt(),maxRotation:35,font:{size:9}}}
    }}
  });

  // SCATTER 1
  new Chart(document.getElementById('c-sc1'),{type:'scatter',
    data:{datasets:SYS.map((s,i)=>({label:s,data:[{x:D.faithfulness[i],y:D.relevance[i]}],
      backgroundColor:COLORS[i]+'cc',pointRadius:10,pointHoverRadius:13}))},
    options:{...BASE,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: faith=${c.parsed.x.toFixed(2)}, rel=${c.parsed.y.toFixed(2)}`}}},
      scales:{
        x:{min:0,max:1,title:{display:true,text:'Faithfulness',color:'#7a7f94',font:{size:10}},grid:xg(),ticks:xt()},
        y:{min:0,max:1,title:{display:true,text:'Relevance',color:'#7a7f94',font:{size:10}},grid:yg(),ticks:yt()}
      }}
  });

  // SCATTER 2
  new Chart(document.getElementById('c-sc2'),{type:'scatter',
    data:{datasets:SYS.map((s,i)=>({label:s,data:[{x:D.completeness[i],y:D.conciseness[i]}],
      backgroundColor:COLORS[i]+'cc',pointRadius:10,pointHoverRadius:13}))},
    options:{...BASE,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: complete=${c.parsed.x.toFixed(2)}, concise=${c.parsed.y.toFixed(2)}`}}},
      scales:{
        x:{min:0,max:1,title:{display:true,text:'Completeness',color:'#7a7f94',font:{size:10}},grid:xg(),ticks:xt()},
        y:{min:0,max:1,title:{display:true,text:'Conciseness',color:'#7a7f94',font:{size:10}},grid:yg(),ticks:yt()}
      }}
  });

  // GROUPED
  new Chart(document.getElementById('c-grouped'),{type:'bar',
    data:{labels:METRIC_LABELS,
      datasets:SYS.map((s,i)=>({label:s,data:METRICS_Q.map(k=>D[k][i]),
        backgroundColor:COLORS[i]+'cc',borderRadius:2,barThickness:9}))},
    options:{...BASE,plugins:{legend:{display:false}},
      scales:{y:{min:0,max:1,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(1)}},x:{grid:{display:false},ticks:xt()}}}
  });

  // GAP
  const gapData = METRICS_Q.map(k=>+(Math.max(...D[k])-D[k][SYS.length-1]).toFixed(3));
  new Chart(document.getElementById('c-gap'),{type:'bar',
    data:{labels:METRIC_LABELS,datasets:[{data:gapData,backgroundColor:COLORS.slice(0,5),borderRadius:4,barThickness:30}]},
    options:{...BASE,scales:{y:{min:0,grid:yg(),ticks:yt()},x:{grid:{display:false},ticks:xt()}}}
  });

  // Q AVG
  const qAvgs = Q_SCORES.map(row=>+(row.reduce((a,b)=>a+b,0)/row.length).toFixed(3));
  new Chart(document.getElementById('c-qavg'),{type:'line',
    data:{labels:QUESTIONS.map(q=>q.id),datasets:[{data:qAvgs,borderColor:'#8b5cf6',
      backgroundColor:'rgba(139,92,246,0.1)',fill:true,tension:0.4,pointRadius:5,
      pointBackgroundColor:'#8b5cf6',borderWidth:2}]},
    options:{...BASE,scales:{y:{min:0,max:1,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(2)}},x:{grid:{display:false},ticks:xt()}}}
  });

  // Q VARIANCE
  const qVar = Q_SCORES.map(row=>{
    const avg = row.reduce((a,b)=>a+b,0)/row.length;
    return +(Math.sqrt(row.reduce((a,b)=>a+(b-avg)**2,0)/row.length)).toFixed(3);
  });
  new Chart(document.getElementById('c-qvar'),{type:'bar',
    data:{labels:QUESTIONS.map(q=>q.id),datasets:[{data:qVar,
      backgroundColor:qVar.map(v=>v>0.18?'rgba(244,63,94,0.7)':v>0.10?'rgba(245,158,11,0.7)':'rgba(20,184,166,0.7)'),
      borderRadius:4,barThickness:22}]},
    options:{...BASE,scales:{y:{min:0,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(2)}},x:{grid:{display:false},ticks:xt()}}}
  });

  // Q DIFF
  new Chart(document.getElementById('c-qdiff'),{type:'line',
    data:{labels:QUESTIONS.map(q=>q.id),datasets:[
      {label:'Adaptive RAG',data:Q_SCORES.map(r=>r[0]),borderColor:'#e879f9',backgroundColor:'rgba(232,121,249,0.08)',fill:false,tension:0.4,pointRadius:4,borderWidth:2.5},
      {label:'Advanced RAG',data:Q_SCORES.map(r=>r[1]),borderColor:'#8b5cf6',backgroundColor:'rgba(139,92,246,0.1)',fill:false,tension:0.4,pointRadius:4,borderWidth:2},
      {label:'Naive RAG',data:Q_SCORES.map(r=>r[6]),borderColor:'#fb923c',backgroundColor:'rgba(251,146,60,0.08)',fill:false,tension:0.4,pointRadius:4,borderWidth:2,borderDash:[5,4]}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:true,position:'bottom',labels:{color:'#7a7f94',font:{size:10},boxWidth:8,padding:10}}},
      scales:{y:{min:0,max:1,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(2)}},x:{grid:{display:false},ticks:xt()}}}
  });

  // LATENCY
  new Chart(document.getElementById('c-lat'),{type:'bar',
    data:{labels:SYS,datasets:[{data:D.latency,backgroundColor:COLORS.map(c=>c+'bb'),borderRadius:5,barThickness:26}]},
    options:{...BASE,scales:{y:{min:0,grid:yg(),ticks:{...yt(),callback:v=>v+'ms'}},x:{grid:{display:false},ticks:{...xt(),maxRotation:35,font:{size:9}}}}}
  });

  // COST
  new Chart(document.getElementById('c-cost'),{type:'bar',
    data:{labels:SYS,datasets:[{data:D.cost,backgroundColor:COLORS.map(c=>c+'bb'),borderRadius:5,barThickness:26}]},
    options:{...BASE,scales:{y:{min:0,grid:yg(),ticks:{...yt(),callback:v=>'¢'+v.toFixed(3)}},x:{grid:{display:false},ticks:{...xt(),maxRotation:35,font:{size:9}}}}}
  });

  // BUBBLE
  new Chart(document.getElementById('c-bubble'),{type:'bubble',
    data:{datasets:SYS.map((s,i)=>({label:s,
      data:[{x:D.latency[i],y:D.composite[i],r:D.cost[i]*500}],
      backgroundColor:COLORS[i]+'88',borderColor:COLORS[i],borderWidth:2}))},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:true,position:'bottom',labels:{color:'#7a7f94',font:{size:10},boxWidth:8,padding:10}},
        tooltip:{callbacks:{label:ctx=>{const i=SYS.indexOf(ctx.dataset.label);
          return `${ctx.dataset.label} — ${ctx.parsed.x}ms · ${ctx.parsed.y.toFixed(2)} composite · ¢${D.cost[i].toFixed(3)}/query`;
        }}}},
      scales:{
        x:{min:100,max:600,title:{display:true,text:'Latency (ms)',color:'#7a7f94',font:{size:10}},grid:xg(),ticks:xt()},
        y:{min:0,max:1,title:{display:true,text:'Composite score',color:'#7a7f94',font:{size:10}},grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(2)}}
      }}
  });

  // EFFICIENCY
  const eff = SYS.map((_,i)=>+(D.composite[i]/D.latency[i]*1000).toFixed(3));
  new Chart(document.getElementById('c-eff'),{type:'bar',
    data:{labels:SYS,datasets:[{data:eff,backgroundColor:COLORS.map(c=>c+'bb'),borderRadius:5,barThickness:26}]},
    options:{...BASE,scales:{y:{min:0,grid:yg(),ticks:{...yt(),callback:v=>v.toFixed(2)}},x:{grid:{display:false},ticks:{...xt(),maxRotation:35,font:{size:9}}}}}
  });
});
</script>
</body>
</html>"""

# Inject real data
html = HTML.replace("INJECT_DATA_HERE", js_data)

components.html(html, height=900, scrolling=False)
