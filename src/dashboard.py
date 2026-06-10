"""Live browser monitor for ModelBreaker: battery plus sweeps in one session.

Launches, in sequence, the robustness battery, the arithmetic sweep, and the
distractor sweep (any phase can be skipped), and streams progress to a read-only
page. Phases share one progress ring; each sweep gets a live K-rate panel with
in-band rows highlighted. Run configuration comes from CLI flags.

Usage:
    python src/dashboard.py
    python src/dashboard.py --no-battery --trials 10
    python src/dashboard.py --no-arith --no-distractor --category long_context

Run from the repository root.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse
import json
import statistics
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from shared.algebra import build_algebra_sweep
from shared.arithmetic import build_arithmetic_sweep
from shared.cases import build_cases
from shared.lmstudio_client import build_client, probe
from shared.reporting import build_summary, write_report
from shared.sweep import build_sweep

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
except ImportError:
    print("[ERR] pip install fastapi uvicorn")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(__file__).parent

DEFAULTS = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen/qwen3-8b",
    "category": "all",
    "context_tokens": 32768,
    "max_tokens": None,
    "steps": [12, 14, 16],
    "arith_instances": 5,
    "arith_ops": "mixed",
    "algebra_factors": [3, 4, 5, 6, 7, 8],
    "algebra_instances": 5,
    "algebra_max_tokens": 4096,
    "arith_max_tokens": 1024,
    "timeout": 120,
    "distractors": [0, 2, 4, 8, 16, 32],
    "trials": 8,
    "temperature": 0.7,
    "band_low": 0.3,
    "band_high": 0.7,
    "port": 7850,
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    """Format a duration as compact minutes and seconds."""
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def preview(text, head=280, tail=160):
    """Shorten a long prompt for display, keeping its start and end."""
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n  ...[{len(text)} chars total]...\n{text[-tail:]}"


def prompt_text(case):
    """Return the user-message content for a case."""
    for message in case.messages:
        if message.get("role") == "user":
            return message["content"]
    return ""

# ─────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────
class SessionState:
    """Thread-safe snapshot of the running session across all phases."""

    def __init__(self):
        self.status = "starting"        # starting | running | done | error
        self.phase = ""
        self.model = DEFAULTS["model"]
        self.total = 0
        self.completed = 0
        self.results = []
        self.transcript = []
        self.summary = None
        self.sweeps = {"arith": [], "distractor": [], "algebra": []}
        self.error = ""
        self.start_time = 0.0
        self.lock = threading.Lock()

    def begin(self, model):
        with self.lock:
            self.status = "running"
            self.phase = "starting"
            self.model = model
            self.total = 0
            self.completed = 0
            self.results = []
            self.transcript = []
            self.summary = None
            self.sweeps = {"arith": [], "distractor": [], "algebra": []}
            self.error = ""
            self.start_time = time.time()

    def to_json(self) -> str:
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            progress = self.completed / self.total if self.total else 0
            return json.dumps({
                "status": self.status,
                "phase": self.phase,
                "model": self.model,
                "total": self.total,
                "completed": self.completed,
                "pct": progress,
                "elapsed": fmt_time(elapsed),
                "results": [asdict(result) for result in self.results],
                "transcript": self.transcript,
                "summary": self.summary,
                "sweeps": self.sweeps,
                "error": self.error,
            })


STATE = SessionState()

# ─────────────────────────────────────────────────────────────
# Phases
# ─────────────────────────────────────────────────────────────
def run_battery_phase(client, model, cases, max_tokens_override):
    """Run the robustness battery, streaming results and a transcript."""
    with STATE.lock:
        STATE.phase = "battery"
    for case in cases:
        if max_tokens_override is not None:
            case.params["max_tokens"] = max_tokens_override
        result = probe(client, model, case.category, case.name, case.messages,
                       validator=case.validator, **case.params)
        with STATE.lock:
            STATE.results.append(result)
            STATE.transcript.append({
                "name": case.name,
                "category": case.category,
                "passed": result.passed,
                "prompt": preview(prompt_text(case)),
                "response": result.response_text or result.error or "(no response)",
            })
            STATE.completed += 1
    with STATE.lock:
        STATE.summary = build_summary(STATE.results)
        return list(STATE.results)


def run_sweep_phase(client, model, variants, phase_name, bucket, trials, temperature, band):
    """Run a sweep phase, K trials per variant, streaming the K-rate per level."""
    with STATE.lock:
        STATE.phase = phase_name
    band_low, band_high = band
    entries = []
    for level, cases in variants:
        passes = 0
        attempts = 0
        latencies = []
        tokens = []
        last_response = ""
        for case in cases:
            prompt_snippet = preview(prompt_text(case))
            for trial_index in range(trials):
                result = probe(client, model, case.category, case.name, case.messages,
                               validator=case.validator, temperature=temperature, **case.params)
                attempts += 1
                if result.passed:
                    passes += 1
                if result.latency_seconds is not None:
                    latencies.append(result.latency_seconds)
                if result.completion_tokens is not None:
                    tokens.append(result.completion_tokens)
                last_response = result.response_text or result.error or ""
                with STATE.lock:
                    STATE.completed += 1
                    STATE.transcript.append({
                        "name": f"{case.name} · trial {trial_index + 1}",
                        "category": case.category,
                        "passed": result.passed,
                        "prompt": prompt_snippet,
                        "response": (result.response_text or result.error or "(no response)")[:600],
                    })
        rate = passes / attempts if attempts else 0.0
        entry = {
            "level": level,
            "passes": passes,
            "trials": attempts,
            "rate": rate,
            "in_band": band_low <= rate <= band_high,
            "med_lat": round(statistics.median(latencies), 2) if latencies else 0.0,
            "med_tok": round(statistics.median(tokens), 1) if tokens else 0.0,
            "sample": last_response[:240],
        }
        entries.append(entry)
        with STATE.lock:
            STATE.sweeps[bucket].append(entry)
    return entries

# ─────────────────────────────────────────────────────────────
# Session runner
# ─────────────────────────────────────────────────────────────
def run_session(args):
    """Build the enabled phases, run them in order, and write reports."""
    STATE.begin(args.model)
    try:
        client = build_client(args.base_url, timeout=args.timeout)
        battery_cases = build_cases(args.category, args.context_tokens, no_think=args.no_think) if args.run_battery else []
        arith_variants = build_arithmetic_sweep(args.steps, instances=args.arith_instances, no_think=args.no_think, max_tokens=args.arith_max_tokens, ops=args.arith_ops) if args.run_arith else []
        distractor_variants = build_sweep(args.distractors, args.context_tokens, no_think=args.no_think) if args.run_distractor else []
        algebra_variants = build_algebra_sweep(args.algebra_factors, instances=args.algebra_instances, no_think=args.no_think, max_tokens=args.algebra_max_tokens) if args.run_algebra else []
    except Exception as exception:
        with STATE.lock:
            STATE.status = "error"
            STATE.error = f"{type(exception).__name__}: {exception}"
        return

    sweep_cases = (sum(len(c) for _, c in arith_variants) + sum(len(c) for _, c in distractor_variants) + sum(len(c) for _, c in algebra_variants))
    sweep_units = sweep_cases * args.trials
    with STATE.lock:
        STATE.total = len(battery_cases) + sweep_units

    band = (args.band_low, args.band_high)

    if battery_cases:
        battery_results = run_battery_phase(client, args.model, battery_cases, args.max_tokens)
        write_report(battery_results, str(ROOT / "reports/dashboard_run.json"),
                     metadata={"model": args.model, "category": args.category,
                               "context_tokens": args.context_tokens, "no_think": args.no_think})

    if arith_variants:
        arith = run_sweep_phase(client, args.model, arith_variants, "arithmetic sweep", "arith",
                                args.trials, args.temperature, band)
        write_sweep_report(arith, str(ROOT / "reports/dashboard_arith.json"), args, "arith")

    if distractor_variants:
        distractor = run_sweep_phase(client, args.model, distractor_variants, "distractor sweep", "distractor",
                                     args.trials, args.temperature, band)
        write_sweep_report(distractor, str(ROOT / "reports/dashboard_distractor.json"), args, "distractor")

    if algebra_variants:
        algebra = run_sweep_phase(client, args.model, algebra_variants, "algebra sweep", "algebra",
                                  args.trials, args.temperature, band)
        write_sweep_report(algebra, str(ROOT / "reports/dashboard_algebra.json"), args, "algebra")

    with STATE.lock:
        STATE.status = "done"
        STATE.phase = "done"


def write_sweep_report(entries, path, args, kind):
    """Write a sweep report with metadata and per-level records."""
    report = {
        "metadata": {
            "model": args.model, "kind": kind, "trials": args.trials,
            "temperature": args.temperature, "band": [args.band_low, args.band_high],
            "no_think": args.no_think, "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": entries,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

# ─────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ModelBreaker — Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {
    --bg:#0f0f0f; --surface:#161616; --border:#252525;
    --text:#e2e2e2; --muted:#666;
    --blue:#378ADD; --green:#1D9E75; --red:#D85A30; --purple:#7F77DD;
    --radius:10px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'SF Mono','JetBrains Mono','Fira Code',monospace; font-size:13px; padding:20px; min-height:100vh; }
  header { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
  header h1 { font-size:13px; font-weight:500; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
  .run-config { margin-left:auto; font-size:11px; color:var(--muted); }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:blink 2s infinite; }
  .dot.done { background:var(--muted); animation:none; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

  .row { display:grid; gap:12px; margin-bottom:12px; }
  .row-2 { grid-template-columns:1fr 1fr; }

  .card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; margin-bottom:12px; }
  .card-label { font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }

  .runtime-row { display:flex; align-items:center; gap:20px; }
  .ring-container { position:relative; width:88px; height:88px; flex-shrink:0; }
  .ring-container svg { width:88px; height:88px; transform:rotate(-90deg); }
  .ring-bg  { fill:none; stroke:#232323; stroke-width:8; }
  .ring-arc { fill:none; stroke:var(--blue); stroke-width:8; stroke-linecap:round; transition:stroke-dashoffset .6s cubic-bezier(.4,0,.2,1); }
  .ring-label { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; pointer-events:none; }
  .ring-done { font-size:15px; font-weight:600; line-height:1; }
  .ring-sep { display:block; width:20px; height:1px; background:#333; margin:4px auto; }
  .ring-total { font-size:11px; color:var(--muted); }
  .runtime-text { flex:1; }
  .runtime-status { font-size:22px; font-weight:600; margin-bottom:4px; }
  .runtime-elapsed { font-size:11px; color:var(--muted); }

  .summary-metrics { display:grid; grid-template-columns:repeat(4,1fr); }
  .summary-metric { padding:0 18px; border-right:1px solid var(--border); }
  .summary-metric:first-child { padding-left:0; }
  .summary-metric:last-child { border-right:none; }
  .summary-label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
  .summary-value { font-size:28px; font-weight:700; }

  .sweep-block { margin-bottom:18px; }
  .sweep-block:last-child { margin-bottom:0; }
  .sweep-title { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:8px; }
  .sweep-row { display:flex; align-items:center; gap:10px; padding:3px 0; font-size:11px; color:var(--muted); }
  .sweep-row.band { color:var(--text); }
  .sweep-level { width:38px; }
  .sweep-bar { flex:1; height:8px; background:#1a1a1a; border-radius:4px; overflow:hidden; }
  .sweep-bar > span { display:block; height:100%; background:var(--blue); transition:width .5s ease; }
  .sweep-row.band .sweep-bar > span { background:var(--purple); }
  .sweep-rate { width:44px; text-align:right; color:var(--text); }
  .sweep-k { width:48px; text-align:right; }

  .chart-wrap { height:220px; position:relative; }

  .transcript { max-height:380px; overflow-y:auto; display:flex; flex-direction:column; gap:14px; }
  .turn { border-left:2px solid var(--border); padding-left:12px; }
  .turn.fail { border-left-color:var(--red); }
  .turn.pass { border-left-color:var(--green); }
  .turn-name { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:6px; }
  .bubble { font-size:11px; line-height:1.5; white-space:pre-wrap; word-break:break-word; padding:8px 10px; border-radius:6px; margin-bottom:5px; }
  .bubble.prompt { background:#0a0a0a; color:#9aa; }
  .bubble.reply { background:#12150f; color:var(--text); }

  .results-table { width:100%; border-collapse:collapse; font-size:11px; }
  .results-table th { text-align:left; color:var(--muted); text-transform:uppercase; font-size:9px; letter-spacing:.08em; padding:7px 10px; border-bottom:1px solid var(--border); }
  .results-table td { padding:7px 10px; border-bottom:1px solid #1a1a1a; vertical-align:top; word-break:break-word; }
  .muted { color:var(--muted); }
  .pass-true { color:var(--green); font-weight:600; }
  .pass-false { color:var(--red); font-weight:600; }
</style>
</head>
<body>

<header>
  <div class="dot" id="statusDot"></div>
  <h1>ModelBreaker &mdash; Monitor</h1>
  <div class="run-config" id="runConfig"></div>
</header>

<div class="row row-2">
  <div class="card">
    <div class="card-label">Runtime</div>
    <div class="runtime-row">
      <div class="ring-container">
        <svg viewBox="0 0 88 88">
          <circle class="ring-bg" cx="44" cy="44" r="36"/>
          <circle class="ring-arc" id="ringArc" cx="44" cy="44" r="36" stroke-dasharray="226.2" stroke-dashoffset="226.2"/>
        </svg>
        <div class="ring-label">
          <div class="ring-done" id="ringDone">0</div>
          <span class="ring-sep"></span>
          <div class="ring-total" id="ringTotal">&mdash;</div>
        </div>
      </div>
      <div class="runtime-text">
        <div class="runtime-status" id="statusText">Starting</div>
        <div class="runtime-elapsed" id="elapsed">0s</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-label">Battery summary</div>
    <div class="summary-metrics">
      <div class="summary-metric"><div class="summary-label">Passed</div><div class="summary-value" style="color:var(--green)" id="sumPassed">&mdash;</div></div>
      <div class="summary-metric"><div class="summary-label">Failed</div><div class="summary-value" style="color:var(--red)" id="sumFailed">&mdash;</div></div>
      <div class="summary-metric"><div class="summary-label">Total</div><div class="summary-value" id="sumTotal">&mdash;</div></div>
      <div class="summary-metric"><div class="summary-label">Median lat</div><div class="summary-value" style="color:var(--blue)" id="sumLat">&mdash;</div></div>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-label">Sweeps (K-rate, purple = in band)</div>
  <div id="sweeps"><div class="muted" style="font-size:11px">Sweep results appear here as each level completes.</div></div>
</div>

<div class="card">
  <div class="card-label">Battery: pass / fail by category</div>
  <div class="chart-wrap"><canvas id="catChart" role="img" aria-label="Pass and fail counts by category">Pass and fail counts by category.</canvas></div>
</div>

<div class="card">
  <div class="card-label">Transcript</div>
  <div class="transcript" id="transcript"><div class="muted" style="font-size:11px">Prompts and responses appear here as the run progresses.</div></div>
</div>

<div class="card">
  <div class="card-label">Battery results</div>
  <table class="results-table">
    <thead><tr><th>Category</th><th>Test</th><th>Result</th><th>Latency</th><th>Finish</th><th>Tokens</th><th>Note</th></tr></thead>
    <tbody id="resultsBody"><tr><td colspan="7" class="muted">Waiting...</td></tr></tbody>
  </table>
</div>

<script>
const CIRC = 2 * Math.PI * 36;
let catChart = null;

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function statusLabel(d) {
  if (d.status === 'running')  return d.phase ? d.phase : 'Running';
  if (d.status === 'done')     return 'Complete';
  if (d.status === 'error')    return 'Error';
  return 'Starting';
}

function renderSweeps(sweeps) {
  const wrap = document.getElementById('sweeps');
  const labels = [['arith', 'Arithmetic — by step count'], ['distractor', 'Distractors — by decoy count'], ['algebra', 'Algebra — by factor count']];
  const blocks = [];
  labels.forEach(([key, title]) => {
    const rows = (sweeps && sweeps[key]) || [];
    if (!rows.length) return;
    const body = rows.map(r => {
      const pct = Math.round(r.rate * 100);
      return `<div class="sweep-row ${r.in_band ? 'band' : ''}">
        <span class="sweep-level">${r.level}</span>
        <span class="sweep-bar"><span style="width:${pct}%"></span></span>
        <span class="sweep-rate">${r.rate.toFixed(2)}</span>
        <span class="sweep-k">${r.passes}/${r.trials}</span>
      </div>`;
    }).join('');
    blocks.push(`<div class="sweep-block"><div class="sweep-title">${title}</div>${body}</div>`);
  });
  wrap.innerHTML = blocks.length ? blocks.join('') : '<div class="muted" style="font-size:11px">No sweeps in this run.</div>';
}

function renderTable(results) {
  const body = document.getElementById('resultsBody');
  if (!results || !results.length) {
    body.innerHTML = '<tr><td colspan="7" class="muted">Waiting...</td></tr>';
    return;
  }
  body.innerHTML = results.map(r => {
    const detail = r.note || r.error || '';
    const cls = r.passed ? 'pass-true' : 'pass-false';
    const mark = r.passed ? 'pass' : 'fail';
    const lat = r.latency_seconds == null ? '\u2014' : r.latency_seconds + 's';
    const finish = r.finish_reason == null ? '\u2014' : escapeHtml(r.finish_reason);
    const tok = r.completion_tokens == null ? '\u2014' : r.completion_tokens;
    return `<tr>
      <td class="muted">${escapeHtml(r.category)}</td>
      <td>${escapeHtml(r.name)}</td>
      <td class="${cls}">${mark}</td>
      <td>${lat}</td>
      <td class="muted">${finish}</td>
      <td>${tok}</td>
      <td class="muted">${escapeHtml(detail)}</td>
    </tr>`;
  }).join('');
}

function renderTranscript(transcript) {
  const box = document.getElementById('transcript');
  if (!transcript || !transcript.length) {
    box.innerHTML = '<div class="muted" style="font-size:11px">Prompts and responses appear here as the run progresses.</div>';
    return;
  }
  box.innerHTML = transcript.map(t => `
    <div class="turn ${t.passed ? 'pass' : 'fail'}">
      <div class="turn-name">${escapeHtml(t.category)} · ${escapeHtml(t.name)}</div>
      <div class="bubble prompt">${escapeHtml(t.prompt)}</div>
      <div class="bubble reply">${escapeHtml(t.response)}</div>
    </div>`).join('');
}

function updateChart(results) {
  const cats = {};
  (results || []).forEach(r => {
    if (!cats[r.category]) cats[r.category] = { pass: 0, fail: 0 };
    if (r.passed) cats[r.category].pass++; else cats[r.category].fail++;
  });
  const labels = Object.keys(cats);
  const passData = labels.map(c => cats[c].pass);
  const failData = labels.map(c => cats[c].fail);

  if (!catChart) {
    catChart = new Chart(document.getElementById('catChart'), {
      type: 'bar',
      data: { labels, datasets: [
        { label: 'Pass', data: passData, backgroundColor: '#1D9E75' },
        { label: 'Fail', data: failData, backgroundColor: '#D85A30' },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
        scales: {
          x: { stacked: true, ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1d1d1d' } },
          y: { stacked: true, ticks: { color: '#555', precision: 0 }, grid: { color: '#1d1d1d' } }
        }
      }
    });
  } else {
    catChart.data.labels = labels;
    catChart.data.datasets[0].data = passData;
    catChart.data.datasets[1].data = failData;
    catChart.update('none');
  }
}

function applyState(d) {
  document.getElementById('statusDot').className = 'dot' + (d.status === 'running' || d.status === 'starting' ? '' : ' done');
  document.getElementById('ringArc').style.strokeDashoffset = CIRC * (1 - (d.pct || 0));
  document.getElementById('ringDone').textContent = d.completed;
  document.getElementById('ringTotal').textContent = d.total || '\u2014';
  document.getElementById('elapsed').textContent = d.elapsed;
  document.getElementById('runConfig').textContent = `${d.model} · ${d.phase || ''}`;

  const statusText = document.getElementById('statusText');
  statusText.textContent = d.status === 'error' ? (d.error || 'Error') : statusLabel(d);
  statusText.style.color = d.status === 'error' ? 'var(--red)' : 'var(--text)';
  statusText.style.fontSize = d.status === 'error' ? '13px' : '22px';

  renderSweeps(d.sweeps);
  renderTable(d.results);
  renderTranscript(d.transcript);
  updateChart(d.results);

  if (d.summary) {
    document.getElementById('sumPassed').textContent = d.summary.passed;
    document.getElementById('sumFailed').textContent = d.summary.failed;
    document.getElementById('sumTotal').textContent  = d.summary.total;
    document.getElementById('sumLat').textContent    = d.summary.latency_median_seconds == null ? '\u2014' : d.summary.latency_median_seconds + 's';
  }
}

const es = new EventSource('/stream');
es.onmessage = e => { try { applyState(JSON.parse(e.data)); } catch (err) { console.error(err); } };
es.onerror   = () => { document.getElementById('statusText').textContent = 'Connection lost — reload'; };
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD_HTML


@app.get("/stream")
def stream():
    def gen():
        while True:
            yield f"data: {STATE.to_json()}\n\n"
            time.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/state")
def state_endpoint():
    return json.loads(STATE.to_json())

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ModelBreaker multi-phase monitor.")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--category", default=DEFAULTS["category"])
    parser.add_argument("--context-tokens", type=int, default=DEFAULTS["context_tokens"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULTS["steps"])
    parser.add_argument("--arith-instances", type=int, default=DEFAULTS["arith_instances"])
    parser.add_argument("--arith-ops", choices=("mixed", "addsub"), default=DEFAULTS["arith_ops"])
    parser.add_argument("--algebra-factors", type=int, nargs="+", default=DEFAULTS["algebra_factors"])
    parser.add_argument("--algebra-instances", type=int, default=DEFAULTS["algebra_instances"])
    parser.add_argument("--algebra-max-tokens", type=int, default=DEFAULTS["algebra_max_tokens"])
    parser.add_argument("--arith-max-tokens", type=int, default=DEFAULTS["arith_max_tokens"])
    parser.add_argument("--timeout", type=int, default=DEFAULTS["timeout"])
    parser.add_argument("--distractors", type=int, nargs="+", default=DEFAULTS["distractors"])
    parser.add_argument("--trials", type=int, default=DEFAULTS["trials"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--band-low", type=float, default=DEFAULTS["band_low"])
    parser.add_argument("--band-high", type=float, default=DEFAULTS["band_high"])
    parser.add_argument("--no-think", action="store_true", help="Append /no_think to each prompt")
    parser.add_argument("--no-battery", action="store_true", help="Skip the robustness battery phase")
    parser.add_argument("--no-arith", action="store_true", help="Skip the arithmetic sweep phase")
    parser.add_argument("--no-distractor", action="store_true", help="Skip the distractor sweep phase")
    parser.add_argument("--no-algebra", action="store_true", help="Skip the algebra sweep phase")
    parser.add_argument("--port", type=int, default=DEFAULTS["port"])
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()
    args.run_battery = not args.no_battery
    args.run_arith = not args.no_arith
    args.run_distractor = not args.no_distractor
    args.run_algebra = not args.no_algebra
    return args


def main():
    args = parse_args()
    url = f"http://localhost:{args.port}"
    print(f"\n  ModelBreaker Monitor → {url}")
    print("  Press Ctrl+C to stop.\n")

    threading.Thread(target=run_session, args=(args,), daemon=True).start()

    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
