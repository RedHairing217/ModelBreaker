"""Manual chat dashboard for a local LM Studio model.

A browser chat box with an input field and a multi-turn session held in the
server process. Runs no battery. A dropdown loads any automated battery prompt
into the input so you can read and tweak it before sending.

Usage:
    python src/chat_dashboard.py
    python src/chat_dashboard.py --model qwen/qwen3-8b --port 7851

Run from the repository root.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import argparse
import sys
import threading
import time
import webbrowser

from shared.cases import build_cases
from shared.lmstudio_client import build_client

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
except ImportError:
    print("[ERR] pip install fastapi uvicorn")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Constants and defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "base_url": "http://127.0.0.1:1234/v1",
    "model": "qwen/qwen3-8b",
    "context_tokens": 32768,
    "temperature": 0.7,
    "max_tokens": 1024,
    "port": 7851,
}

# ─────────────────────────────────────────────────────────────
# Chat session
# ─────────────────────────────────────────────────────────────
class ChatSession:
    """Multi-turn conversation state held in the dashboard process."""

    def __init__(self, base_url, model, temperature, max_tokens):
        self.client = build_client(base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        self.lock = threading.Lock()

    def send(self, content):
        """Append a user turn, query the model with full history, append reply."""
        with self.lock:
            self.messages.append({"role": "user", "content": content})
            history = list(self.messages)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=history,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            choice = response.choices[0]
            reply = choice.message.content or ""
            usage = getattr(response, "usage", None)
            meta = {
                "finish_reason": choice.finish_reason,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            }
        except Exception as exception:
            reply = f"[ERR] {type(exception).__name__}: {exception}"
            meta = {"finish_reason": "error", "completion_tokens": None}
        with self.lock:
            self.messages.append({"role": "assistant", "content": reply, **meta})
            return list(self.messages)

    def reset(self):
        with self.lock:
            self.messages = []

    def snapshot(self):
        with self.lock:
            return list(self.messages)


SESSION = None

# ─────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────
CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ModelBreaker — Manual Chat</title>
<style>
  :root {
    --bg:#0f0f0f; --surface:#161616; --border:#252525;
    --text:#e2e2e2; --muted:#666;
    --blue:#378ADD; --green:#1D9E75; --red:#D85A30;
    --radius:10px;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:'SF Mono','JetBrains Mono','Fira Code',monospace; font-size:13px; padding:20px; min-height:100vh; }
  header { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
  header h1 { font-size:13px; font-weight:500; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
  .run-config { margin-left:auto; font-size:11px; color:var(--muted); }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; margin-bottom:12px; }
  .card-label { font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:14px; }

  .transcript { max-height:55vh; overflow-y:auto; display:flex; flex-direction:column; gap:12px; margin-bottom:14px; }
  .turn { display:flex; flex-direction:column; gap:4px; }
  .role { font-size:9px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  .bubble { font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; padding:9px 12px; border-radius:8px; }
  .bubble.user { background:#0a0a0a; align-self:flex-end; max-width:80%; }
  .bubble.assistant { background:#12150f; max-width:90%; }
  .bubble.assistant.error { background:#1a0e0c; color:var(--red); }
  .meta { font-size:9px; color:var(--muted); }

  .controls { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; }
  textarea { flex:1; min-height:64px; background:#0a0a0a; border:1px solid var(--border); color:var(--text); border-radius:6px; padding:9px 11px; font-family:inherit; font-size:12px; resize:vertical; }
  textarea:focus { outline:none; border-color:var(--blue); }
  select, .btn { background:#0a0a0a; border:1px solid var(--border); color:var(--text); border-radius:6px; padding:8px 12px; font-family:inherit; font-size:11px; }
  .btn { background:var(--blue); color:#fff; border:none; text-transform:uppercase; letter-spacing:.06em; cursor:pointer; }
  .btn.ghost { background:transparent; color:var(--muted); }
  .btn:disabled { background:#2a2a2a; color:#666; cursor:not-allowed; }
  .toolbar { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
</style>
</head>
<body>

<header>
  <h1>ModelBreaker &mdash; Manual Chat</h1>
  <div class="run-config" id="runConfig"></div>
</header>

<div class="card">
  <div class="toolbar">
    <select id="caseSelect"><option value="">Load an automated prompt...</option></select>
    <button class="btn ghost" onclick="resetSession()">Reset session</button>
  </div>
  <div class="transcript" id="transcript"><div class="muted" style="font-size:11px">Send a message to start.</div></div>
  <div class="controls">
    <textarea id="input" placeholder="Type a prompt and press Ctrl+Enter..."></textarea>
    <button class="btn" id="sendBtn" onclick="send()">Send</button>
  </div>
</div>

<script>
let cases = {};

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderTranscript(messages) {
  const box = document.getElementById('transcript');
  if (!messages || !messages.length) {
    box.innerHTML = '<div class="muted" style="font-size:11px">Send a message to start.</div>';
    return;
  }
  box.innerHTML = messages.map(m => {
    const isUser = m.role === 'user';
    const err = m.finish_reason === 'error' ? ' error' : '';
    const meta = (!isUser && m.completion_tokens != null)
      ? `<div class="meta">${m.finish_reason} · ${m.completion_tokens} tok</div>` : '';
    return `<div class="turn">
      <div class="role">${m.role}</div>
      <div class="bubble ${isUser ? 'user' : 'assistant'}${err}">${escapeHtml(m.content)}</div>
      ${meta}
    </div>`;
  }).join('');
  box.scrollTop = box.scrollHeight;
}

async function send() {
  const input = document.getElementById('input');
  const content = input.value.trim();
  if (!content) return;
  const btn = document.getElementById('sendBtn');
  btn.disabled = true; btn.textContent = '...';
  try {
    const r = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content }) });
    const j = await r.json();
    renderTranscript(j.messages);
    input.value = '';
  } finally {
    btn.disabled = false; btn.textContent = 'Send';
  }
}

async function resetSession() {
  await fetch('/reset', { method: 'POST' });
  renderTranscript([]);
}

function loadCase() {
  const name = document.getElementById('caseSelect').value;
  if (name && cases[name] != null) document.getElementById('input').value = cases[name];
}

document.getElementById('input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) send();
});
document.getElementById('caseSelect').addEventListener('change', loadCase);

async function init() {
  const cfg = await (await fetch('/config')).json();
  document.getElementById('runConfig').textContent = `${cfg.model} · temp ${cfg.temperature}`;
  const data = await (await fetch('/cases')).json();
  cases = data.cases;
  const sel = document.getElementById('caseSelect');
  Object.keys(cases).forEach(name => {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  });
  renderTranscript(await (await fetch('/session')).json());
}
init();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI()
CASE_PROMPTS = {}
CONFIG = {}


@app.get("/", response_class=HTMLResponse)
def index():
    return CHAT_HTML


@app.get("/config")
def config():
    return CONFIG


@app.get("/cases")
def cases():
    return {"cases": CASE_PROMPTS}


@app.get("/session")
def session():
    return SESSION.snapshot()


@app.post("/chat")
async def chat(body: dict):
    return {"messages": SESSION.send(body.get("content", ""))}


@app.post("/reset")
async def reset():
    SESSION.reset()
    return {"ok": True}

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def collect_case_prompts(context_tokens):
    """Map each automated case name to its user-message prompt."""
    prompts = {}
    for case in build_cases("all", context_tokens):
        for message in case.messages:
            if message.get("role") == "user":
                prompts[case.name] = message["content"]
    return prompts


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ModelBreaker manual chat dashboard.")
    parser.add_argument("--base-url", default=DEFAULTS["base_url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--context-tokens", type=int, default=DEFAULTS["context_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--port", type=int, default=DEFAULTS["port"])
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser window")
    return parser.parse_args()


def main():
    global SESSION, CASE_PROMPTS, CONFIG
    args = parse_args()
    SESSION = ChatSession(args.base_url, args.model, args.temperature, args.max_tokens)
    CASE_PROMPTS = collect_case_prompts(args.context_tokens)
    CONFIG = {"model": args.model, "temperature": args.temperature}

    url = f"http://localhost:{args.port}"
    print(f"\n  ModelBreaker Manual Chat → {url}")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")


if __name__ == "__main__":
    main()
