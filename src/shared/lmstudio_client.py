"""Client and single-probe primitive for talking to an LM Studio server.

Usage:
    from shared.lmstudio_client import build_client, probe
    client = build_client("http://localhost:1234/v1")
    result = probe(client, "qwen/qwen3-8b", "edge", "empty_string", messages=[...])
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import time
from dataclasses import dataclass
from typing import Callable, Optional

from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "not-needed"
DEFAULT_TIMEOUT = 120

# ─────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────
@dataclass
class ProbeResult:
    category: str
    name: str
    passed: bool
    latency_seconds: Optional[float] = None
    finish_reason: Optional[str] = None
    response_chars: Optional[int] = None
    completion_tokens: Optional[int] = None
    response_text: str = ""
    note: str = ""
    error: str = ""


Validator = Callable[[str], tuple]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def build_client(base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT):
    """Return an OpenAI client pointed at a local LM Studio server."""
    return OpenAI(base_url=base_url, api_key=DEFAULT_API_KEY, timeout=timeout, max_retries=0)


def probe(client, model, category, name, messages, validator=None, **params):
    """Send one request and capture timing, finish reason, and any failure.

    A transport or server exception is recorded rather than raised, so a hard
    crash and a graceful-but-wrong response are both visible in the report.
    """
    started_at = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model, messages=messages, **params)
    except Exception as exception:
        return ProbeResult(
            category=category,
            name=name,
            passed=False,
            latency_seconds=round(time.perf_counter() - started_at, 2),
            error=f"{type(exception).__name__}: {exception}",
        )

    latency_seconds = round(time.perf_counter() - started_at, 2)
    choice = response.choices[0]
    response_text = choice.message.content or ""
    usage = getattr(response, "usage", None)
    result = ProbeResult(
        category=category,
        name=name,
        passed=True,
        latency_seconds=latency_seconds,
        finish_reason=choice.finish_reason,
        response_chars=len(response_text),
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        response_text=response_text,
    )
    if validator is not None:
        passed, note = validator(response_text)
        result.passed = passed
        result.note = note
    return result
