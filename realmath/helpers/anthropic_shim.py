"""
Drop-in replacement for an openai.OpenAI() client that calls Anthropic's
Claude API underneath. Only the surface used by the RealMath helpers is
implemented: client.chat.completions.create(...).choices[0].message.content

The shim translates an OpenAI-style call into an Anthropic messages call:
  - the system message is pulled out and passed as Anthropic's system= param
  - remaining messages are forwarded as user/assistant turns
  - response_format={"type": "json_object"} is honored by instructing the
    model to emit only JSON and by stripping fences / extracting the object
  - the incoming OpenAI model string is ignored; a Claude model is used
  - max_tokens is supplied (Anthropic requires it; OpenAI calls here omit it)

When JSON is requested the shim validates the output with json.loads and
retries internally a couple of times, falling back to "{}" so the callers'
key-presence checks and iteration guards degrade gracefully instead of
raising.
"""

import os
import re
import json

import anthropic

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
DEBUG = bool(os.getenv("ANTHROPIC_SHIM_DEBUG"))


def _dbg(label, text):
    if DEBUG:
        import sys
        snippet = text if len(text) <= 600 else text[:600] + " ...[truncated]"
        print(f"[shim] {label}: {snippet!r}", file=sys.stderr)

JSON_ONLY_INSTRUCTION = (
    "Respond with only a single valid JSON object and nothing else. "
    "Do not include any prose before or after it, and do not wrap it in "
    "markdown code fences."
)


def _extract_json(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    if not t.startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start:end + 1]
    return t


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._create(**kwargs)


class _Chat:
    def __init__(self, parent):
        self.completions = _Completions(parent)


class AnthropicOpenAIShim:
    def __init__(self, api_key=None, model=None, max_tokens=None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.chat = _Chat(self)

    def _call(self, system, messages, max_tokens):
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or None,
            messages=messages,
        )
        return "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    def _create(self, model=None, messages=None, response_format=None,
                max_tokens=None, **kwargs):
        messages = messages or []
        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        forwarded = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"
        ]
        want_json = bool(response_format) and response_format.get("type") == "json_object"
        mt = max_tokens or self._max_tokens

        if not want_json:
            return _Response(self._call(system, forwarded, mt))

        sys_json = (system + "\n\n" + JSON_ONLY_INSTRUCTION).strip()
        text = ""
        for attempt in range(3):
            text = self._call(sys_json, forwarded, mt)
            _dbg(f"raw attempt {attempt}", text)
            candidate = _extract_json(text)
            try:
                json.loads(candidate)
                return _Response(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        _dbg("fallback after 3 unparseable replies, last raw", text)
        return _Response("{}")
