"""Thin client for Apertus behind any OpenAI-compatible endpoint.

Robustness matters more than features here:
- retries with backoff on transient errors;
- Public AI occasionally routes a request to a backend that rejects the system
  role (HTTP 400); Featherless (also reachable through the Hugging Face router)
  rejects it always, and any assistant turn as well. After such a rejection the
  client folds the system prompt into the first user message, and if needed the
  whole conversation into one user message, for the rest of the session;
- JSON extraction tolerant to code fences and leading/trailing chatter.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from openai import APIStatusError, OpenAI

from .config import settings

log = logging.getLogger(__name__)
_log_lock = threading.Lock()


def _log_call(model: str, mode: int, messages: list[dict], seconds: float, reply: str | None = None,
              error: Exception | None = None) -> None:
    """Append one request and its reply (or error) to LLM_LOG as JSON lines. Off unless LLM_LOG is set."""
    if not settings.llm_log:
        return
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "mode": mode,
              "seconds": round(seconds, 2), "messages": messages, "reply": reply,
              "error": str(error)[:500] if error else None}
    path = Path(settings.llm_log)
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMReply:
    text: str
    seconds: float
    model: str


class ApertusClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 fallback: str = ""):
        self.model = model or settings.llm_name
        self.fallback = fallback  # used for the rest of the session once self.model is throttled (HTTP 429)
        key = api_key or settings.llm_api_key
        if not key:
            raise LLMError("LLM_API_KEY is not set. See .env.example.")
        self.client = OpenAI(base_url=base_url or settings.llm_base_url, api_key=key,
                             timeout=settings.llm_timeout, max_retries=0)
        # The app fires several short calls in the background; extra ones wait here instead of being refused.
        self._slots = threading.BoundedSemaphore(max(1, settings.llm_concurrency))
        # Message format the endpoint accepts, learnt from its HTTP 400s and kept for the session:
        # 0 = as is, 1 = system prompt folded into the first user turn, 2 = whole conversation in one user turn.
        # Featherless is known to need 2, which saves two rejected requests (5-10 s) after every start.
        self.mode = 2 if self.model.endswith(":featherless-ai") else 0

    def chat(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 900,
             attempts: int = 6) -> LLMReply:
        last: Exception | None = None
        mode = self.mode
        for attempt in range(attempts + 3):
            if attempt == attempts and mode == 0:  # plain retries exhausted: fold system prompt into the user turn
                mode = 1
                log.warning("Falling back to system-prompt folding after %d failures", attempts)
            msgs = (messages, fold_system(messages), flatten_conversation(messages))[mode]
            t0 = time.time()
            try:
                with self._slots:
                    t0 = time.time()  # time the request itself, not the wait for a slot
                    r = self.client.chat.completions.create(
                        model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise LLMError("empty response")
                _log_call(self.model, mode, msgs, time.time() - t0, reply=text)
                return LLMReply(text=text, seconds=time.time() - t0, model=self.model)
            except APIStatusError as e:
                last = e
                _log_call(self.model, mode, msgs, time.time() - t0, error=e)
                if e.status_code in (401, 403, 404):
                    raise LLMError(f"{e.status_code}: {e.message}") from e
                # Public AI holds a 429 for ~26 s before answering, so one is enough to give up on the model.
                if e.status_code == 429 and self.fallback and self.model != self.fallback:
                    log.warning("%s is throttled (HTTP 429); switching to %s", self.model, self.fallback)
                    self.model = self.fallback
                    continue
                if e.status_code == 400 and mode == 0 and "system message" in str(e).lower():
                    self.mode = mode = 1
                    log.warning("%s rejects system messages; folding the system prompt into the user turn", self.model)
                    continue
                if e.status_code == 400 and mode == 1 and any(m["role"] == "assistant" for m in messages):
                    self.mode = mode = 2
                    log.warning("%s rejects multi-turn requests; sending the conversation as one user turn", self.model)
                    continue
            except Exception as e:  # noqa: BLE001 - network errors, timeouts, empty replies
                last = e
                _log_call(self.model, mode, msgs, time.time() - t0, error=e)
            log.warning("%s attempt %d/%d failed after %.1f s: %s", self.model, attempt + 1, attempts + 3,
                        time.time() - t0, str(last)[:160])
            time.sleep(min(1.5 * (attempt + 1), 6))
        raise LLMError(f"Apertus did not answer: {last}")

    def chat_json(self, messages: list[dict], **kw) -> tuple[dict | None, LLMReply]:
        reply = self.chat(messages, **kw)
        data = extract_json(reply.text)
        if data is None:  # one repair round
            repair = messages + [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": "Das war kein gültiges JSON. Gib genau dieselbe Antwort "
                                            "als EIN gültiges JSON-Objekt zurück, ohne weiteren Text."},
            ]
            reply2 = self.chat(repair, **kw)
            data = extract_json(reply2.text)
            reply = LLMReply(text=reply2.text, seconds=reply.seconds + reply2.seconds, model=reply.model)
        return data, reply


def fold_system(messages: list[dict]) -> list[dict]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return rest
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"[Anweisungen]\n{system}\n\n[Nachricht]\n{m['content']}"
            break
    else:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def flatten_conversation(messages: list[dict]) -> list[dict]:
    """Instructions, the conversation so far and the new message, all in a single user turn."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    *history, new = [m for m in messages if m["role"] != "system"]
    parts = [f"[Anweisungen]\n{system}"] if system else []
    if history:
        parts.append("[Bisheriges Gespräch]\n" + "\n".join(
            f"{'Du' if m['role'] == 'assistant' else 'Lernende Person'}: {m['content']}" for m in history))
    parts.append(f"[Nachricht]\n{new['content']}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        value = json.loads(candidate)
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        continue
                return None
    return None
