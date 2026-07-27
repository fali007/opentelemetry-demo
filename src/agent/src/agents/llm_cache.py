#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""File-backed cache for LLM chat-completion requests.

Each model's interactions live in a single append-only JSONL file, one
`{"digest", "request", "response"}` document per line:

    <LLM_CACHE_DIR>/<model>.jsonl

All entries are kept in an in-memory index keyed by the SHA-256 of the
canonicalized request. Lookups try an exact digest match first, then
fall back to fuzzy matching so that near-identical requests (e.g.
MCP-wrapped tool results) can reuse a stored response. On an exact miss
the file tail is re-read to pick up entries appended by other replicas
sharing the fixtures volume. Writes append one line while holding an
exclusive `flock`, so concurrent writers -- asyncio tasks, sync
callers, or several agent replicas -- never interleave; a line left
truncated by a crashed writer is healed by the next append and skipped
by loaders.

Modes (`LLM_CACHE_MODE`): `hybrid` serves cached responses and records
misses from the live LLM; `replay` never calls the live LLM and raises
`LLMCacheMiss` on a miss; `record` always calls the live LLM and stores
the result; `off` disables the cache entirely.
"""

import asyncio
import copy
import fcntl
import hashlib
import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Volatile identifiers are masked before hashing/matching so requests
# that differ only in per-session identifiers share a cache entry.
# Matches RFC 4122 style UUIDs (8-4-4-4-12 hex digits), e.g. the demo's
# user and session ids:
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# Matches provider tool-call ids: `call_` (OpenAI) or `toolu_`
# (Anthropic) followed by the id token:
_TOOL_CALL_ID_RE = re.compile(r"\b(?:call|toolu)_[A-Za-z0-9_-]+")


class LLMCacheMiss(Exception):
    """Raised in replay mode when no cached response matches."""


def _scrub_volatile(text):
    text = _UUID_RE.sub("<uuid>", text)
    return _TOOL_CALL_ID_RE.sub("<tool-call-id>", text)


def _unwrap_tool_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return "".join(parts)
    return json.dumps(content, sort_keys=True, default=str)


def _normalize_message(message):
    if isinstance(message, dict) and message.get("role") == "tool":
        normalized = dict(message)
        normalized["content"] = _unwrap_tool_content(message.get("content"))
        return normalized
    return message


def _message_signature(message):
    if not isinstance(message, dict):
        return json.dumps(message, sort_keys=True, default=str)
    parts = [str(message.get("role", ""))]
    content = message.get("content")
    if content:
        parts.append(_unwrap_tool_content(content))
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            parts.append(str(function.get("name", "")))
            parts.append(str(function.get("arguments", "")))
    return "|".join(parts)


def _canonicalize(payload):
    canonical = json.loads(json.dumps(payload, sort_keys=True, default=str))
    canonical.pop("stream", None)
    canonical.pop("stream_options", None)
    messages = canonical.get("messages")
    if isinstance(messages, list):
        canonical["messages"] = [_normalize_message(m) for m in messages]
    return canonical


def _clean_response(response):
    if not isinstance(response, dict) or "choices" not in response:
        return None
    cleaned = json.loads(json.dumps(response, default=str))
    if "id" in cleaned:
        cleaned["id"] = "chatcmpl-fixture"
    if "created" in cleaned:
        cleaned["created"] = 0
    for key in (
        "system_fingerprint",
        "usage",
        "prompt_filter_results",
        "service_tier",
    ):
        cleaned.pop(key, None)
    for choice in cleaned.get("choices", []):
        if not isinstance(choice, dict):
            continue
        choice.pop("provider_specific_fields", None)
        message = choice.get("message")
        if isinstance(message, dict):
            message.pop("provider_specific_fields", None)
            message.pop("annotations", None)
    return cleaned


class CacheKey:
    def __init__(self, payload):
        self.request = _canonicalize(payload)
        canonical = _scrub_volatile(json.dumps(self.request, sort_keys=True))
        self.digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        messages = self.request.get("messages")
        if isinstance(messages, list) and messages:
            self.n_messages = len(messages)
            self.messages_repr = _scrub_volatile(
                json.dumps(messages, sort_keys=True)
            )
            self.last_repr = _scrub_volatile(
                _message_signature(messages[-1])
            )
        else:
            self.n_messages = 0
            self.messages_repr = None
            self.last_repr = None


class _Entry:
    __slots__ = (
        "digest",
        "n_messages",
        "messages_repr",
        "last_repr",
        "response",
    )

    def __init__(self, digest, key, response):
        self.digest = digest
        self.n_messages = key.n_messages
        self.messages_repr = key.messages_repr
        self.last_repr = key.last_repr
        self.response = response


class LLMCache:
    def __init__(self, path, mode, threshold, max_entries):
        self.mode = mode
        self._path = path
        self._threshold = threshold
        self._max_entries = max_entries
        self._entries = {}
        self._offset = 0
        self._lock = threading.Lock()
        self._inflight = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._refresh()
        logger.info(
            "LLM cache: mode=%s path=%s entries=%d threshold=%.2f",
            mode,
            path,
            len(self._entries),
            threshold,
        )

    def lookup(self, key):
        """Return a stored response for the request, or None."""
        entry = self._entries.get(key.digest)
        if entry is None:
            self._refresh()
            entry = self._entries.get(key.digest)
        if entry is not None:
            logger.info("LLM cache exact hit (%s)", key.digest[:12])
            return copy.deepcopy(entry.response)
        entry, score = self._closest(key)
        if entry is not None:
            logger.info(
                "LLM cache fuzzy hit (score=%.4f, %s)",
                score,
                entry.digest[:12],
            )
            return copy.deepcopy(entry.response)
        logger.info("LLM cache miss (%s)", key.digest[:12])
        return None

    def store(self, key, response):
        cleaned = _clean_response(response)
        if cleaned is None:
            logger.warning(
                "LLM cache: not storing response without choices (%s)",
                key.digest[:12],
            )
            return
        if (
            key.digest not in self._entries
            and len(self._entries) >= self._max_entries
        ):
            logger.warning(
                "LLM cache full (%d entries), not storing %s",
                self._max_entries,
                key.digest[:12],
            )
            return
        document = {
            "digest": key.digest,
            "request": key.request,
            "response": cleaned,
        }
        line = json.dumps(document, sort_keys=True) + "\n"
        try:
            self._append(line.encode("utf-8"))
        except OSError:
            logger.exception("LLM cache: could not store %s", key.digest[:12])
            return
        with self._lock:
            self._entries[key.digest] = _Entry(key.digest, key, cleaned)
        logger.info("LLM cache stored (%s)", key.digest[:12])

    async def fetch(self, key, factory):
        future = self._inflight.get(key.digest)
        if future is not None:
            return copy.deepcopy(await future)
        future = asyncio.get_running_loop().create_future()
        self._inflight[key.digest] = future
        try:
            response = await factory()
        except BaseException as exc:
            self._inflight.pop(key.digest, None)
            future.set_exception(exc)
            # Mark retrieved so the loop does not warn when no other
            # caller was awaiting this future.
            future.exception()
            raise
        self._inflight.pop(key.digest, None)
        future.set_result(response)
        self.store(key, response)
        return response

    def _append(self, line):
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(fd, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            end = f.seek(0, os.SEEK_END)
            # Start on a fresh line even if a crashed writer left the
            # previous one truncated.
            if end > 0:
                f.seek(end - 1)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(line)
            f.flush()

    def _refresh(self):
        with self._lock:
            try:
                size = os.path.getsize(self._path)
            except OSError:
                return
            if size < self._offset:
                # The file shrank (e.g. hand-pruned during development);
                # re-read it from the start.
                self._offset = 0
            if size == self._offset:
                return
            try:
                with open(self._path, "rb") as f:
                    f.seek(self._offset)
                    chunk = f.read()
            except OSError as exc:
                logger.warning(
                    "LLM cache: cannot read %s: %s", self._path, exc
                )
                return
            # Consume only complete lines; a trailing fragment still
            # being written is re-read once its newline lands.
            complete = chunk.rfind(b"\n") + 1
            for line in chunk[:complete].splitlines():
                self._add_line(line)
            self._offset += complete

    def _add_line(self, line):
        line = line.strip()
        if not line:
            return
        try:
            data = json.loads(line)
            key = CacheKey(data["request"])
            entry = _Entry(key.digest, key, data["response"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "LLM cache: skipping unreadable line in %s: %s",
                self._path,
                exc,
            )
            return
        self._entries[key.digest] = entry

    def _closest(self, key):
        if key.messages_repr is None:
            return None, 0.0
        best = None
        best_score = -1.0
        for entry in list(self._entries.values()):
            if (
                entry.messages_repr is None
                or entry.n_messages != key.n_messages
            ):
                continue
            # The newest message carries the most signal between
            # otherwise-identical requests, so it is weighted heavier
            # than the conversation as a whole.
            score = 0.4 * SequenceMatcher(
                None, entry.messages_repr, key.messages_repr
            ).ratio() + 0.6 * SequenceMatcher(
                None, entry.last_repr, key.last_repr
            ).ratio()
            if score > best_score:
                best = entry
                best_score = score
        if best is not None and best_score >= self._threshold:
            return best, best_score
        return None, 0.0


_MODES = ("hybrid", "replay", "record", "off")

_caches = {}
_caches_lock = threading.Lock()


def _resolve_mode():
    mode = os.getenv("LLM_CACHE_MODE", "hybrid").strip().lower()
    if mode not in _MODES:
        raise ValueError(
            f"invalid LLM_CACHE_MODE {mode!r}; expected one of {_MODES}"
        )
    return mode


def get_cache(model_name):
    """Return the process-wide cache for a model, or None when off."""
    if _resolve_mode() == "off":
        return None
    with _caches_lock:
        cache = _caches.get(model_name)
        if cache is None:
            cache = LLMCache(
                path=os.path.join(
                    os.getenv("LLM_CACHE_DIR", "fixtures/llm_cache"),
                    model_name.replace("/", "_") + ".jsonl",
                ),
                mode=_resolve_mode(),
                threshold=float(
                    os.getenv("LLM_CACHE_MATCH_THRESHOLD", "0.85")
                ),
                max_entries=int(os.getenv("LLM_CACHE_MAX_ENTRIES", "1000")),
            )
            _caches[model_name] = cache
    return cache
