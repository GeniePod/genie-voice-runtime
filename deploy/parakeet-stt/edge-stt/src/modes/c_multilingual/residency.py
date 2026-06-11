#!/usr/bin/env python3
"""
Mode C — model registry + Whisper-tiny LID router + TTL/LRU one-resident manager.

The multilingual mode must keep **exactly ONE heavy recognizer resident** on the 8 GB box (two
0.6 B models would not co-fit — see results/feasibility.md GATE A). This module enforces that:

  Registry      logical name -> model dir + recognizer family (whisper | parakeet)
  LidRouter     Whisper-tiny detects the language (the JSON "lang" field) -> picks a recognizer
  Residency     holds at most ONE heavy model "resident"; on a route to a different heavy model it
                EVICTS-BEFORE-LOAD (serialized, never two resident at once); a TTL expires an idle
                model and LRU breaks ties.

Architecture note: the shipped runners drive sherpa-onnx via a fresh subprocess per chunk, so the
"resident" handle here is the *active model selection* (a counted, mutex-guarded slot) rather than a
warm in-process session — the production M2 path swaps this loader for a resident `serve` process
(as the parakeet.cpp STT server already does on :8178). The invariant it guarantees — never run two
heavy recognizers concurrently — is what GATE A needs, and is enforced regardless.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


# whisper-tiny is LIGHT (~104 MB) and always available as the LID + fallback recognizer; only the
# 0.6 B Parakeet recognizers are "heavy" and subject to the one-resident cap.
HEAVY = {"parakeet"}


@dataclass
class ModelSpec:
    name: str
    family: str          # "whisper" | "parakeet"
    model_dir: str
    heavy: bool = field(init=False)

    def __post_init__(self):
        self.heavy = self.family in HEAVY


class Registry:
    """logical name -> ModelSpec."""
    def __init__(self):
        self._m: dict[str, ModelSpec] = {}

    def add(self, spec: ModelSpec) -> "Registry":
        self._m[spec.name] = spec
        return self

    def get(self, name: str) -> ModelSpec:
        return self._m[name]

    def __contains__(self, name: str) -> bool:
        return name in self._m


class LidRouter:
    """Map a Whisper-detected language code to a registered recognizer name.

    English routes to the accurate English model (Parakeet-v2); the v3 25-lang model serves the
    other supported languages; anything out-of-set falls back to whisper-tiny itself (which is
    multilingual and already loaded as the LID)."""
    def __init__(self, *, english: str, multilingual: str, fallback: str,
                 multilingual_langs: set[str] | None = None):
        self.english = english
        self.multilingual = multilingual
        self.fallback = fallback
        # Parakeet-v3 supported languages (subset; extend from the model card).
        self.multilingual_langs = multilingual_langs or {
            "de", "es", "fr", "it", "pt", "nl", "pl", "ru", "uk", "cs", "sk", "hr", "bg",
            "da", "fi", "sv", "el", "hu", "lt", "lv", "et", "ro", "sl", "mt", "en",
        }

    def route(self, lang: str) -> str:
        lang = (lang or "").lower()
        if lang in ("en", "english"):
            return self.english
        if lang in self.multilingual_langs:
            return self.multilingual
        return self.fallback


class Residency:
    """At most ONE heavy model resident. evict-before-load, serialized; TTL + LRU."""
    def __init__(self, registry: Registry, *, ttl_sec: float = 300.0, clock=time.monotonic):
        self.reg = registry
        self.ttl = ttl_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._resident: str | None = None      # name of the currently-resident HEAVY model
        self._last_used: float = 0.0
        self.events: list[tuple[str, str]] = []  # (action, name) audit trail

    def _evict(self):
        if self._resident is not None:
            self.events.append(("evict", self._resident))
            self._resident = None

    def acquire(self, name: str):
        """Ensure `name` is the resident heavy model (light models never evict anything). Returns
        the ModelSpec. Guarantees the one-resident invariant under the lock."""
        spec = self.reg.get(name)
        with self._lock:
            now = self._clock()
            # TTL: drop an idle resident before doing anything else.
            if self._resident is not None and (now - self._last_used) > self.ttl:
                self._evict()
            if spec.heavy:
                if self._resident != name:
                    self._evict()                  # evict-before-load: never two heavy resident
                    self.events.append(("load", name))
                    self._resident = name
                self._last_used = now
            return spec

    def resident(self) -> str | None:
        return self._resident

    def assert_single_resident(self) -> bool:
        """The audit invariant GATE A needs: scanning the event log, the number of loads minus
        evicts is never > 1 at any point (never two heavy models resident simultaneously)."""
        live = 0
        for action, _ in self.events:
            live += 1 if action == "load" else -1
            if live > 1:
                return False
        return True
