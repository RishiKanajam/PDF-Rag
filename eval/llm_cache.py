"""
llm_cache.py
------------
Wraps a LangChain chat model so every response is cached in SQLite, keyed by
(model, generation params, prompt). Reruns are free and return byte-identical
answers, and a run interrupted by a rate limit resumes where it stopped.

The wrapper exposes the same .invoke(prompt) -> obj.content interface the
pipeline uses, so it can be passed straight into RAGPipeline(llm=...).
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class CachedResponse:
    content: str
    usage_metadata: Dict[str, int] | None
    response_metadata: Dict[str, Any]
    latency_s: float          # latency of the original (uncached) call
    cached: bool
    key: str = field(default="")


class CachedChatModel:
    def __init__(self, model, model_name: str, params: Dict[str, Any], cache_path: Path):
        self.model = model
        self.model_name = model_name
        self.params = params
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.hits = 0
        self.misses = 0
        self.last: CachedResponse | None = None

    def _key(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model_name, "params": self.params, "prompt": prompt},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def invoke(self, prompt: str) -> CachedResponse:
        key = self._key(prompt)
        row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            self.last = CachedResponse(**json.loads(row[0]), cached=True, key=key)
            return self.last

        start = time.perf_counter()
        response = self.model.invoke(prompt)
        latency = time.perf_counter() - start

        meta = response.response_metadata or {}
        record = {
            "content": response.content,
            "usage_metadata": dict(response.usage_metadata) if response.usage_metadata else None,
            "response_metadata": {
                "model_name": meta.get("model_name"),
                "system_fingerprint": meta.get("system_fingerprint"),
                "finish_reason": meta.get("finish_reason"),
            },
            "latency_s": latency,
        }
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, json.dumps(record)),
        )
        self._db.commit()
        self.misses += 1
        self.last = CachedResponse(**record, cached=False, key=key)
        return self.last
