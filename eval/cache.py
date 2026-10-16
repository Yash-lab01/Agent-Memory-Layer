"""Deterministic disk cache for LLM completions."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class DiskCache:
    """Thread-safe file-backed disk cache for model completions."""

    def __init__(self, cache_dir: Path | str = "eval/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, model: str, prompt: str, **kwargs: Any) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, **kwargs: Any) -> str | None:
        """Retrieve cached text completion if present, else None."""
        key = self._compute_key(model, prompt, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("response")
            return str(val) if isinstance(val, str) else None
        except Exception:
            return None

    def set(self, model: str, prompt: str, response: str, **kwargs: Any) -> None:
        """Write text completion to cache."""
        key = self._compute_key(model, prompt, **kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        payload = {
            "key": key,
            "model": model,
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        tmp_file = cache_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_file.replace(cache_file)

    def has(self, model: str, prompt: str, **kwargs: Any) -> bool:
        """Check whether a cached completion exists."""
        key = self._compute_key(model, prompt, **kwargs)
        return (self.cache_dir / f"{key}.json").exists()

    def clear(self) -> int:
        """Clear all cached entries and return count of deleted files."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        return count
