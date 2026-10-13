"""Embedder abstractions and implementations for semantic vector generation."""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

import litellm


class BaseEmbedder(ABC):
    """Abstract base interface for text embedders."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector embedding dimensionality."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a single text string."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of text strings."""


class MockEmbedder(BaseEmbedder):
    """Deterministic, offline pseudo-embedder for fast tests and zero-network execution."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized vector from text content."""
        raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector: list[float] = []

        # Generate dimension values by cycling and hashing
        for i in range(self._dimension):
            byte_val = raw_hash[i % len(raw_hash)]
            # Spread pseudo-randomly based on position and character sum
            val = math.sin((byte_val + i + sum(text.encode("utf-8"))) * 0.1)
            vector.append(val)

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            return [v / norm for v in vector]
        return vector

    async def embed_text(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]


class LiteLLMEmbedder(BaseEmbedder):
    """Embedder using LiteLLM to support providers (OpenAI, Gemini, Ollama, etc.)."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self._dimension = dimension
        self.extra_kwargs = kwargs

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await litellm.aembedding(
            model=self.model_name,
            input=texts,
            **self.extra_kwargs,
        )
        embeddings: list[list[float]] = []
        for item in response.data:
            embeddings.append(item["embedding"])
        return embeddings
