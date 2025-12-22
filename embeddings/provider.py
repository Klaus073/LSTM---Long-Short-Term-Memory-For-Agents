"""
Embedding providers for Memory SDK.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from memory_sdk.config import MemoryConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.config.embedding.openai_api_key)
        return self._client
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        response = await self.client.embeddings.create(
            model=self.config.embedding.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        
        # Process in batches
        all_embeddings = []
        batch_size = self.config.embedding.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                model=self.config.embedding.model,
                input=batch,
            )
            embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding provider."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.config.embedding.ollama_base_url,
                timeout=60.0,
            )
        return self._client
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        response = await self.client.post(
            "/api/embeddings",
            json={
                "model": self.config.embedding.model,
                "prompt": text,
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        # Ollama doesn't support batch embedding, so we parallelize
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)


def get_embedding_provider(config: MemoryConfig) -> EmbeddingProvider:
    """Get embedding provider based on configuration."""
    provider = config.embedding.provider.lower()
    
    if provider == "openai":
        return OpenAIEmbedding(config)
    elif provider == "ollama":
        return OllamaEmbedding(config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

