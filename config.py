"""
Configuration for Memory SDK.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 50
    socket_timeout: float = 5.0
    
    # Key prefixes
    session_prefix: str = "mem:session:"
    message_prefix: str = "mem:msg:"
    ltm_cache_prefix: str = "mem:ltm:"
    checkpoint_prefix: str = "mem:ckpt:"
    
    # TTL settings
    session_ttl: int = 3600  # 1 hour
    message_ttl: int = 86400  # 24 hours
    ltm_cache_ttl: int = 604800  # 7 days


@dataclass
class MilvusConfig:
    """Milvus configuration."""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "memories"
    
    # Index settings
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 16
    
    # Search settings
    default_top_k: int = 10
    min_similarity: float = 0.5


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str = "openai"  # openai, ollama
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    
    # Provider-specific
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class LLMConfig:
    """LLM configuration for extraction."""
    provider: str = "openai"  # openai, anthropic, ollama
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4096
    
    # Provider-specific
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class ExtractionConfig:
    """Memory extraction configuration."""
    enabled: bool = True
    auto_extract: bool = False  # Extract on session close
    batch_size: int = 50  # Messages per extraction
    
    # Default strategies
    strategies: List[str] = field(default_factory=lambda: ["profile", "preferences", "facts"])


@dataclass
class MemoryConfig:
    """Main configuration for Memory SDK."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    
    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Redis
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        config.redis.password = os.getenv("REDIS_PASSWORD")
        
        # Milvus
        config.milvus.host = os.getenv("MILVUS_HOST", config.milvus.host)
        config.milvus.port = int(os.getenv("MILVUS_PORT", config.milvus.port))
        
        # Embedding
        config.embedding.provider = os.getenv("EMBEDDING_PROVIDER", config.embedding.provider)
        config.embedding.model = os.getenv("EMBEDDING_MODEL", config.embedding.model)
        config.embedding.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # LLM
        config.llm.provider = os.getenv("LLM_PROVIDER", config.llm.provider)
        config.llm.model = os.getenv("LLM_MODEL", config.llm.model)
        config.llm.openai_api_key = os.getenv("OPENAI_API_KEY")
        config.llm.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        return config
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Load configuration from dictionary."""
        config = cls()
        
        if "redis" in data:
            for k, v in data["redis"].items():
                if hasattr(config.redis, k):
                    setattr(config.redis, k, v)
        
        if "milvus" in data:
            for k, v in data["milvus"].items():
                if hasattr(config.milvus, k):
                    setattr(config.milvus, k, v)
        
        if "embedding" in data:
            for k, v in data["embedding"].items():
                if hasattr(config.embedding, k):
                    setattr(config.embedding, k, v)
        
        if "llm" in data:
            for k, v in data["llm"].items():
                if hasattr(config.llm, k):
                    setattr(config.llm, k, v)
        
        if "extraction" in data:
            for k, v in data["extraction"].items():
                if hasattr(config.extraction, k):
                    setattr(config.extraction, k, v)
        
        return config

