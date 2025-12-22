"""
Async Milvus backend for Long-Term Memory.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from memory_sdk.config import MemoryConfig
from memory_sdk.models import Memory, SearchResult

logger = logging.getLogger(__name__)


class LTMBackend:
    """Async Milvus backend for long-term memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._client: Optional[MilvusClient] = None
        self._collection: Optional[Collection] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def connect(self) -> None:
        """Connect to Milvus."""
        # Milvus client is sync, so we run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._connect_sync)
        logger.info(f"Connected to Milvus at {self.config.milvus.host}:{self.config.milvus.port}")
    
    def _connect_sync(self) -> None:
        """Synchronous connection to Milvus."""
        connections.connect(
            alias="default",
            host=self.config.milvus.host,
            port=self.config.milvus.port,
        )
        
        # Create collection if not exists
        if not utility.has_collection(self.config.milvus.collection_name):
            self._create_collection()
        
        self._collection = Collection(self.config.milvus.collection_name)
        self._collection.load()
    
    def _create_collection(self) -> None:
        """Create the memories collection."""
        fields = [
            FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="strategy", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding.dimensions),
        ]
        
        schema = CollectionSchema(fields, description="User memories")
        collection = Collection(self.config.milvus.collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": self.config.milvus.metric_type,
            "index_type": self.config.milvus.index_type,
            "params": {"nlist": self.config.milvus.nlist},
        }
        collection.create_index("embedding", index_params)
        
        logger.info(f"Created collection {self.config.milvus.collection_name}")
    
    async def close(self) -> None:
        """Close Milvus connection."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._close_sync)
        self._executor.shutdown(wait=False)
    
    def _close_sync(self) -> None:
        """Synchronous close."""
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
    
    @property
    def collection(self) -> Collection:
        if not self._collection:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._collection
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    async def upsert_memory(self, memory: Memory) -> None:
        """Insert or update a memory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            partial(self._upsert_memory_sync, memory)
        )
    
    def _upsert_memory_sync(self, memory: Memory) -> None:
        """Synchronous upsert."""
        import json
        
        # Delete existing if any
        self.collection.delete(f'memory_id == "{memory.memory_id}"')
        
        # Insert new
        data = [
            [memory.memory_id],
            [memory.user_id],
            [memory.strategy],
            [json.dumps(memory.content)],
            [memory.summary or ""],
            [memory.confidence],
            [memory.created_at.isoformat()],
            [memory.updated_at.isoformat()],
            [memory.embedding or [0.0] * self.config.embedding.dimensions],
        ]
        
        self.collection.insert(data)
        self.collection.flush()
    
    async def upsert_memories(self, memories: List[Memory]) -> None:
        """Batch insert or update memories."""
        if not memories:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            partial(self._upsert_memories_sync, memories)
        )
    
    def _upsert_memories_sync(self, memories: List[Memory]) -> None:
        """Synchronous batch upsert."""
        import json
        
        # Delete existing
        ids = [f'"{m.memory_id}"' for m in memories]
        if ids:
            expr = f"memory_id in [{','.join(ids)}]"
            self.collection.delete(expr)
        
        # Batch insert
        data = [
            [m.memory_id for m in memories],
            [m.user_id for m in memories],
            [m.strategy for m in memories],
            [json.dumps(m.content) for m in memories],
            [m.summary or "" for m in memories],
            [m.confidence for m in memories],
            [m.created_at.isoformat() for m in memories],
            [m.updated_at.isoformat() for m in memories],
            [m.embedding or [0.0] * self.config.embedding.dimensions for m in memories],
        ]
        
        self.collection.insert(data)
        self.collection.flush()
        
        logger.debug(f"Upserted {len(memories)} memories")
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._get_memory_sync, memory_id)
        )
    
    def _get_memory_sync(self, memory_id: str) -> Optional[Memory]:
        """Synchronous get."""
        import json
        
        results = self.collection.query(
            expr=f'memory_id == "{memory_id}"',
            output_fields=["memory_id", "user_id", "strategy", "content", "summary", "confidence", "created_at", "updated_at"],
        )
        
        if not results:
            return None
        
        r = results[0]
        return Memory(
            memory_id=r["memory_id"],
            user_id=r["user_id"],
            strategy=r["strategy"],
            content=json.loads(r["content"]),
            summary=r.get("summary", ""),
            confidence=r["confidence"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )
    
    async def get_user_memories(
        self,
        user_id: str,
        strategies: Optional[List[str]] = None
    ) -> List[Memory]:
        """Get all memories for a user."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._get_user_memories_sync, user_id, strategies)
        )
    
    def _get_user_memories_sync(
        self,
        user_id: str,
        strategies: Optional[List[str]] = None
    ) -> List[Memory]:
        """Synchronous get user memories."""
        import json
        
        expr = f'user_id == "{user_id}"'
        if strategies:
            strats = ",".join(f'"{s}"' for s in strategies)
            expr += f" and strategy in [{strats}]"
        
        results = self.collection.query(
            expr=expr,
            output_fields=["memory_id", "user_id", "strategy", "content", "summary", "confidence", "created_at", "updated_at"],
        )
        
        memories = []
        for r in results:
            memories.append(Memory(
                memory_id=r["memory_id"],
                user_id=r["user_id"],
                strategy=r["strategy"],
                content=json.loads(r["content"]),
                summary=r.get("summary", ""),
                confidence=r["confidence"],
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            ))
        
        return memories
    
    async def search_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        strategies: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """Semantic search for memories."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._search_memories_sync, user_id, query_embedding, top_k, strategies, min_score)
        )
    
    def _search_memories_sync(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int,
        strategies: Optional[List[str]],
        min_score: Optional[float],
    ) -> List[SearchResult]:
        """Synchronous search."""
        import json
        
        expr = f'user_id == "{user_id}"'
        if strategies:
            strats = ",".join(f'"{s}"' for s in strategies)
            expr += f" and strategy in [{strats}]"
        
        min_score = min_score or self.config.milvus.min_similarity
        
        search_params = {
            "metric_type": self.config.milvus.metric_type,
            "params": {"nprobe": self.config.milvus.nprobe},
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["memory_id", "user_id", "strategy", "content", "summary", "confidence", "created_at", "updated_at"],
        )
        
        search_results = []
        for hits in results:
            for hit in hits:
                if hit.score < min_score:
                    continue
                
                entity = hit.entity
                memory = Memory(
                    memory_id=entity.get("memory_id"),
                    user_id=entity.get("user_id"),
                    strategy=entity.get("strategy"),
                    content=json.loads(entity.get("content", "{}")),
                    summary=entity.get("summary", ""),
                    confidence=entity.get("confidence", 1.0),
                    created_at=datetime.fromisoformat(entity.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(entity.get("updated_at", datetime.utcnow().isoformat())),
                )
                search_results.append(SearchResult(memory=memory, score=hit.score))
        
        return search_results
    
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            partial(self._delete_memory_sync, memory_id)
        )
    
    def _delete_memory_sync(self, memory_id: str) -> None:
        """Synchronous delete."""
        self.collection.delete(f'memory_id == "{memory_id}"')
    
    async def delete_user_memories(self, user_id: str) -> None:
        """Delete all memories for a user."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            partial(self._delete_user_memories_sync, user_id)
        )
    
    def _delete_user_memories_sync(self, user_id: str) -> None:
        """Synchronous delete user memories."""
        self.collection.delete(f'user_id == "{user_id}"')
        self.collection.flush()
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> bool:
        """Check if Milvus is healthy."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: utility.get_server_version() is not None
            )
        except Exception:
            return False

