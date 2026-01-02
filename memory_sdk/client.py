"""
Main Memory SDK Client.

This is the primary interface for interacting with the memory system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from memory_sdk.config import MemoryConfig
from memory_sdk.embeddings.provider import EmbeddingProvider, get_embedding_provider
from memory_sdk.extraction.extractor import MemoryExtractor
from memory_sdk.ltm.backend import LTMBackend
from memory_sdk.models import (
    ExtractionResult,
    LTMStatus,
    Memory,
    Message,
    Role,
    SearchResult,
    Session,
    SessionStatus,
    UserContext,
)
from memory_sdk.stm.backend import STMBackend

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Main client for Memory SDK.
    
    Provides a simple, unified interface for:
    - Short-term memory (conversation history)
    - Long-term memory (extracted insights)
    - Memory extraction
    - Semantic search
    
    Usage:
        # Async initialization (recommended)
        client = await MemoryClient.create()
        
        # Add messages
        await client.add_message(
            user_id="alice",
            session_id="session-1",
            role="user",
            content="Hi, I'm Alice!"
        )
        
        # Get context for LLM
        context = await client.get_context("alice")
        
        # Extract memories after session
        result = await client.extract_memories("alice", "session-1")
        
        # Search memories
        results = await client.search_memories("alice", "What does Alice like?")
        
        # Close
        await client.close()
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the client.
        
        Args:
            config: Configuration. If None, loads from environment.
        """
        self.config = config or MemoryConfig.from_env()
        
        # Initialize backends
        self.stm = STMBackend(self.config)
        self.ltm = LTMBackend(self.config)
        
        # Initialize embedding provider
        self.embedding: Optional[EmbeddingProvider] = None
        
        # Initialize extractor
        self.extractor: Optional[MemoryExtractor] = None
        
        self._connected = False
    
    @classmethod
    async def create(cls, config: Optional[MemoryConfig] = None) -> "MemoryClient":
        """
        Create and connect a new client.
        
        This is the recommended way to create a client.
        
        Args:
            config: Configuration. If None, loads from environment.
            
        Returns:
            Connected MemoryClient instance.
        """
        client = cls(config)
        await client.connect()
        return client
    
    async def connect(self) -> None:
        """Connect to all backends."""
        if self._connected:
            return
        
        # Connect to Redis and Milvus
        await asyncio.gather(
            self.stm.connect(),
            self.ltm.connect(),
        )
        
        # Initialize embedding provider
        self.embedding = get_embedding_provider(self.config)
        
        # Initialize extractor
        self.extractor = MemoryExtractor(
            config=self.config,
            embedding_provider=self.embedding,
        )
        
        self._connected = True
        logger.info("MemoryClient connected")
    
    async def close(self) -> None:
        """Close all connections."""
        await asyncio.gather(
            self.stm.close(),
            self.ltm.close(),
        )
        self._connected = False
        logger.info("MemoryClient closed")
    
    async def __aenter__(self) -> "MemoryClient":
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    async def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID (generated if not provided)
            metadata: Optional metadata
            
        Returns:
            Created Session object
        """
        session_id = session_id or str(uuid4())
        return await self.stm.create_session(user_id, session_id, metadata)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return await self.stm.get_session(session_id)
    
    async def get_active_session(self, user_id: str) -> Optional[Session]:
        """Get active session for a user."""
        return await self.stm.get_active_session(user_id)
    
    async def close_session(self, session_id: str) -> None:
        """Close a session."""
        await self.stm.close_session(session_id)
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to a session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message object
        """
        # Ensure session exists
        session = await self.stm.get_session(session_id)
        if not session:
            await self.stm.create_session(user_id, session_id)
        
        message = Message(
            role=Role(role),
            content=content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        
        await self.stm.add_message(message)
        return message
    
    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Get messages from a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of Message objects
        """
        return await self.stm.get_messages(session_id, limit)
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    async def extract_memories(
        self,
        user_id: str,
        session_id: str,
    ) -> ExtractionResult:
        """
        Extract memories from a session.
        
        This should be called after a session ends to extract
        structured memories into long-term storage.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            ExtractionResult with stats
        """
        if not self.extractor:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Set status to processing
        await self.stm.set_ltm_status(user_id, LTMStatus.PROCESSING)
        
        try:
            # Get messages
            messages = await self.stm.get_messages(session_id)
            if not messages:
                return ExtractionResult(
                    user_id=user_id,
                    session_id=session_id,
                    memories_extracted=0,
                    memories_updated=0,
                    strategies_processed=[],
                    duration_ms=0,
                )
            
            # Get existing memories
            existing = await self.ltm.get_user_memories(user_id)
            existing_map = {m.strategy: m for m in existing}
            
            # Extract
            result, memories = await self.extractor.extract_memories(
                user_id=user_id,
                session_id=session_id,
                messages=messages,
                existing_memories=existing_map,
            )
            
            # Save to LTM
            if memories:
                await self.ltm.upsert_memories(memories)
                
                # Cache in Redis
                await self.stm.set_ltm_cache(user_id, memories)
            
            # Set status to ready
            await self.stm.set_ltm_status(user_id, LTMStatus.READY)
            
            logger.info(f"Extracted {len(memories)} memories for user {user_id}")
            return result
        
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            await self.stm.set_ltm_status(user_id, LTMStatus.EMPTY)
            raise
    
    async def get_memories(
        self,
        user_id: str,
        strategies: Optional[List[str]] = None,
    ) -> List[Memory]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User identifier
            strategies: Optional list of strategies to filter by
            
        Returns:
            List of Memory objects
        """
        return await self.ltm.get_user_memories(user_id, strategies)
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        strategies: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search for memories.
        
        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of results to return
            strategies: Optional list of strategies to filter by
            
        Returns:
            List of SearchResult objects
        """
        if not self.embedding:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Get query embedding
        embedding = await self.embedding.embed_text(query)
        
        # Search
        return await self.ltm.search_memories(
            user_id=user_id,
            query_embedding=embedding,
            top_k=top_k,
            strategies=strategies,
        )
    
    async def delete_memories(self, user_id: str) -> None:
        """
        Delete all memories for a user.
        
        Args:
            user_id: User identifier
        """
        await asyncio.gather(
            self.ltm.delete_user_memories(user_id),
            self.stm.delete_ltm_cache(user_id),
        )
    
    # =========================================================================
    # Context Operations
    # =========================================================================
    
    async def get_context(
        self,
        user_id: str,
        include_recent_messages: bool = True,
        session_id: Optional[str] = None,
        message_limit: int = 10,
    ) -> UserContext:
        """
        Get full context for a user.
        
        This is the main method to call before invoking an LLM.
        It returns both long-term memories and recent messages.
        
        Args:
            user_id: User identifier
            include_recent_messages: Include recent messages from current session
            session_id: Session to get messages from (uses active session if not provided)
            message_limit: Maximum recent messages to include
            
        Returns:
            UserContext with memories and messages
        """
        # Get LTM status and cache
        ltm_status = await self.stm.get_ltm_status(user_id)
        ltm_cache = await self.stm.get_ltm_cache(user_id)
        
        # Build memories from cache
        memories = []
        context_string = ""
        
        if ltm_cache:
            context_string = ltm_cache.get("context_string", "")
            for m in ltm_cache.get("memories", []):
                memories.append(Memory(
                    memory_id=m["memory_id"],
                    user_id=user_id,
                    strategy=m["strategy"],
                    content=m["content"],
                    summary=m["summary"],
                    confidence=m.get("confidence", 1.0),
                ))
        
        # Get recent messages
        recent_messages = []
        if include_recent_messages:
            if not session_id:
                session = await self.stm.get_active_session(user_id)
                if session:
                    session_id = session.session_id
            
            if session_id:
                recent_messages = await self.stm.get_messages(session_id, message_limit)
        
        return UserContext(
            user_id=user_id,
            ltm_status=ltm_status,
            memories=memories,
            recent_messages=recent_messages,
            context_string=context_string,
        )
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all backends.
        
        Returns:
            Dict with health status of each component
        """
        redis_ok, milvus_ok = await asyncio.gather(
            self.stm.health_check(),
            self.ltm.health_check(),
        )
        
        return {
            "redis": redis_ok,
            "milvus": milvus_ok,
            "connected": self._connected,
        }

