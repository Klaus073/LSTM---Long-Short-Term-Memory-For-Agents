"""
Async Redis backend for Short-Term Memory.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from memory_sdk.config import MemoryConfig
from memory_sdk.models import Message, Session, SessionStatus, LTMStatus, Role, Memory

logger = logging.getLogger(__name__)


class STMBackend:
    """Async Redis backend for short-term memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self._pool = aioredis.ConnectionPool(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            password=self.config.redis.password,
            max_connections=self.config.redis.max_connections,
            socket_timeout=self.config.redis.socket_timeout,
            decode_responses=True,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        
        # Test connection
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.config.redis.host}:{self.config.redis.port}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
    
    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client
    
    # =========================================================================
    # Session Operations
    # =========================================================================
    
    def _session_key(self, session_id: str) -> str:
        return f"{self.config.redis.session_prefix}{session_id}"
    
    def _user_session_key(self, user_id: str) -> str:
        return f"{self.config.redis.session_prefix}user:{user_id}"
    
    async def create_session(self, user_id: str, session_id: str, metadata: Optional[Dict] = None) -> Session:
        """Create a new session."""
        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": 0,
            "metadata": json.dumps(session.metadata),
        }
        
        pipe = self.client.pipeline()
        pipe.hset(self._session_key(session_id), mapping=data)
        pipe.expire(self._session_key(session_id), self.config.redis.session_ttl)
        pipe.set(self._user_session_key(user_id), session_id, ex=self.config.redis.session_ttl)
        await pipe.execute()
        
        logger.debug(f"Created session {session_id} for user {user_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        data = await self.client.hgetall(self._session_key(session_id))
        if not data:
            return None
        
        return Session(
            session_id=data["session_id"],
            user_id=data["user_id"],
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            message_count=int(data["message_count"]),
            metadata=json.loads(data.get("metadata", "{}")),
        )
    
    async def get_active_session(self, user_id: str) -> Optional[Session]:
        """Get active session for user."""
        session_id = await self.client.get(self._user_session_key(user_id))
        if not session_id:
            return None
        return await self.get_session(session_id)
    
    async def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update session status."""
        await self.client.hset(
            self._session_key(session_id),
            mapping={
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
    
    async def close_session(self, session_id: str) -> None:
        """Close a session."""
        await self.update_session_status(session_id, SessionStatus.CLOSED)
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its messages."""
        session = await self.get_session(session_id)
        if session:
            await self.client.delete(self._user_session_key(session.user_id))
        
        # Delete messages
        msg_key = f"{self.config.redis.message_prefix}{session_id}"
        await self.client.delete(msg_key, self._session_key(session_id))
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    def _message_key(self, session_id: str) -> str:
        return f"{self.config.redis.message_prefix}{session_id}"
    
    async def add_message(self, message: Message) -> None:
        """Add a message to a session."""
        data = {
            "message_id": message.message_id,
            "role": message.role.value,
            "content": message.content,
            "user_id": message.user_id,
            "session_id": message.session_id,
            "timestamp": message.timestamp.isoformat(),
            "metadata": json.dumps(message.metadata),
        }
        
        pipe = self.client.pipeline()
        pipe.rpush(self._message_key(message.session_id), json.dumps(data))
        pipe.expire(self._message_key(message.session_id), self.config.redis.message_ttl)
        pipe.hincrby(self._session_key(message.session_id), "message_count", 1)
        pipe.hset(self._session_key(message.session_id), "updated_at", datetime.utcnow().isoformat())
        await pipe.execute()
    
    async def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get messages from a session."""
        key = self._message_key(session_id)
        
        if limit:
            data = await self.client.lrange(key, -limit, -1)
        else:
            data = await self.client.lrange(key, 0, -1)
        
        messages = []
        for item in data:
            d = json.loads(item)
            messages.append(Message(
                message_id=d["message_id"],
                role=Role(d["role"]),
                content=d["content"],
                user_id=d["user_id"],
                session_id=d["session_id"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
                metadata=json.loads(d.get("metadata", "{}")),
            ))
        
        return messages
    
    # =========================================================================
    # LTM Cache Operations
    # =========================================================================
    
    def _ltm_cache_key(self, user_id: str) -> str:
        return f"{self.config.redis.ltm_cache_prefix}{user_id}"
    
    def _ltm_status_key(self, user_id: str) -> str:
        return f"{self.config.redis.ltm_cache_prefix}status:{user_id}"
    
    async def set_ltm_cache(self, user_id: str, memories: List[Memory]) -> None:
        """Cache LTM for a user."""
        # Build context string
        summaries = [m.summary for m in memories if m.summary]
        context_string = " ".join(summaries)
        
        # Store memories
        cache_data = {
            "user_id": user_id,
            "updated_at": datetime.utcnow().isoformat(),
            "context_string": context_string,
            "memories": json.dumps([
                {
                    "memory_id": m.memory_id,
                    "strategy": m.strategy,
                    "content": m.content,
                    "summary": m.summary,
                    "confidence": m.confidence,
                }
                for m in memories
            ]),
        }
        
        pipe = self.client.pipeline()
        pipe.hset(self._ltm_cache_key(user_id), mapping=cache_data)
        pipe.expire(self._ltm_cache_key(user_id), self.config.redis.ltm_cache_ttl)
        pipe.set(self._ltm_status_key(user_id), LTMStatus.READY.value, ex=self.config.redis.ltm_cache_ttl)
        await pipe.execute()
    
    async def get_ltm_cache(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached LTM for a user."""
        data = await self.client.hgetall(self._ltm_cache_key(user_id))
        if not data:
            return None
        
        return {
            "user_id": data["user_id"],
            "updated_at": data["updated_at"],
            "context_string": data.get("context_string", ""),
            "memories": json.loads(data.get("memories", "[]")),
        }
    
    async def get_ltm_status(self, user_id: str) -> LTMStatus:
        """Get LTM status for a user."""
        status = await self.client.get(self._ltm_status_key(user_id))
        if not status:
            return LTMStatus.EMPTY
        return LTMStatus(status)
    
    async def set_ltm_status(self, user_id: str, status: LTMStatus) -> None:
        """Set LTM status for a user."""
        await self.client.set(
            self._ltm_status_key(user_id),
            status.value,
            ex=self.config.redis.ltm_cache_ttl
        )
    
    async def delete_ltm_cache(self, user_id: str) -> None:
        """Delete LTM cache for a user."""
        await self.client.delete(
            self._ltm_cache_key(user_id),
            self._ltm_status_key(user_id)
        )
    
    # =========================================================================
    # Checkpoint Operations (for LangGraph)
    # =========================================================================
    
    def _checkpoint_key(self, thread_id: str) -> str:
        return f"{self.config.redis.checkpoint_prefix}{thread_id}"
    
    async def save_checkpoint(self, thread_id: str, data: Dict[str, Any]) -> None:
        """Save LangGraph checkpoint."""
        await self.client.set(
            self._checkpoint_key(thread_id),
            json.dumps(data),
            ex=self.config.redis.session_ttl
        )
    
    async def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get LangGraph checkpoint."""
        data = await self.client.get(self._checkpoint_key(thread_id))
        if not data:
            return None
        return json.loads(data)
    
    async def delete_checkpoint(self, thread_id: str) -> None:
        """Delete LangGraph checkpoint."""
        await self.client.delete(self._checkpoint_key(thread_id))
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

