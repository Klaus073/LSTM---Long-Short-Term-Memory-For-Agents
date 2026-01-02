"""
LangGraph integration for Memory SDK.

Features:
- Fire-and-forget LTM extraction (zero user latency)
- STM bridging during LTM processing
- Automatic context switching when LTM ready
"""

import asyncio
import base64
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import redis

from memory_sdk.config import MemoryConfig

logger = logging.getLogger(__name__)

# Thread pool for background tasks
_executor = ThreadPoolExecutor(max_workers=10)

# Check for LangGraph availability
try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.store.base import BaseStore, Item, Op, Result, SearchItem
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object
    BaseStore = object


class MemoryCheckpointer(BaseCheckpointSaver if LANGGRAPH_AVAILABLE else object):
    """
    LangGraph checkpointer using Redis.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required. Install: pip install langgraph langchain-core")
        
        super().__init__()
        self.config = config or MemoryConfig.from_env()
        self.serde = JsonPlusSerializer()
        
        self._redis = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            password=self.config.redis.password,
            decode_responses=True,
        )
    
    def _serialize(self, data: Any) -> str:
        serialized = self.serde.dumps_typed(data)
        if isinstance(serialized, (bytes, tuple)):
            if isinstance(serialized, tuple):
                type_str, payload = serialized
                if isinstance(payload, bytes):
                    return json.dumps([type_str, base64.b64encode(payload).decode('utf-8')])
                return json.dumps(list(serialized))
            return base64.b64encode(serialized).decode('utf-8')
        return json.dumps(serialized)
    
    def _deserialize(self, data_str: str) -> Any:
        try:
            loaded = json.loads(data_str)
            if isinstance(loaded, list) and len(loaded) == 2 and isinstance(loaded[1], str):
                try:
                    type_str, payload_str = loaded
                    payload = base64.b64decode(payload_str)
                    return self.serde.loads_typed((type_str, payload))
                except Exception:
                    pass
            return self.serde.loads_typed(tuple(loaded) if isinstance(loaded, list) else loaded)
        except json.JSONDecodeError:
            decoded = base64.b64decode(data_str)
            return self.serde.loads_typed(decoded)
    
    def _key(self, thread_id: str) -> str:
        return f"mem:ckpt:{thread_id}"
    
    def get_tuple(self, config: Dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        data_str = self._redis.get(self._key(thread_id))
        
        if not data_str:
            return None
        
        data = json.loads(data_str)
        checkpoint = self._deserialize(data["checkpoint"])
        metadata = self._deserialize(data.get("metadata", "{}"))
        
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=data.get("parent_config"),
        )
    
    def put(
        self,
        config: Dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict:
        thread_id = config["configurable"]["thread_id"]
        
        data = {
            "checkpoint": self._serialize(checkpoint),
            "metadata": self._serialize(metadata),
            "parent_config": config.get("configurable", {}).get("checkpoint_id"),
        }
        
        self._redis.set(self._key(thread_id), json.dumps(data), ex=3600)
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": config.get("configurable", {}).get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint["id"],
            }
        }
    
    def put_writes(
        self,
        config: Dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        pass
    
    def list(
        self,
        config: Optional[Dict] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config:
            result = self.get_tuple(config)
            if result:
                yield result


class MemoryStore(BaseStore if LANGGRAPH_AVAILABLE else object):
    """
    LangGraph store with fire-and-forget LTM extraction.
    
    Features:
    - Zero latency session end (fire-and-forget)
    - STM bridging during LTM processing
    - Automatic context switching when ready
    """
    
    # Redis key prefixes
    PREFIX_LTM = "mem:ltm:"           # LTM content
    PREFIX_STM = "mem:msg:"           # STM messages
    PREFIX_STATUS = "mem:status:"     # LTM processing status
    PREFIX_PREV = "mem:prev:"         # Previous session for bridging
    
    # LTM Status values
    STATUS_READY = "ready"
    STATUS_PROCESSING = "processing"
    
    # System prompt for LTM extraction
    EXTRACTION_SYSTEM_PROMPT = """##Your Role##
You are user's long-term memory assistant. Your job is to build and maintain a user's long-term memory.

##YOUR TASK##
- Extract user's information from the RECENT SESSION and create/update the user's long-term memory.

##CRITICAL RULES##
1. If EXISTING LONG-TERM MEMORY is empty: Create new memory from the session
2. If EXISTING LONG-TERM MEMORY exists: UPDATE and APPEND new information
3. NEVER REMOVE existing information - only add or update
4. If a field has new value, update it (e.g., job changed from "Engineer" to "Senior Engineer")
5. For list items: APPEND new items, keep existing ones
6. Output in clean Markdown format

##Key Points - Personify the User##
Extract everything that helps personify the user:
- Name, job, company, location
- Likes, dislikes, preferences
- Activities, interests, hobbies
- Goals, dreams, aspirations
- Important facts
- Any other personal information

##OUTPUT FORMAT##
Return ONLY the user memory in Markdown format (omit sections with no info):

## About
[Name], [Job Title] at [Company]. [Other personal details].

## Interests & Preferences
- [Interest/preference 1]
- [Interest/preference 2]

## Facts
- [Fact 1]
- [Fact 2]

## Other
- [Any other relevant info]"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required. Install: pip install langgraph langchain-core")
        
        self.config = config or MemoryConfig.from_env()
        
        self._redis = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            password=self.config.redis.password,
            decode_responses=True,
        )
        
        self._llm = None
    
    @property
    def llm(self):
        if self._llm is None:
            from openai import OpenAI
            self._llm = OpenAI(api_key=self.config.llm.openai_api_key)
        return self._llm
    
    # =========================================================================
    # Redis Key Helpers
    # =========================================================================
    
    def _ltm_key(self, user_id: str) -> str:
        return f"{self.PREFIX_LTM}{user_id}"
    
    def _stm_key(self, session_id: str) -> str:
        return f"{self.PREFIX_STM}{session_id}"
    
    def _status_key(self, user_id: str) -> str:
        return f"{self.PREFIX_STATUS}{user_id}"
    
    def _prev_session_key(self, user_id: str) -> str:
        return f"{self.PREFIX_PREV}{user_id}"
    
    # =========================================================================
    # STM Operations
    # =========================================================================
    
    def save_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """Save a message to STM."""
        self._redis.rpush(
            self._stm_key(session_id),
            json.dumps({"role": role, "content": content, "user_id": user_id})
        )
    
    def get_stm_messages(self, session_id: str) -> List[Dict]:
        """Get all messages from a session."""
        messages_raw = self._redis.lrange(self._stm_key(session_id), 0, -1)
        return [json.loads(m) for m in messages_raw] if messages_raw else []
    
    def delete_stm(self, session_id: str) -> None:
        """Delete STM for a session."""
        self._redis.delete(self._stm_key(session_id))
    
    # =========================================================================
    # LTM Operations
    # =========================================================================
    
    def get_ltm(self, user_id: str) -> Optional[str]:
        """Get LTM content for a user."""
        key = self._ltm_key(user_id)
        key_type = self._redis.type(key)
        
        # Handle old HASH format migration
        if key_type == "hash":
            self._redis.delete(key)
            return None
        
        return self._redis.get(key)
    
    def set_ltm(self, user_id: str, content: str) -> None:
        """Set LTM content for a user."""
        self._redis.set(self._ltm_key(user_id), content)
    
    def get_ltm_status(self, user_id: str) -> str:
        """Get LTM processing status."""
        return self._redis.get(self._status_key(user_id)) or self.STATUS_READY
    
    def set_ltm_status(self, user_id: str, status: str) -> None:
        """Set LTM processing status."""
        self._redis.set(self._status_key(user_id), status, ex=300)  # 5 min TTL
    
    def get_prev_session(self, user_id: str) -> Optional[str]:
        """Get previous session ID for bridging."""
        return self._redis.get(self._prev_session_key(user_id))
    
    def set_prev_session(self, user_id: str, session_id: str) -> None:
        """Set previous session ID for bridging."""
        self._redis.set(self._prev_session_key(user_id), session_id, ex=300)  # 5 min TTL
    
    def clear_prev_session(self, user_id: str) -> None:
        """Clear previous session reference."""
        self._redis.delete(self._prev_session_key(user_id))
    
    # =========================================================================
    # Context Retrieval (with bridging)
    # =========================================================================
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get context for a user.
        
        If LTM is processing, includes previous session STM as bridge.
        Returns immediately - never waits for LTM extraction.
        """
        status = self.get_ltm_status(user_id)
        ltm = self.get_ltm(user_id) or ""
        
        context_parts = []
        bridge_stm = ""
        
        # Always include LTM if available
        if ltm:
            context_parts.append(ltm)
        
        # If processing, add bridge STM from previous session
        if status == self.STATUS_PROCESSING:
            prev_session_id = self.get_prev_session(user_id)
            if prev_session_id:
                prev_messages = self.get_stm_messages(prev_session_id)
                if prev_messages:
                    bridge_stm = "\n\n## Recent Conversation (pending processing)\n"
                    for msg in prev_messages[-10:]:  # Last 10 messages
                        bridge_stm += f"- {msg['role'].upper()}: {msg['content'][:200]}\n"
                    context_parts.append(bridge_stm)
        
        return {
            "context_string": "\n".join(context_parts),
            "ltm": ltm,
            "bridge_stm": bridge_stm,
            "ltm_status": status,
            "is_processing": status == self.STATUS_PROCESSING,
        }
    
    # =========================================================================
    # Session End & LTM Extraction
    # =========================================================================
    
    def end_session(self, user_id: str, session_id: str) -> None:
        """
        End a session and trigger LTM extraction.
        
        FIRE-AND-FORGET: Returns immediately (0ms latency).
        LTM extraction happens in background.
        Previous session STM is kept as bridge until LTM is ready.
        """
        # Check if there are messages to extract
        messages = self.get_stm_messages(session_id)
        if not messages:
            return
        
        # Set status to processing
        self.set_ltm_status(user_id, self.STATUS_PROCESSING)
        
        # Keep this session for bridging
        self.set_prev_session(user_id, session_id)
        
        # Fire-and-forget: submit to thread pool
        _executor.submit(self._extract_ltm_background, user_id, session_id)
        
        logger.info(f"LTM extraction queued for user {user_id}, session {session_id}")
    
    def _extract_ltm_background(self, user_id: str, session_id: str) -> None:
        """
        Background LTM extraction task.
        
        Called by thread pool - do not call directly.
        """
        try:
            logger.info(f"Starting LTM extraction for user {user_id}")
            
            # Get messages
            messages = self.get_stm_messages(session_id)
            if not messages:
                self.set_ltm_status(user_id, self.STATUS_READY)
                return
            
            # Build conversation text
            conversation = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in messages
            ])
            
            # Get existing LTM
            existing_ltm = self.get_ltm(user_id)
            
            # Build prompt
            if existing_ltm:
                existing_section = f"##EXISTING LONG-TERM MEMORY##\n{existing_ltm}"
            else:
                existing_section = "##EXISTING LONG-TERM MEMORY##\nNone - This is a new user with no saved memory yet."
            
            prompt = f"""{existing_section}

##RECENT SESSION (Just Happened)##
{conversation}

Now output the updated user memory in Markdown format."""
            
            # Call LLM
            response = self.llm.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            
            # Save new LTM
            new_ltm = response.choices[0].message.content.strip()
            self.set_ltm(user_id, new_ltm)
            
            # Cleanup: set status to ready, clear bridge
            self.set_ltm_status(user_id, self.STATUS_READY)
            self.clear_prev_session(user_id)
            self.delete_stm(session_id)
            
            logger.info(f"LTM extraction complete for user {user_id}")
            
        except Exception as e:
            logger.error(f"LTM extraction failed for user {user_id}: {e}")
            # On failure, set status back to ready (will use old LTM)
            self.set_ltm_status(user_id, self.STATUS_READY)
    
    def extract_from_session(self, user_id: str, session_id: str) -> int:
        """
        Synchronous extraction (for backward compatibility).
        
        Use end_session() for fire-and-forget behavior.
        """
        messages = self.get_stm_messages(session_id)
        if not messages:
            return 0
        
        self._extract_ltm_background(user_id, session_id)
        return 1
    
    # =========================================================================
    # LangGraph BaseStore Interface
    # =========================================================================
    
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Item]:
        if len(namespace) < 1:
            return None
        user_id = namespace[0]
        ltm = self.get_ltm(user_id)
        if not ltm:
            return None
        return Item(
            namespace=namespace,
            key=key,
            value={"content": ltm},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[SearchItem]:
        if len(namespace_prefix) < 1:
            return []
        user_id = namespace_prefix[0]
        ltm = self.get_ltm(user_id)
        if not ltm:
            return []
        return [SearchItem(
            namespace=namespace_prefix,
            key="ltm",
            value={"content": ltm},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            score=1.0,
        )]
    
    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        index: Optional[Union[bool, List[str]]] = None,
    ) -> None:
        if len(namespace) < 2:
            return
        user_id = namespace[0]
        session_id = namespace[1]
        
        if "message" in value:
            msg = value["message"]
            content = msg.content if hasattr(msg, "content") else str(msg)
            role = "assistant" if (hasattr(msg, "type") and msg.type == "ai") else "user"
            self.save_message(user_id, session_id, role, content)
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        if len(namespace) < 1:
            return
        user_id = namespace[0]
        self._redis.delete(self._ltm_key(user_id))
    
    def list(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Item]:
        return [Item(**r.__dict__) for r in self.search(namespace_prefix, limit=limit, offset=offset)]
    
    def batch(self, ops: List[Op]) -> List[Result]:
        results = []
        for op in ops:
            try:
                if hasattr(op, 'namespace') and hasattr(op, 'key'):
                    if hasattr(op, 'value'):
                        self.put(op.namespace, op.key, op.value)
                        results.append(None)
                    else:
                        results.append(self.get(op.namespace, op.key))
                else:
                    results.append(None)
            except Exception:
                results.append(None)
        return results
    
    async def abatch(self, ops: List[Op]) -> List[Result]:
        return self.batch(ops)
    
    # =========================================================================
    # LangGraph Hook
    # =========================================================================
    
    @property
    def memory_hook(self):
        """Pre-model hook that injects LTM and saves messages."""
        store = self
        
        def hook(state: Dict, config=None, **kwargs) -> Dict:
            if config is None:
                config = kwargs.get("config", {"configurable": {}})
            
            user_id = config.get("configurable", {}).get("user_id", "default")
            session_id = config.get("configurable", {}).get("thread_id", "default")
            messages = list(state.get("messages", []))
            
            # Save last user message to STM
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    store.save_message(user_id, session_id, "user", msg.content)
                    break
            
            # Get context (LTM + bridge if processing)
            context = store.get_user_context(user_id)
            if context.get("context_string"):
                memory_msg = SystemMessage(
                    content=f"## What you know about this user:\n{context['context_string']}\n\nUse this to personalize your responses."
                )
                messages.insert(0, memory_msg)
            
            return {"llm_input_messages": messages}
        
        return hook
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def clear_memories(self, user_id: str) -> None:
        """Clear all memories for a user."""
        self._redis.delete(
            self._ltm_key(user_id),
            self._status_key(user_id),
            self._prev_session_key(user_id),
        )
    
    def close(self) -> None:
        """Close Redis connection."""
        self._redis.close()


def create_memory_hook(store: MemoryStore):
    """Factory for pre-model hook."""
    return store.memory_hook
