"""
LangGraph integration for Memory SDK.

Uses synchronous operations for compatibility with LangGraph's threading model.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import redis

from memory_sdk.config import MemoryConfig
from memory_sdk.models import LTMStatus

logger = logging.getLogger(__name__)

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
    
    Uses synchronous Redis for compatibility with LangGraph threading.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required. Install: pip install langgraph langchain-core")
        
        super().__init__()
        self.config = config or MemoryConfig.from_env()
        self.serde = JsonPlusSerializer()
        
        # Sync Redis client
        self._redis = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            password=self.config.redis.password,
            decode_responses=True,
        )
    
    def _serialize(self, data: Any) -> str:
        """Serialize data for storage."""
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
        """Deserialize data from storage."""
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
        """Get checkpoint tuple."""
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
        """Save checkpoint."""
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
    LangGraph store with memory.
    
    Uses synchronous operations for LangGraph compatibility.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph required. Install: pip install langgraph langchain-core")
        
        self.config = config or MemoryConfig.from_env()
        
        # Sync Redis
        self._redis = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            password=self.config.redis.password,
            decode_responses=True,
        )
        
        # LLM and embedding clients (lazy)
        self._llm = None
        self._embeddings = None
    
    @property
    def llm(self):
        if self._llm is None:
            from openai import OpenAI
            self._llm = OpenAI(api_key=self.config.llm.openai_api_key)
        return self._llm
    
    @property  
    def embeddings(self):
        if self._embeddings is None:
            from openai import OpenAI
            self._embeddings = OpenAI(api_key=self.config.embedding.openai_api_key)
        return self._embeddings
    
    def _ltm_key(self, user_id: str) -> str:
        return f"mem:ltm:{user_id}"
    
    def _msg_key(self, session_id: str) -> str:
        return f"mem:msg:{session_id}"
    
    # BaseStore interface
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Item]:
        if len(namespace) < 1:
            return None
        user_id = namespace[0]
        data = self._redis.hget(self._ltm_key(user_id), key)
        if not data:
            return None
        return Item(
            namespace=namespace,
            key=key,
            value=json.loads(data),
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
        # Return cached memories
        if len(namespace_prefix) < 1:
            return []
        user_id = namespace_prefix[0]
        data = self._redis.hgetall(self._ltm_key(user_id))
        results = []
        for key, value in list(data.items())[offset:offset + limit]:
            results.append(SearchItem(
                namespace=namespace_prefix,
                key=key,
                value=json.loads(value),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                score=1.0,
            ))
        return results
    
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
        
        # Save message
        if "message" in value:
            msg = value["message"]
            content = msg.content if hasattr(msg, "content") else str(msg)
            role = "user"
            if hasattr(msg, "type"):
                role = "assistant" if msg.type == "ai" else "user"
            
            self._redis.rpush(
                self._msg_key(session_id),
                json.dumps({"role": role, "content": content, "user_id": user_id})
            )
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        if len(namespace) < 1:
            return
        user_id = namespace[0]
        self._redis.hdel(self._ltm_key(user_id), key)
    
    def list(
        self,
        namespace_prefix: Tuple[str, ...],
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Item]:
        return [SearchItem(**r.__dict__) for r in self.search(namespace_prefix, limit=limit, offset=offset)]
    
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
    
    # Memory methods
    @property
    def memory_hook(self):
        """Pre-model hook that injects memories and saves messages."""
        store = self
        
        def hook(state: Dict, config=None, **kwargs) -> Dict:
            if config is None:
                config = kwargs.get("config", {"configurable": {}})
            
            user_id = config.get("configurable", {}).get("user_id", "default")
            session_id = config.get("configurable", {}).get("thread_id", "default")
            messages = list(state.get("messages", []))
            
            # Save last user message for extraction
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    store._redis.rpush(
                        store._msg_key(session_id),
                        json.dumps({"role": "user", "content": msg.content, "user_id": user_id})
                    )
                    break
            
            # Get cached LTM
            context = store.get_user_context(user_id)
            if context.get("context_string"):
                memory_msg = SystemMessage(
                    content=f"What you know about this user:\n{context['context_string']}\n\nUse this to personalize."
                )
                messages.insert(0, memory_msg)
            
            return {"llm_input_messages": messages}
        
        return hook
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get cached LTM for user."""
        data = self._redis.hgetall(self._ltm_key(user_id))
        
        summaries = []
        memories = []
        for key, value in data.items():
            try:
                mem = json.loads(value)
                summaries.append(mem.get("summary", ""))
                memories.append(mem)
            except Exception:
                pass
        
        return {
            "context_string": " ".join(filter(None, summaries)),
            "memories": memories,
            "ltm_status": "ready" if data else "empty",
        }
    
    def extract_from_session(self, user_id: str, session_id: str) -> int:
        """Extract memories from session messages."""
        # Get messages
        messages_raw = self._redis.lrange(self._msg_key(session_id), 0, -1)
        if not messages_raw:
            return 0
        
        messages = [json.loads(m) for m in messages_raw]
        
        # Build conversation text
        conversation = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        ])
        
        # Get existing memories
        existing = self._redis.hgetall(self._ltm_key(user_id))
        existing_str = ""
        if existing:
            for key, value in existing.items():
                try:
                    mem = json.loads(value)
                    existing_str += f"\n{key}: {mem.get('summary', '')}"
                except Exception:
                    pass
        
        # Extract with LLM
        prompt = f"""Extract structured memories from this conversation.

CONVERSATION:
{conversation}

EXISTING MEMORIES:
{existing_str or 'None'}

Return a JSON object with these fields:
- profile: {{name, job_title, company, location}} - user profile info
- preferences: {{interests: [], favorites: [], dislikes: []}} - user preferences  
- facts: {{facts: []}} - important facts about user

For each field, include a "summary" with a 1-2 sentence plain English description.
Only include fields where you found relevant information.
Merge with existing memories."""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.config.llm.model,
                messages=[
                    {"role": "system", "content": "Extract structured memories from conversations."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            data = json.loads(response.choices[0].message.content)
            
            # Save each memory type
            count = 0
            for strategy in ["profile", "preferences", "facts"]:
                if strategy in data and data[strategy]:
                    mem_data = data[strategy]
                    
                    # Generate summary if not present
                    if "summary" not in mem_data:
                        mem_data["summary"] = f"{strategy}: {json.dumps(mem_data)[:100]}"
                    
                    self._redis.hset(
                        self._ltm_key(user_id),
                        strategy,
                        json.dumps(mem_data)
                    )
                    count += 1
            
            return count
        
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return 0
    
    def clear_memories(self, user_id: str) -> None:
        """Clear all memories for user."""
        self._redis.delete(self._ltm_key(user_id))
    
    def close(self) -> None:
        """Close Redis connection."""
        self._redis.close()


def create_memory_hook(store: MemoryStore):
    """Factory for pre-model hook."""
    return store.memory_hook
