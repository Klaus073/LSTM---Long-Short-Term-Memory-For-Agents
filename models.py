"""
Data models for Memory SDK.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class Role(str, Enum):
    """Message role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    CLOSED = "closed"
    PROCESSING = "processing"
    PROCESSED = "processed"


class LTMStatus(str, Enum):
    """Long-term memory status."""
    EMPTY = "empty"
    READY = "ready"
    PROCESSING = "processing"


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: str
    user_id: str
    session_id: str
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """An extracted long-term memory."""
    memory_id: str
    user_id: str
    strategy: str  # e.g., "profile", "preferences", "facts"
    content: Dict[str, Any]  # Structured payload
    summary: str  # Plain English summary for embedding
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserContext:
    """Complete context for a user (STM + LTM)."""
    user_id: str
    ltm_status: LTMStatus
    memories: List[Memory] = field(default_factory=list)
    recent_messages: List[Message] = field(default_factory=list)
    context_string: str = ""
    
    def to_system_prompt(self) -> str:
        """Format as a system prompt for LLM."""
        if not self.context_string:
            return ""
        return f"What you know about this user from past conversations:\n{self.context_string}"


@dataclass
class ExtractionResult:
    """Result of memory extraction."""
    user_id: str
    session_id: str
    memories_extracted: int
    memories_updated: int
    strategies_processed: List[str]
    duration_ms: float


@dataclass 
class SearchResult:
    """Result from semantic memory search."""
    memory: Memory
    score: float
    
    
@dataclass
class Strategy:
    """Memory extraction strategy."""
    name: str
    description: str
    schema: Dict[str, Any]
    prompt_template: Optional[str] = None
    enabled: bool = True

