"""
Memory SDK - Production-grade Long-Term and Short-Term Memory for AI Agents

A fast, async-first memory framework for building agents with persistent memory.

Usage:
    from memory_sdk import MemoryClient
    
    # Initialize
    client = MemoryClient()
    
    # Or async
    client = await MemoryClient.create()
    
    # Add memory
    await client.add_message(user_id="alice", session_id="s1", role="user", content="Hi!")
    
    # Get context for LLM
    context = await client.get_context(user_id="alice")
    
    # Extract memories after session
    await client.extract_memories(user_id="alice", session_id="s1")
    
    # LangGraph integration
    from memory_sdk.integrations import MemoryCheckpointer, MemoryStore
"""

__version__ = "0.1.0"
__author__ = "Memory SDK Team"

from memory_sdk.client import MemoryClient
from memory_sdk.config import MemoryConfig
from memory_sdk.models import (
    Message,
    Memory,
    Session,
    UserContext,
    ExtractionResult,
)

__all__ = [
    "MemoryClient",
    "MemoryConfig",
    "Message",
    "Memory",
    "Session",
    "UserContext",
    "ExtractionResult",
]

