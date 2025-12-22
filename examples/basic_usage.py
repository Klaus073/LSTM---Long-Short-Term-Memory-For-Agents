#!/usr/bin/env python3
"""
Basic usage example for Memory SDK.

Run:
    python examples/basic_usage.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Add parent to path for local development
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory_sdk import MemoryClient


async def main():
    print("=" * 50)
    print("  Memory SDK - Basic Usage")
    print("=" * 50)
    
    # Create client
    print("\n1. Connecting to backends...")
    client = await MemoryClient.create()
    
    # Health check
    health = await client.health_check()
    print(f"   Health: {health}")
    
    user_id = "demo_user"
    session_id = "session-1"
    
    # Clean up previous data
    await client.delete_memories(user_id)
    
    # Add messages
    print("\n2. Adding messages to session...")
    
    await client.add_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content="Hi! I'm John, a software engineer at TechCorp."
    )
    
    await client.add_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content="Nice to meet you John! What brings you here today?"
    )
    
    await client.add_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content="I love Python and enjoy hiking on weekends. Also, I prefer morning meetings."
    )
    
    # Get messages
    messages = await client.get_messages(session_id)
    print(f"   Added {len(messages)} messages")
    
    # Extract memories
    print("\n3. Extracting memories...")
    result = await client.extract_memories(user_id, session_id)
    print(f"   Extracted: {result.memories_extracted} new, {result.memories_updated} updated")
    print(f"   Strategies: {result.strategies_processed}")
    print(f"   Duration: {result.duration_ms:.0f}ms")
    
    # Get context
    print("\n4. Getting user context...")
    context = await client.get_context(user_id)
    print(f"   Status: {context.ltm_status}")
    print(f"   Context: {context.context_string[:100]}...")
    
    # Search memories
    print("\n5. Searching memories...")
    results = await client.search_memories(user_id, "What does John like to do?")
    print(f"   Found {len(results)} results:")
    for r in results[:3]:
        print(f"   - [{r.score:.2f}] {r.memory.summary[:60]}...")
    
    # New session with memory
    print("\n6. Starting new session...")
    session_id_2 = "session-2"
    
    context = await client.get_context(user_id)
    print(f"   System prompt: {context.to_system_prompt()[:80]}...")
    
    # Cleanup
    print("\n7. Cleaning up...")
    await client.delete_memories(user_id)
    await client.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())

