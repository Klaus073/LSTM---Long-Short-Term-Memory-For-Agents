#!/usr/bin/env python3
"""
LangGraph agent example with Memory SDK.

Run:
    python memory_sdk/examples/langgraph_agent.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from memory_sdk.integrations import MemoryCheckpointer, MemoryStore


def main():
    print("=" * 50)
    print("  Memory SDK - LangGraph Agent")
    print("=" * 50)
    
    # Initialize memory
    print("\n1. Initializing memory...")
    checkpointer = MemoryCheckpointer()  # STM
    store = MemoryStore()                 # LTM
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Create agent
    print("2. Creating agent...")
    graph = create_react_agent(
        model=llm,
        tools=[],
        checkpointer=checkpointer,
        store=store,
        pre_model_hook=store.memory_hook,
    )
    
    user_id = "alice"
    
    # Clean up any previous data
    store.clear_memories(user_id)
    
    # Session 1: User introduces themselves
    print("\n--- Session 1 ---")
    config = {"configurable": {"thread_id": "session-1", "user_id": user_id}}
    
    response = graph.invoke(
        {"messages": [("human", "Hi! I'm Alice, I work at TechCorp and love Python.")]},
        config
    )
    print(f"User: Hi! I'm Alice, I work at TechCorp and love Python.")
    print(f"Agent: {response['messages'][-1].content}")
    
    response = graph.invoke(
        {"messages": [("human", "I also enjoy sushi and hiking on weekends.")]},
        config
    )
    print(f"\nUser: I also enjoy sushi and hiking on weekends.")
    print(f"Agent: {response['messages'][-1].content}")
    
    # Extract memories
    print("\n--- Extracting Memories ---")
    count = store.extract_from_session(user_id, "session-1")
    print(f"Extracted {count} memories")
    
    # Show context
    ctx = store.get_user_context(user_id)
    print(f"Context: {ctx['context_string'][:80]}...")
    
    # Session 2: New session - agent should remember
    print("\n--- Session 2 (new session) ---")
    config = {"configurable": {"thread_id": "session-2", "user_id": user_id}}
    
    response = graph.invoke(
        {"messages": [("human", "What should I have for lunch?")]},
        config
    )
    print(f"User: What should I have for lunch?")
    print(f"Agent: {response['messages'][-1].content}")
    
    response = graph.invoke(
        {"messages": [("human", "What do you know about me?")]},
        config
    )
    print(f"\nUser: What do you know about me?")
    print(f"Agent: {response['messages'][-1].content}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    store.clear_memories(user_id)
    store.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
