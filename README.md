# Memory SDK

**Production-grade Long-Term and Short-Term Memory for AI Agents**

A fast, async-first memory framework for building agents with persistent memory. Designed for production use with Redis (STM) and Milvus (LTM).

## Features

- 🚀 **Async-first** - Built for high performance with async/await
- 🧠 **Short-Term Memory** - Session-scoped conversation history in Redis
- 📚 **Long-Term Memory** - Extracted insights stored in Milvus (vector DB)
- 🔍 **Semantic Search** - Find relevant memories using embeddings
- 🔗 **LangGraph Integration** - Drop-in replacement for checkpointers and stores
- ⚡ **Fast** - Connection pooling, batching, and caching

## Installation

```bash
pip install memory-sdk

# With LangGraph support
pip install memory-sdk[langgraph]

# With all providers
pip install memory-sdk[all]
```

## Quick Start

### Basic Usage

```python
from memory_sdk import MemoryClient

async def main():
    # Create client
    client = await MemoryClient.create()
    
    # Add messages
    await client.add_message(
        user_id="alice",
        session_id="session-1",
        role="user",
        content="Hi! I'm Alice, a Python developer."
    )
    
    # Extract memories after session
    result = await client.extract_memories("alice", "session-1")
    print(f"Extracted {result.memories_extracted} memories")
    
    # Get context for next session
    context = await client.get_context("alice")
    print(f"User context: {context.context_string}")
    
    # Search memories
    results = await client.search_memories("alice", "What does Alice do?")
    
    await client.close()

asyncio.run(main())
```

### LangGraph Integration

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from memory_sdk.integrations import MemoryCheckpointer, MemoryStore

# Initialize memory
checkpointer = MemoryCheckpointer()  # STM
store = MemoryStore()                 # LTM

# Create agent
graph = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    checkpointer=checkpointer,
    store=store,
    pre_model_hook=store.memory_hook,  # Auto-inject memories
)

# Use it
config = {"configurable": {"thread_id": "session-1", "user_id": "alice"}}
response = graph.invoke({"messages": [("human", "Hi!")]}, config)

# Extract memories after session
store.extract_from_session("alice", "session-1")
```

## Configuration

### Environment Variables

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# OpenAI
OPENAI_API_KEY=sk-...

# LLM Provider (openai, anthropic, ollama)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Embedding Provider (openai, ollama)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### Programmatic Configuration

```python
from memory_sdk import MemoryClient, MemoryConfig

config = MemoryConfig()
config.redis.host = "localhost"
config.redis.port = 6379
config.milvus.host = "localhost"
config.llm.provider = "openai"
config.llm.model = "gpt-4o-mini"

client = await MemoryClient.create(config)
```

## Memory Extraction Strategies

The SDK extracts structured memories using LLM-based extraction. Default strategies:

- **profile**: User profile (name, job, location, expertise)
- **preferences**: User preferences (interests, favorites, dislikes)
- **facts**: Important facts about the user

### Custom Strategies

```python
from memory_sdk.extraction import Strategy, DEFAULT_STRATEGIES

custom_strategy = Strategy(
    name="goals",
    description="User's goals and objectives",
    schema={
        "type": "object",
        "properties": {
            "short_term": {"type": "array", "items": {"type": "string"}},
            "long_term": {"type": "array", "items": {"type": "string"}},
        },
    },
)

DEFAULT_STRATEGIES["goals"] = custom_strategy
```

## API Reference

### MemoryClient

| Method | Description |
|--------|-------------|
| `create()` | Create and connect a client |
| `add_message()` | Add a message to a session |
| `get_messages()` | Get messages from a session |
| `extract_memories()` | Extract memories from a session |
| `get_memories()` | Get all memories for a user |
| `search_memories()` | Semantic search for memories |
| `get_context()` | Get full context (STM + LTM) |
| `delete_memories()` | Delete all memories for a user |
| `health_check()` | Check backend health |

### LangGraph Integration

| Class | Description |
|-------|-------------|
| `MemoryCheckpointer` | STM - Conversation state persistence |
| `MemoryStore` | LTM - Long-term memory storage |
| `create_memory_hook()` | Factory for pre-model hooks |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MemoryClient                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌─────────────────┐             │
│  │   STMBackend    │      │   LTMBackend    │             │
│  │    (Redis)      │      │   (Milvus)      │             │
│  └────────┬────────┘      └────────┬────────┘             │
│           │                        │                       │
│  ┌────────▼────────┐      ┌────────▼────────┐             │
│  │ Sessions        │      │ Memories        │             │
│  │ Messages        │      │ Embeddings      │             │
│  │ LTM Cache       │      │ Search          │             │
│  └─────────────────┘      └─────────────────┘             │
│                                                             │
│  ┌─────────────────────────────────────────────┐          │
│  │           MemoryExtractor                    │          │
│  │  - LLM-based extraction                      │          │
│  │  - Strategies (profile, preferences, facts) │          │
│  │  - Embedding generation                      │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- Redis 7.0+
- Milvus 2.4+
- OpenAI API key (or Ollama for local models)

## License

MIT

