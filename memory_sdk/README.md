# Memory SDK

**Production-grade Long-Term and Short-Term Memory for AI Agents**

A fast, async-first memory framework for building agents with persistent memory. Designed for production use with Redis (STM) and Milvus (LTM).

## Features

- 🚀 **Fire-and-Forget Extraction** - Zero-latency session end (0ms)
- 🌉 **STM Bridging** - Seamless context during LTM processing
- 🧠 **Short-Term Memory** - Session-scoped conversation history in Redis
- 📚 **Long-Term Memory** - Extracted user memory in Markdown format
- 🔗 **LangGraph Integration** - Drop-in checkpointers and stores
- ⚡ **Fast** - ~0.2ms context retrieval, background processing

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/Klaus073/LSTM---Long-Short-Term-Memory-For-Agents.git

# Or clone and install locally
git clone https://github.com/Klaus073/LSTM---Long-Short-Term-Memory-For-Agents.git
cd LSTM---Long-Short-Term-Memory-For-Agents/memory_sdk
pip install -e .
```

### Requirements

```bash
pip install redis pymilvus openai langchain-openai langgraph
```

## Quick Start

### LangGraph Integration (Recommended)

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from memory_sdk.integrations import MemoryCheckpointer, MemoryStore

# Initialize memory
checkpointer = MemoryCheckpointer()  # STM (conversation state)
store = MemoryStore()                 # LTM (user memory)

# Create agent
graph = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    checkpointer=checkpointer,
    store=store,
    pre_model_hook=store.memory_hook,  # Auto-inject LTM
)

# Use it
config = {"configurable": {"thread_id": "session-1", "user_id": "alice"}}
response = graph.invoke({"messages": [("human", "Hi! I'm Alice, a Python dev")]}, config)

# End session (FIRE-AND-FORGET - returns immediately!)
store.end_session("alice", "session-1")

# Next session - context available immediately (with bridging if needed)
config2 = {"configurable": {"thread_id": "session-2", "user_id": "alice"}}
response = graph.invoke({"messages": [("human", "What do you know about me?")]}, config2)
```

### Fire-and-Forget Flow

```
Session End → Returns in 0ms (fire-and-forget)
     ↓
Background: LTM extraction via LLM
     ↓
While processing: STM Bridge provides context
     ↓
LTM Ready: Switch to new LTM, bridge cleaned up
```

### Direct Store Usage

```python
from memory_sdk.integrations import MemoryStore

store = MemoryStore()

# Save messages
store.save_message("alice", "session-1", "user", "I love Python!")
store.save_message("alice", "session-1", "assistant", "Great choice!")

# End session (fire-and-forget)
store.end_session("alice", "session-1")

# Get context (includes bridge if processing)
context = store.get_user_context("alice")
print(context["context_string"])   # LTM content
print(context["ltm_status"])       # "ready" or "processing"
print(context["is_processing"])    # True/False

# Synchronous extraction (if needed)
store.extract_from_session("alice", "session-1")
```

## Configuration

### Environment Variables

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Milvus (optional - for vector search)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# OpenAI
OPENAI_API_KEY=sk-...

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Embedding Settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### Programmatic Configuration

```python
from memory_sdk import MemoryConfig
from memory_sdk.integrations import MemoryStore

config = MemoryConfig()
config.redis.host = "localhost"
config.redis.port = 6379
config.llm.model = "gpt-4o-mini"

store = MemoryStore(config)
```

## Memory Format

LTM is stored as a Markdown string for easy injection into prompts:

```markdown
## About
Alex, ML Engineer at Google.

## Interests & Preferences
- Hiking
- Photography
- Machine Learning

## Facts
- Works at Google as a Machine Learning Engineer
- Prefers Python for development
```

## API Reference

### MemoryStore

| Method | Description | Latency |
|--------|-------------|---------|
| `end_session(user_id, session_id)` | Fire-and-forget LTM extraction | **0ms** |
| `get_user_context(user_id)` | Get LTM + bridge if processing | ~0.2ms |
| `save_message(user_id, session_id, role, content)` | Save to STM | ~0.1ms |
| `get_stm_messages(session_id)` | Get session messages | ~0.2ms |
| `extract_from_session(user_id, session_id)` | Sync extraction | ~3-5s |
| `clear_memories(user_id)` | Delete all user memories | ~0.1ms |

### MemoryCheckpointer

| Method | Description |
|--------|-------------|
| `get_tuple(config)` | Get checkpoint state |
| `put(config, checkpoint, metadata)` | Save checkpoint state |

### Context Response

```python
context = store.get_user_context("alice")

context["context_string"]  # Full context (LTM + bridge)
context["ltm"]             # Raw LTM content
context["bridge_stm"]      # Bridge messages (if processing)
context["ltm_status"]      # "ready", "processing", or "empty"
context["is_processing"]   # Boolean
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MemoryStore                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │   Redis Keys    │         │  Thread Pool    │           │
│  ├─────────────────┤         ├─────────────────┤           │
│  │ mem:ltm:{user}  │         │ Background LTM  │           │
│  │ mem:msg:{sess}  │    ←    │ Extraction      │           │
│  │ mem:status:{u}  │         │ (Fire & Forget) │           │
│  │ mem:prev:{user} │         └─────────────────┘           │
│  └─────────────────┘                                        │
│                                                             │
│  ┌─────────────────────────────────────────────┐           │
│  │           get_user_context()                 │           │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐  │           │
│  │  │Check    │ →  │Get LTM  │ →  │Add      │  │           │
│  │  │Status   │    │         │    │Bridge?  │  │           │
│  │  └─────────┘    └─────────┘    └─────────┘  │           │
│  │                                   ~0.2ms    │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Fire-and-Forget Flow

```
end_session()          get_user_context()
     │                        │
     ▼                        ▼
┌─────────┐            ┌─────────────┐
│ Status  │            │ Check       │
│=process │            │ Status      │
└────┬────┘            └──────┬──────┘
     │                        │
     ▼                        ▼
┌─────────┐            ┌─────────────┐
│ Set prev│            │ Ready?      │──Yes──→ Return LTM
│ session │            │             │
└────┬────┘            └──────┬──────┘
     │                        │No
     ▼                        ▼
┌─────────┐            ┌─────────────┐
│ Submit  │            │ Return LTM  │
│ to pool │            │ + Bridge    │
└────┬────┘            └─────────────┘
     │
     ▼
[Background Thread]
     │
     ▼
┌─────────────┐
│ Extract LTM │
│ via LLM     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Status=ready│
│ Clear bridge│
└─────────────┘
```

## Running the Demo

```bash
# Start Redis and Milvus (Docker)
docker run -d -p 6379:6379 redis:7
docker run -d -p 19530:19530 milvusdb/milvus:latest

# Run Streamlit demo
cd memory_sdk
pip install streamlit
streamlit run examples/streamlit_demo.py --server.port 8504
```

## Requirements

- Python 3.10+
- Redis 7.0+
- OpenAI API key (or compatible LLM)
- Milvus 2.4+ (optional, for vector search)

## License

MIT
