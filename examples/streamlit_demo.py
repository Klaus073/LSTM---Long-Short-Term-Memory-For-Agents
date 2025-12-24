"""
Streamlit Demo - Memory SDK with Fire-and-Forget LTM

Features:
1. Zero latency session end (fire-and-forget)
2. STM bridging during LTM processing
3. Real-time status indicators
4. Automatic context switching when LTM ready
"""

import streamlit as st
import json
import time
from datetime import datetime
import os
import sys

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_sdk.integrations.langgraph import MemoryStore

# Page config
st.set_page_config(
    page_title="Memory SDK Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .memory-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .status-ready { color: #4ade80; font-weight: bold; }
    .status-processing { color: #facc15; font-weight: bold; }
    .status-empty { color: #94a3b8; }
    .bridging-indicator {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize components
@st.cache_resource
def get_memory_store():
    """Initialize memory store (cached across reruns)."""
    return MemoryStore()


@st.cache_resource
def get_llm():
    """Initialize LLM for chat."""
    from openai import OpenAI
    from memory_sdk.config import MemoryConfig
    config = MemoryConfig.from_env()
    return OpenAI(api_key=config.llm.openai_api_key)


# Get resources
store = get_memory_store()
llm = get_llm()


# Session state initialization
def init_session_state():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session_state()


def end_session_and_extract(user_id: str, session_id: str):
    """
    End session with fire-and-forget extraction.
    Returns immediately - extraction happens in background!
    """
    store.end_session(user_id, session_id)
    return True


def get_chat_response(user_id: str, session_id: str, user_message: str) -> str:
    """Get AI response with memory context."""
    # Get user context (includes bridging if LTM is processing)
    context = store.get_user_context(user_id)
    
    # Build messages
    messages = []
    
    # Add system message with memory context
    system_msg = "You are a helpful assistant."
    if context.get("context_string"):
        system_msg += f"\n\n## What you know about this user:\n{context['context_string']}\n\nUse this to personalize your responses."
    
    messages.append({"role": "system", "content": system_msg})
    
    # Add conversation history
    for msg in st.session_state.messages[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    # Call LLM
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    
    return response.choices[0].message.content


# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## 🔧 Session Settings")
    
    # User ID
    new_user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Unique identifier for the user"
    )
    
    # Session ID
    new_session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Current conversation session"
    )
    
    # Check if session changed
    if new_session_id != st.session_state.session_id:
        old_session = st.session_state.session_id
        old_user = st.session_state.user_id
        
        # Fire-and-forget: End old session (returns immediately!)
        end_session_and_extract(old_user, old_session)
        st.toast("🚀 LTM extraction started in background!", icon="🧠")
        
        # Update to new session
        st.session_state.session_id = new_session_id
        st.session_state.messages = []
    
    # Update user ID
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.messages = []
    
    st.divider()
    
    # Session Actions
    st.markdown("## 🎬 Session Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 New Session", use_container_width=True):
            # Fire-and-forget: End current session
            end_session_and_extract(
                st.session_state.user_id,
                st.session_state.session_id
            )
            st.toast("🚀 LTM extraction started!", icon="🧠")
            
            # Start new session immediately (no wait!)
            st.session_state.session_id = f"session_{int(time.time())}"
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🧠 Extract Now", use_container_width=True, help="Synchronous extraction"):
            with st.spinner("Extracting..."):
                count = store.extract_from_session(
                    st.session_state.user_id,
                    st.session_state.session_id
                )
            if count > 0:
                st.success("✅ Extracted!")
            else:
                st.info("No messages to extract")
            st.rerun()
    
    if st.button("🗑️ Clear Memories", use_container_width=True):
        store.clear_memories(st.session_state.user_id)
        st.success("Memories cleared!")
        st.rerun()
    
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        store.delete_stm(st.session_state.session_id)
        st.rerun()
    
    st.divider()
    
    # LTM Status
    st.markdown("## 🧠 Long-Term Memory")
    
    context = store.get_user_context(st.session_state.user_id)
    ltm_status = context.get("ltm_status", "empty")
    is_processing = context.get("is_processing", False)
    
    # Status indicator
    if ltm_status == "ready" and context.get("ltm"):
        st.markdown("**Status:** <span class='status-ready'>✅ Ready</span>", unsafe_allow_html=True)
    elif is_processing:
        st.markdown("**Status:** <span class='status-processing'>⏳ Processing</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class="bridging-indicator">
            🌉 <strong>STM Bridge Active</strong><br>
            <small>Previous session context is being used while LTM processes</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh to check status
        if st.button("🔄 Refresh Status"):
            st.rerun()
    else:
        st.markdown("**Status:** <span class='status-empty'>📭 Empty</span>", unsafe_allow_html=True)
    
    # Display LTM content
    if context.get("ltm"):
        st.markdown("### 📦 User Memory")
        with st.expander("View LTM", expanded=True):
            st.markdown(context["ltm"])
    
    # Show bridge STM if processing
    if context.get("bridge_stm"):
        st.markdown("### 🌉 Bridge Context")
        with st.expander("Recent Messages (Bridging)", expanded=False):
            st.markdown(context["bridge_stm"])


# ===== MAIN CONTENT =====
st.markdown('<h1 class="main-header">🧠 Memory SDK Demo</h1>', unsafe_allow_html=True)

# Info bar
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"👤 **User:** {st.session_state.user_id}")
with col2:
    st.info(f"💬 **Session:** {st.session_state.session_id[:16]}...")
with col3:
    status_emoji = "✅" if ltm_status == "ready" else ("⏳" if is_processing else "📭")
    st.info(f"🧠 **LTM:** {status_emoji} {ltm_status.title()}")

st.divider()

# Display context if available
context = store.get_user_context(st.session_state.user_id)
if context.get("context_string"):
    with st.expander("🧠 Active Memory Context (Injected to AI)", expanded=False):
        st.markdown(context["context_string"])
        if is_processing:
            st.warning("⏳ LTM is processing. Showing cached LTM + bridge STM.")

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Save to STM
    store.save_message(
        st.session_state.user_id,
        st.session_state.session_id,
        "user",
        prompt
    )
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)
    
    # Get AI response
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(
                    st.session_state.user_id,
                    st.session_state.session_id,
                    prompt
                )
                st.write(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save to STM
    store.save_message(
        st.session_state.user_id,
        st.session_state.session_id,
        "assistant",
        response
    )

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    <p>💡 <strong>How Fire-and-Forget Works:</strong></p>
    <ul style="list-style: none;">
        <li>🚀 <strong>Session End:</strong> Returns immediately (0ms latency)</li>
        <li>⏳ <strong>Background:</strong> LTM extraction runs in thread pool</li>
        <li>🌉 <strong>Bridging:</strong> Previous STM used while processing</li>
        <li>✅ <strong>Ready:</strong> New LTM cached, bridge removed</li>
    </ul>
</div>
""", unsafe_allow_html=True)
