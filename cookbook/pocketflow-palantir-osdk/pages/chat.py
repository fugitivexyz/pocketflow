"""Streamlit Chat UI for Palantir OSDK Bot.

Features:
- Chat interface with conversation history
- REAL-TIME streaming of agent thinking steps
- Full transparency: shows all LLM prompts and responses as they happen
- Data display with interactive tables
- Plotly chart visualization
- Multi-agent architecture (Coordinator-Planner-Executor)
- Conversation memory across queries

This is the main chat page of the multi-page app.
"""

import streamlit as st
import pandas as pd
import copy
import threading

# Import from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flow import run_query as run_multi_agent_query
from utils.osdk_client import get_osdk_client
from utils.streaming import StreamingCallback, run_flow_with_streaming
from utils.config import load_config, Config


# =============================================================================
# Custom Styling
# =============================================================================

# Custom CSS for visual consistency and accessibility
st.markdown("""
<style>
    /* Sidebar section spacing */
    [data-testid="stSidebar"] .stSubheader {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Thinking step styling */
    .thinking-step {
        border-left: 3px solid #0d6efd;
        padding-left: 12px;
        margin: 8px 0;
    }
    
    /* Status container spacing */
    [data-testid="stStatusWidget"] {
        margin-bottom: 1rem;
    }
    
    /* Expander consistency */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    
    /* Improve focus visibility for accessibility */
    button:focus, input:focus, [tabindex]:focus {
        outline: 2px solid #0d6efd;
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization (Consolidated)
# =============================================================================

def init_session_state():
    """Initialize all session state variables with defaults."""
    # Load config if not already loaded
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    
    defaults = {
        "messages": [],
        "conversation_history": [],
        "is_generating": False,
        "stop_requested": False,  # For proper cancellation
        "pending_process_query": None,  # For two-phase query processing
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()


# =============================================================================
# Config-aware helpers
# =============================================================================

def get_config() -> Config:
    """Get current config from session state."""
    return st.session_state.config


# =============================================================================
# Sidebar Configuration
# =============================================================================

# Cache OSDK client to prevent recreation on every rerun
@st.cache_resource
def get_cached_osdk_client():
    """Get cached OSDK client instance."""
    return get_osdk_client()


with st.sidebar:
    
    
    # Quick transparency toggle (synced with config)
    config = get_config()
    st.subheader("👁️ Transparency")
    show_thinking = st.checkbox(
        "Show agent thinking process",
        value=config.transparency.show_thinking_steps,
        help="Display all LLM prompts and responses as they happen",
    )
    # Sync back to config
    if show_thinking != config.transparency.show_thinking_steps:
        config.transparency.show_thinking_steps = show_thinking
    
    st.markdown("---")
    
    # Available object types (with better error handling)
    st.subheader("📦 Available Object Types")
    st.caption("Data schemas available for querying")
    try:
        client = get_cached_osdk_client()
        object_types = client.list_object_types()
        if not object_types:
            st.info("No object types available.")
        else:
            for obj_type in object_types:
                with st.expander(obj_type):
                    schema = client.get_object_schema(obj_type)
                    st.caption(schema.get("description", "No description"))
                    st.json(schema.get("properties", {}))
    except Exception as e:
        st.error(f"Error loading object types: {e}")
        if st.button("🔄 Retry Loading", key="retry_schema"):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation", use_container_width=True, help="Remove all messages and start fresh"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    # Stop generation button with proper cancellation
    if st.session_state.is_generating:
        if st.button("🛑 Stop Generation", use_container_width=True, type="primary", help="Cancel the current query"):
            st.session_state.stop_requested = True
            st.session_state.is_generating = False
            
            # Signal backend cancellation
            if "current_stop_event" in st.session_state:
                st.session_state.current_stop_event.set()
            
            st.warning("⚠️ Generation stopped. The background process may take a moment to halt.")
            st.rerun()
    
    st.markdown("---")
    
    # Example queries (with accessible label)
    st.subheader("💡 Example Queries")
    st.caption("Click to try a sample question")
    examples = [
        "Show me all completed experiments",
        "What surfactants are available?",
        "Compare turbidity across different surfactants",
        "Show viscosity results for Polysorbate 80 samples",
        "Create a chart of SEC monomer values by time point",
    ]
    for example in examples:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()


# =============================================================================
# Main Chat Interface
# =============================================================================

st.title("🔮 Palantir OSDK Assistant")

# Show current config summary
config = get_config()
mode_emoji = {"auto": "🤖", "simple": "⚡", "complex": "🔬"}.get(config.agent.complexity_mode, "🤖")
st.caption(f"Model: {config.model.model} | Mode: {mode_emoji} {config.agent.complexity_mode.title()} | {'Mock' if config.connection.use_mock_data else 'Live'} Data")


# =============================================================================
# Helper Function: Run Flow with Streaming UI
# =============================================================================

def run_with_streaming_ui(query: str, conversation_history: list, status_container):
    """
    Run the agent flow with real-time streaming of thinking steps.
    
    Args:
        query: User's question
        conversation_history: Previous conversation for context
        status_container: Streamlit status container for live updates
        
    Returns:
        dict with final_answer, fetched_data, figure, thinking_steps
    """
    config = get_config()
    
    # Reset stop flag at start
    st.session_state.stop_requested = False
    
    # Create stop event for backend cancellation
    stop_event = threading.Event()
    st.session_state.current_stop_event = stop_event
    
    # Create streaming callback
    callback = StreamingCallback()
    
    # Use multi-agent flow runner with config
    flow_runner = run_multi_agent_query
    
    # Start the flow in a background thread
    thread = run_flow_with_streaming(
        flow_runner,
        callback,
        query,
        conversation_history,
        config,  # Pass config to flow
        stop_event=stop_event  # Pass stop event
    )
    
    # Track displayed steps
    displayed_steps = []
    step_count = 0
    
    # Stream thinking steps as they arrive
    if config.transparency.show_thinking_steps:
        status_container.write("🧠 **Agent Thinking (Live)**")
    
    for step in callback.iter_steps(timeout=0.1):
        # Check for stop request
        if st.session_state.stop_requested:
            displayed_steps.append({
                "type": "⚠️ Cancelled",
                "content": "Query processing was stopped by user."
            })
            break
            
        step_count += 1
        displayed_steps.append({
            "type": step.step_type,
            "content": step.content
        })
        
        if config.transparency.show_thinking_steps:
            # Show step in status container with accessible markup
            status_container.markdown(f"**Step {step_count}: {step.step_type}**")
            # Truncate very long content for live display
            content = step.content
            max_length = 1500 if config.transparency.show_raw_prompts else 500
            if len(content) > max_length:
                content = content[:max_length] + "\n\n... (truncated for live view)"
            status_container.code(content, language="yaml")
            status_container.divider()
    
    # Wait for thread to complete with timeout from config
    timeout_seconds = config.agent.query_timeout
    
    # Only wait if not cancelled
    if not st.session_state.stop_requested:
        thread.join(timeout=timeout_seconds)
    
    # Check if thread is still running after timeout
    if thread.is_alive() and not st.session_state.stop_requested:
        st.warning(f"⏱️ Query processing timed out after {timeout_seconds} seconds. The background process is still running.")
        return {
            "final_answer": "The query took too long to process. Please try a simpler question or try again later.",
            "fetched_data": None,
            "figure": None,
            "thinking_steps": displayed_steps,
        }
    
    # Handle stop request
    if st.session_state.stop_requested:
        return {
            "final_answer": "Query processing was stopped by user.",
            "fetched_data": None,
            "figure": None,
            "thinking_steps": displayed_steps,
        }
    
    # Get the result
    try:
        result = callback.get_result()
        # Ensure thinking_steps from result is used if available
        if not displayed_steps and result.get("thinking_steps"):
            displayed_steps = result["thinking_steps"]
        result["thinking_steps"] = displayed_steps
    except Exception as e:
        st.error(f"Error during execution: {e}")
        result = {
            "final_answer": f"An error occurred: {e}",
            "fetched_data": None,
            "figure": None,
            "thinking_steps": displayed_steps,
        }
    
    return result


# =============================================================================
# Shared functions for query processing and display
# =============================================================================

def display_assistant_response(result: dict):
    """Display assistant response with data and charts."""
    # Add spacing after status
    st.markdown("")  
    
    # Display answer
    st.markdown(result.get("final_answer", "No response generated."))
    
    # Display data if present
    if result.get("fetched_data") is not None:
        try:
            if not result["fetched_data"].empty:
                with st.expander("📊 View Data (Table)", expanded=True):
                    st.dataframe(result["fetched_data"], use_container_width=True)
        except AttributeError:
            pass  # Not a DataFrame
    
    # Display figure if present
    if result.get("figure") is not None:
        st.plotly_chart(result["figure"], use_container_width=True)


def store_assistant_message(result: dict, query: str):
    """Store assistant message and update conversation history."""
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("final_answer", ""),
        "data": result.get("fetched_data"),
        "figure": result.get("figure"),
        "thinking_steps": result.get("thinking_steps", []),
    })
    
    # Update conversation history for context
    st.session_state.conversation_history.append({"role": "user", "content": query})
    st.session_state.conversation_history.append({"role": "assistant", "content": result.get("final_answer", "")})


def process_user_query(query: str):
    """
    Process a user query with streaming UI (shared logic for sidebar examples and chat input).
    Uses two-phase pattern: first rerun to show Stop button, then process query.
    
    Args:
        query: The user's question
    """
    # Validate input
    if not query or not query.strip():
        st.warning("Please enter a question.")
        return
    
    query = query.strip()
    if len(query) > 2000:
        st.warning("Query is too long. Please limit to 2000 characters.")
        return
    
    # Phase 1: Set state and trigger rerun to show Stop button
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.is_generating = True
    st.session_state.pending_process_query = query
    st.rerun()


def execute_pending_query():
    """
    Phase 2: Execute the query after rerun (Stop button is now visible).
    """
    query = st.session_state.pending_process_query
    st.session_state.pending_process_query = None  # Clear pending
    
    with st.chat_message("assistant"):
        with st.status("🤔 Thinking...", expanded=True) as status:
            result = run_with_streaming_ui(
                query,
                st.session_state.conversation_history,
                status
            )
            status.update(label="✅ Complete", state="complete")
        
        # Clear generating state
        st.session_state.is_generating = False
        
        # Display response using shared function
        display_assistant_response(result)
        
        # Store message using shared function
        store_assistant_message(result, query)
        
        # Rerun to update UI state (remove Stop button)
        st.rerun()


# =============================================================================
# Display chat history
# =============================================================================

config = get_config()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display data if present (with accessible label)
        if "data" in message and message["data"] is not None:
            try:
                if not message["data"].empty:
                    with st.expander("📊 View Data (Table)", expanded=False):
                        st.dataframe(message["data"], use_container_width=True)
            except AttributeError:
                pass  # Not a DataFrame
        
        # Display figure if present
        if "figure" in message and message["figure"] is not None:
            st.plotly_chart(message["figure"], use_container_width=True)
        
        # Display thinking steps if enabled and present (with accessible label)
        if config.transparency.show_thinking_steps and "thinking_steps" in message and message["thinking_steps"]:
            with st.expander("🧠 Agent Thinking (Processing Steps)", expanded=False):
                for idx, step in enumerate(message["thinking_steps"], 1):
                    st.markdown(f"**Step {idx}: {step['type']}**")
                    content = step["content"]
                    # Truncate content if show_raw_prompts is disabled
                    if not config.transparency.show_raw_prompts and len(content) > 500:
                        content = content[:500] + "\n\n... (enable 'Show raw prompts' in config.yaml for full content)"
                    st.code(content, language="markdown")
                    if idx < len(message["thinking_steps"]):
                        st.divider()


# =============================================================================
# Query Processing Logic
# =============================================================================

# Phase 2: Execute pending query (after rerun, Stop button is now visible)
if st.session_state.pending_process_query is not None:
    execute_pending_query()

# Handle pending query from sidebar examples (uses shared function)
elif "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
    process_user_query(query)

# Chat input (uses shared function)
elif query := st.chat_input("Ask about your Palantir data...", key="main_chat_input"):
    process_user_query(query)
