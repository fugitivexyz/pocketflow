"""Streamlit Multi-Page App Entry Point for Palantir OSDK Bot.

This is the main entry point that sets up navigation between:
- 💬 Chat: Main chat interface for querying data
- ⚙️ Settings: Configuration page for all multi-agent settings

Run with:
    streamlit run app.py
"""

import streamlit as st

# =============================================================================
# Page Configuration (must be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="Palantir OSDK Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Initialize Config on App Start
# =============================================================================

from utils.config import load_config

if "config" not in st.session_state:
    st.session_state.config = load_config()


# =============================================================================
# Multi-Page Navigation
# =============================================================================

# Define pages
chat_page = st.Page("pages/chat.py", title="Chat", icon="💬", default=True)

# Create navigation
pg = st.navigation([chat_page])

# Run the selected page
pg.run()
