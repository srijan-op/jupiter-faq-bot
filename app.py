__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from faq_bot import JupiterFAQBot
from llm_only_bot import LLMOnlyFAQBot

# Set up page
st.set_page_config(page_title="Jupiter FAQ Bot", layout="centered")
st.title("📚 Jupiter FAQ Bot")

# Bot mode toggle
mode = st.radio("Select Bot Mode:", ["Retrieval-Augmented (RAG)", "LLM-Only"])

# Switch bot if mode changes
if "bot" not in st.session_state or st.session_state.get("bot_mode") != mode:
    if mode == "Retrieval-Augmented (RAG)":
        st.session_state.bot = JupiterFAQBot()
    else:
        st.session_state.bot = LLMOnlyFAQBot()
    st.session_state.bot_mode = mode
    st.session_state.response = ""
    st.session_state.query = ""
    st.session_state.suggestions = []
    st.session_state.triggered_by_suggestion = False

# Input state setup
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = ""
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "triggered_by_suggestion" not in st.session_state:
    st.session_state.triggered_by_suggestion = False

# Input box
user_input = st.text_input("Ask your question:", value=st.session_state.query)

# Submit button
if st.button("Get Answer") or st.session_state.triggered_by_suggestion:
    result = st.session_state.bot.get_response(user_input)
    st.session_state.query = user_input
    st.session_state.response = result["response"]
    st.session_state.suggestions = result.get("suggestions", [])
    st.session_state.triggered_by_suggestion = False  # Reset after handling

# Show response
if st.session_state.response:
    st.subheader("Answer")
    st.write(st.session_state.response)

# Show suggestions (RAG only)
if st.session_state.suggestions and mode == "Retrieval-Augmented (RAG)":
    st.subheader("Related Questions")
    for suggestion in st.session_state.suggestions:
        if st.button(suggestion):
            st.session_state.query = suggestion
            st.session_state.triggered_by_suggestion = True
            st.rerun()
