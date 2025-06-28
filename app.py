__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import random
from faq_bot import JupiterFAQBot
from llm_only_bot import LLMOnlyFAQBot

# --- Page Configuration ---
st.set_page_config(
    page_title="Jupiter FAQ Bot",
    layout="wide",
)

# --- Custom CSS for Styling ---
def load_css():
    st.markdown("""
    <style>
        /* Hide default streamlit branding */
        header, footer { visibility: hidden; }
        
        /* Main page layout */
        .main .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Style the right FAQ column */
        div[data-testid="column"]:nth-of-type(2) {
            background-color: #1c1d26;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #333;
        }

        /* Style the suggestion buttons in the right column */
        div[data-testid="column"]:nth-of-type(2) .stButton button {
            background-color: #262730;
            border: 1px solid #444;
            width: 100%;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            white-space: normal; /* Allow text to wrap */
            height: auto;
            padding: 0.75rem;
        }
        div[data-testid="column"]:nth-of-type(2) .stButton button:hover {
            border-color: #888;
            color: #fff;
        }
        
        /* Main content titles */
        h1 { font-size: 2.5rem; font-weight: bold; }
        h3 { font-size: 1.25rem; font-weight: 600; color: #e0e0e0; }

    </style>
    """, unsafe_allow_html=True)

load_css()


# --- Data Loading ---
@st.cache_data
def load_faq_data(file_path="jupiter_faqs_processed.csv"):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

faq_df = load_faq_data()
if not faq_df.empty:
    categories = ["Select a category..."] + sorted(faq_df['category'].unique().tolist())


# --- Session State Initialization ---
if "bot" not in st.session_state or "bot_mode" not in st.session_state:
    st.session_state.bot = JupiterFAQBot()
    st.session_state.bot_mode = "Retrieval-Augmented (RAG)"
for key in ["query", "response", "suggestions"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "suggestions" else []
if "triggered_by_suggestion" not in st.session_state:
    st.session_state.triggered_by_suggestion = False
if "current_category" not in st.session_state:
    st.session_state.current_category = None
if "random_suggestions" not in st.session_state:
    st.session_state.random_suggestions = []


# --- Main Page Layout (Two Columns) ---
col_chat, col_faq = st.columns([2, 1.2], gap="large")


# --- Left Column: Chat Interface ---
with col_chat:
    st.title("📚 Jupiter FAQ Bot")

    mode = st.radio(
        "Select Bot Mode:",
        ["Retrieval-Augmented (RAG)", "LLM-Only"],
        horizontal=True
    )

    if st.session_state.get("bot_mode") != mode:
        if mode == "Retrieval-Augmented (RAG)": st.session_state.bot = JupiterFAQBot()
        else: st.session_state.bot = LLMOnlyFAQBot()
        st.session_state.bot_mode = mode
        st.session_state.response, st.session_state.suggestions = "", []

    user_input = st.text_input(
        "Ask your question:",
        value=st.session_state.query,
        placeholder="e.g., How do I open a savings account?"
    )

    if st.button("Get Answer") or st.session_state.triggered_by_suggestion:
        if user_input:
            with st.spinner("Finding an answer..."):
                result = st.session_state.bot.get_response(user_input)
                st.session_state.query = user_input
                st.session_state.response = result["response"]
                st.session_state.suggestions = result.get("suggestions", [])
                st.session_state.triggered_by_suggestion = False
        else:
            st.warning("Please ask a question.")

    if st.session_state.response:
        st.markdown("---")
        st.subheader("Answer")
        st.write(st.session_state.response)

    if st.session_state.suggestions and mode == "Retrieval-Augmented (RAG)":
        st.subheader("Related Questions")
        for suggestion in st.session_state.suggestions:
            if st.button(suggestion, key=f"rag_{suggestion}"):
                st.session_state.query = suggestion
                st.session_state.triggered_by_suggestion = True
                st.rerun()


# --- Right Column: FAQ Browser ---
with col_faq:
    st.markdown("<h3><span style='font-size: 1.5em;'>🔎</span> Browse FAQs</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:0; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
    
    if not faq_df.empty:
        selected_category = st.selectbox(
            "Choose a category for ideas:",
            options=categories
        )

        if selected_category != st.session_state.current_category:
            st.session_state.current_category = selected_category
            if selected_category and selected_category != "Select a category...":
                filtered_faqs = faq_df[faq_df['category'] == selected_category]
                num_samples = min(3, len(filtered_faqs))
                st.session_state.random_suggestions = random.sample(filtered_faqs['question'].tolist(), num_samples)
            else:
                st.session_state.random_suggestions = []

        if st.session_state.random_suggestions:
            st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
            st.write("**Try asking one of these:**")
            for suggestion in st.session_state.random_suggestions:
                if st.button(suggestion, key=f"cat_{suggestion}"):
                    st.session_state.query = suggestion
                    st.session_state.triggered_by_suggestion = True
                    st.rerun()
