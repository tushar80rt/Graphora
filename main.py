# main.py
import asyncio
import os
import re

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")

from agent import graph

# =========================================================
# Streamlit Config
# =========================================================

st.set_page_config(page_title="Graphora", page_icon="🧠", layout="wide")
# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "target_repo" not in st.session_state:
    st.session_state.target_repo = "langchain-ai/langchain"
if "target_docs" not in st.session_state:
    st.session_state.target_docs = "https://python.langchain.com/docs/"


# ================= UI Styling =================
st.markdown(
    """
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main-header {
        font-size: 2.8rem;
        color: #00D4B1;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2.5rem;
        font-size: 1.1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #555555;
        color: #FAFAFA;
        border: none;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #777777;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #393946;
        border-radius: 8px;
        padding: 0.8rem;
    }
    .stTextInput>div>div>input:focus {
        border-color: #00D4B1;
        box-shadow: 0 0 0 2px rgba(0, 212, 177, 0.2);
    }
    .css-1d391kg, .css-1d391kg>div {
        background-color: #0E1117 !important;
        border-right: 1px solid #262730;
    }
    .css-1d391kg h1,h2,h3,h4,h5,h6,p,label {
        color: #FAFAFA !important;
    }
    .stProgress > div > div > div > div {
        background-color: #00D4B1;
    }
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 8px;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #1A1D25;
        border-radius: 0 0 8px 8px;
    }
    .card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #00D4B1;
    }
    .stRadio > div {
        background-color: #262730;
        padding: 1rem;
        border-radius: 8px;
    }
    label {
        font-weight: 600 !important;
        margin-bottom: 0.5rem;
        display: block;
        color: #CCC !important;
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .main-title img {
        height: 50px;
        margin-left: 20px;
        vertical-align: middle;
    }
    .subtitle {
        font-size: 20px;
        color: #AAAAAA;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header">
    <div class="main-title">
        Graphora With
        <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*QR3Jl4jUu326U2p2.png" alt="ScrapeGraphAI Logo">
        <span style="margin-left:40px;">&</span>
        <img src="https://console.neo4j.io/assets/logo-aura-white-DCfUnkCN.svg" alt="neo4j Logo">
        <span style="margin-left:40px;">&</span>
        <img src="https://img.icons8.com/?size=160&id=LoL4bFzqmAa0&format=png" alt="Github Logo">
    </div>
    <div class="subtitle">Autonomous Repository Intelligence</div>
    <br>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================================================
# Sidebar
# =========================================================


with st.sidebar:
    st.image("./assets/Groq.svg", width=150)
    st.markdown("---")

    groq_key = st.text_input(
        "Enter your Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )
    smartscrape_key = st.text_input(
        "Smartscrape Key", value=os.getenv("SGAI_API_KEY", ""), type="password"
    )
    langsmith_key = st.text_input(
        "🔍 LangSmith API Key (optional)",
        value=os.getenv("LANGCHAIN_API_KEY", ""),
        type="password",
        help="Enables tracing, token monitoring & latency at smith.langchain.com",
    )

    if st.button("💾 Save Keys", use_container_width=True):
        st.session_state["GROQ_API_KEY"] = groq_key
        st.session_state["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["SCRAPEGRAPH_API_KEY"] = smartscrape_key
        if langsmith_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "graphora"
            st.success("✅ Credentials saved — LangSmith tracing ON")
        else:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            st.success("✅ Credentials saved")
        if "agent" in st.session_state:
            del st.session_state["agent"]
        st.rerun()
    if groq_key or smartscrape_key:
        st.caption("Credentials loaded for active session.")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption("🔍 LangSmith tracing active")

    # =========================================================
    # SIDEBAR RAG CHUNK VIEW
    # =========================================================
    st.markdown("---")
    if st.session_state.get("last_retrieved_chunks"):
        JUNK = [
            "unfortunately",
            "i can't provide",
            "i cannot provide",
            "follow these steps",
            "by following",
            "please visit",
            "access the documentation",
            "i'm unable",
            "i do not have access",
        ]

        good_chunks = [
            c
            for c in st.session_state["last_retrieved_chunks"]
            if c.get("content")
            and not any(p in c["content"].lower() for p in JUNK)
            and len(c.get("content", "")) > 80
        ]

        if good_chunks:
            st.subheader("🔍 RAG Context")
            st.caption("Grounding context for the latest report:")
            for i, chunk in enumerate(good_chunks):
                raw_src = chunk.get("source", "")
                score = chunk.get("score", 0)
                parts = [p for p in raw_src.rstrip("/").split("/") if p]
                src = parts[-1] if parts else raw_src or "Unknown"
                with st.expander(f"Chunk {i+1} — {src} (Score: {score:.4f})"):
                    st.caption(f"Source: {raw_src}")
                    st.text_area(
                        "Content",
                        value=chunk["content"],
                        height=150,
                        key=f"rag_{i}",
                        disabled=True,
                    )
        else:
            st.info(
                "⚠️ No quality RAG chunks retrieved. ChromaDB may contain stale data — clear `./chroma_db` and retry."
            )

        st.markdown("---")
    elif "last_retrieved_chunks" in st.session_state:
        st.info(
            "ℹ️ Analyzing repository and documentation. Intelligence extraction is performed strictly on retrieved context."
        )

# =========================================================
# Display Chat History
# =========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# User Input
# =========================================================

prompt = st.chat_input("Ask about any GitHub repository...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                repo = st.session_state.get("target_repo", "langchain-ai/langchain")
                docs_url = st.session_state.get(
                    "target_docs",
                    "https://python.langchain.com/docs/"
                )

                # Extract Repository (GitHub Pattern)
                github_match = re.search(r"github\.com/([\w-]+/[\w-]+)", prompt)
                if github_match:
                    repo = github_match.group(1)

                # Default query uses the full user prompt
                clean_query = prompt.strip()

                # Extract URL (specifically for documentation, skipping github itself)
                urls = re.findall(r"(https?://[^\s,]+)", prompt)
                for u in urls:
                    if "github.com" not in u.lower():
                        docs_url = u
                        break

                # If the user only provided a github URL, keep the default docs URL
                if docs_url and "github.com" in docs_url.lower():
                    docs_url = st.session_state.get(
                        "target_docs",
                        "https://python.langchain.com/docs/"
                    )

                # Clean the query to remove the extracted repo and docs URL
                clean_query = prompt.replace(repo, "").replace(docs_url, "").strip()
                if not clean_query:
                    clean_query = "Analyze repository and documentation"

                result = loop.run_until_complete(
                    graph.ainvoke(
                        {
                            "query": clean_query,
                            "repo": repo,
                            "docs_url": docs_url,
                            "repo_data": "",
                            "architecture_data": "",
                            "docs_data": "",
                            "intelligence_data": "",
                            "graph_data": "",
                            "security_data": "",
                            "roadmap_data": "",
                            "final_report": "",
                            "retrieved_chunks": [],
                            "errors": [],
                        }
                    )
                )

                response = result["final_report"]

                st.markdown(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Store chunks in session state for sidebar access
                st.session_state["last_retrieved_chunks"] = result.get(
                    "retrieved_chunks", []
                )

                # Trigger a rerun so the sidebar updates immediately
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
