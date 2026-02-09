# app.py
import streamlit as st
import asyncio
import threading
import os
from dotenv import load_dotenv
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.report.builder import ReportBuilder

load_dotenv()

st.set_page_config(
    page_title="Pharma Due Diligence",
    page_icon="\U0001f48a",
    layout="wide",
)

st.title("Pharma Due Diligence Chatbot")
st.caption("RAG-powered analysis using ClinicalTrials.gov and FDA data")


def _run_async(coro):
    """Run an async coroutine from synchronous Streamlit context."""
    result = [None]
    exception = [None]

    def target():
        loop = asyncio.new_event_loop()
        try:
            result[0] = loop.run_until_complete(coro)
        except Exception as e:
            exception[0] = e
        finally:
            loop.close()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if exception[0]:
        raise exception[0]
    return result[0]


@st.cache_resource
def get_components():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    fda_key = os.getenv("OPENFDA_API_KEY")

    if not anthropic_key or not openai_key:
        st.error("Please set ANTHROPIC_API_KEY and OPENAI_API_KEY in your .env file")
        st.stop()

    ct_client = ClinicalTrialsClient()
    fda_client = FDAClient(api_key=fda_key)
    embedder = Embedder(openai_api_key=openai_key)
    retriever = Retriever(embedder=embedder)
    generator = Generator(api_key=anthropic_key)
    builder = ReportBuilder(
        ct_client=ct_client,
        fda_client=fda_client,
        chunker_cls=Chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    return builder, retriever, generator


builder, retriever, generator = get_components()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Sidebar - Report Generation
with st.sidebar:
    st.header("Generate Report")
    company_input = st.text_input(
        "Company or Drug Name",
        placeholder="e.g., Moderna, Keytruda, Pfizer",
    )
    generate_btn = st.button("Generate Due Diligence Report", type="primary", use_container_width=True)

    if st.session_state.current_company:
        st.divider()
        st.success(f"Active: {st.session_state.current_company}")
        st.caption("Ask follow-up questions in the chat below.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_company = None
        st.session_state.collection_name = None
        st.rerun()

# Report generation
if generate_btn and company_input:
    st.session_state.messages = []
    st.session_state.current_company = company_input

    with st.spinner(f"Fetching data and generating report for {company_input}..."):
        try:
            report = _run_async(builder.build_report(company_input))
        except Exception as e:
            report = f"Error generating report: {e}"

    st.session_state.collection_name = ReportBuilder.sanitize_collection_name(company_input)

    st.session_state.messages.append({"role": "assistant", "content": report})
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.collection_name:
            chunks = retriever.retrieve_for_chat(st.session_state.collection_name, prompt)
        else:
            chunks = []

        history = [
            msg for msg in st.session_state.messages[:-1]
            if msg["role"] in ("user", "assistant")
        ]

        with st.spinner("Thinking..."):
            response = generator.generate_chat_response(prompt, chunks, history)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
