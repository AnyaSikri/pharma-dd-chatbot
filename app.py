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
    page_title="Pharma DD",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Main content area ── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0 !important;
        max-width: 900px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stTextInput label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        padding: 0.6rem 0.8rem !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #64748b !important;
    }
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }

    /* ── Sidebar buttons ── */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background: rgba(255,255,255,0.06) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }

    /* ── Sidebar dividers ── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1rem 0 !important;
    }

    /* ── Sidebar success banner ── */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(34,197,94,0.12) !important;
        border: 1px solid rgba(34,197,94,0.25) !important;
        border-radius: 8px !important;
        color: #86efac !important;
    }

    /* ── Header area ── */
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ede9fe, #e0e7ff);
        color: #4f46e5;
        padding: 0.25rem 0.75rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem !important;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f8fafc;
        border-color: #e2e8f0;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1.5px solid #e2e8f0 !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ── Markdown inside chat (report) ── */
    [data-testid="stChatMessage"] h2 {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 0.5rem;
    }
    [data-testid="stChatMessage"] h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 1.2rem;
    }
    [data-testid="stChatMessage"] a {
        color: #6366f1;
        text-decoration: none;
        font-weight: 500;
    }
    [data-testid="stChatMessage"] a:hover {
        text-decoration: underline;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #94a3b8;
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .empty-state-text {
        font-size: 1rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .empty-state-hint {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="hero-badge">Powered by ClinicalTrials.gov + FDA + Claude</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Pharma Due Diligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-powered pipeline analysis, regulatory review, and risk assessment for pharmaceutical investments</div>', unsafe_allow_html=True)


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

# ── Sidebar ──
if "quick_pick" not in st.session_state:
    st.session_state.quick_pick = None

THERAPEUTIC_AREAS = [
    "All",
    "Oncology",
    "Cardiology",
    "Neurology",
    "Immunology",
    "Infectious Disease",
    "Rare Disease",
    "Metabolic / Endocrine",
    "Respiratory",
    "Ophthalmology",
    "Dermatology",
]

QUICK_PICKS = {
    "Large Cap": [
        ("Pfizer", "PFE"),
        ("Moderna", "MRNA"),
        ("Novartis", "NVS"),
        ("Regeneron", "REGN"),
        ("Gilead Sciences", "GILD"),
        ("Amgen", "AMGN"),
    ],
    "Mid Cap": [
        ("BioNTech", "BNTX"),
        ("Vertex", "VRTX"),
        ("Alnylam", "ALNY"),
        ("Sarepta", "SRPT"),
        ("Neurocrine", "NBIX"),
        ("Karuna", "KRTX"),
    ],
    "Small / Pre-Revenue": [
        ("Arcus Bio", "RCUS"),
        ("Relay Therapeutics", "RLAY"),
        ("Recursion", "RXRX"),
        ("Arvinas", "ARVN"),
        ("Day One Bio", "DAWN"),
        ("Krystal Biotech", "KRYS"),
    ],
    "Medtech": [
        ("Deepsight", "DSGT"),
        ("Intuitive Surgical", "ISRG"),
        ("Medtronic", "MDT"),
        ("Abbott Labs", "ABT"),
        ("Stryker", "SYK"),
        ("Edwards Lifesciences", "EW"),
    ],
}

with st.sidebar:
    st.markdown("### \U0001f9ec New Report")
    company_input = st.text_input(
        "Company or Drug Name",
        value=st.session_state.quick_pick or "",
        placeholder="e.g., Moderna, Keytruda, Pfizer",
        label_visibility="collapsed",
    )
    # Reset quick pick after it's been consumed
    if st.session_state.quick_pick:
        st.session_state.quick_pick = None

    # ── Filters ──
    with st.expander("Filters", expanded=False):
        therapeutic_area = st.selectbox(
            "Therapeutic Area",
            THERAPEUTIC_AREAS,
            index=0,
        )
        phase_options = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
        selected_phases = st.multiselect(
            "Trial Phase",
            phase_options,
            default=phase_options,
        )

    generate_btn = st.button("Generate Report", type="primary", use_container_width=True)

    if st.session_state.current_company:
        st.divider()
        st.success(f"Active: {st.session_state.current_company}")
        st.caption("Ask follow-up questions in the chat.")

    # ── Quick Picks ──
    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;'
        'color:#64748b !important;margin-bottom:0.3rem;">Quick Search</p>',
        unsafe_allow_html=True,
    )
    for category, companies in QUICK_PICKS.items():
        st.caption(category)
        cols = st.columns(3)
        for i, (name, ticker) in enumerate(companies):
            if cols[i % 3].button(ticker, key=f"qp_{ticker}", use_container_width=True):
                st.session_state.quick_pick = name
                st.rerun()

    st.divider()
    if st.button("Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_company = None
        st.session_state.collection_name = None
        st.rerun()

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem;color:#475569;text-align:center;margin-top:1rem;">'
        'Data: ClinicalTrials.gov &bull; openFDA &bull; FAERS<br>'
        'AI: Claude &bull; OpenAI Embeddings &bull; ChromaDB'
        '</p>',
        unsafe_allow_html=True,
    )

# Report generation
if generate_btn and company_input:
    st.session_state.messages = []
    st.session_state.current_company = company_input

    condition = therapeutic_area if therapeutic_area != "All" else None
    phases = selected_phases if len(selected_phases) < 4 else None

    with st.spinner(f"Pulling data and generating report for **{company_input}**..."):
        try:
            report = _run_async(builder.build_report(company_input, condition=condition, phases=phases))
        except Exception as e:
            report = f"Error generating report: {e}"

    st.session_state.collection_name = ReportBuilder.sanitize_collection_name(company_input)
    st.session_state.messages.append({"role": "assistant", "content": report})
    st.rerun()

# ── Empty state ──
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-state-icon">\U0001f50d</div>'
        '<div class="empty-state-text">Enter a company or drug name to get started</div>'
        '<div class="empty-state-hint">Try "Moderna", "Gilead Sciences", or "Keytruda"</div>'
        '</div>',
        unsafe_allow_html=True,
    )

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
