# app.py
import streamlit as st
import asyncio
import threading
import os
from dotenv import load_dotenv
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.api.sec_edgar import SECEdgarClient
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide ALL Streamlit chrome */
    #MainMenu, footer, header,
    [data-testid="stDecoration"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    .viewerBadge_container__r5tak,
    ._profileContainer_gzau3_53 {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding: 1.5rem 2rem 0 2rem !important;
        max-width: 960px;
    }
    .main {
        background: #f8f9fb;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #131c33 50%, #1a2540 100%);
        border-right: none;
        box-shadow: 4px 0 24px rgba(0,0,0,0.15);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1 !important;
    }
    /* Hide sidebar helper/instruction text */
    [data-testid="stSidebar"] .stTextInput div[data-testid="InputInstructions"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stTextInput label {
        color: #64748b !important;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: #ffffff !important;
        border: 1.5px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #0f172a !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stTextInput input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.2) !important;
    }

    /* ── Sidebar buttons ── */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
        box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="secondary"],
    [data-testid="stSidebar"] .stButton button:not([kind]) {
        background: rgba(255,255,255,0.04) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
    [data-testid="stSidebar"] .stButton button:not([kind]):hover {
        background: rgba(255,255,255,0.08) !important;
        color: #e2e8f0 !important;
    }

    /* ── Sidebar dividers ── */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 0.8rem 0 !important;
    }

    /* ── Sidebar success banner ── */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(34,197,94,0.08) !important;
        border: 1px solid rgba(34,197,94,0.2) !important;
        border-radius: 10px !important;
        color: #86efac !important;
    }

    /* ── Sidebar select/multiselect ── */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748b !important;
    }

    /* ── Header area ── */
    .hero-header {
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.1rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 0;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #6366f1;
        padding: 0.2rem 0.65rem;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        border: 1px solid #e2e8f0;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.4rem 1.6rem !important;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03), 0 4px 12px rgba(0,0,0,0.02);
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] strong {
        color: #0f172a !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        border-top: 1px solid #e5e7eb;
        padding-top: 0.8rem !important;
        background: #f8f9fb;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1.5px solid #d1d5db !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.88rem !important;
        background: #ffffff !important;
        color: #0f172a !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.12) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ── Report markdown styling ── */
    [data-testid="stChatMessage"] h2 {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a !important;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 0.5rem;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
    }
    [data-testid="stChatMessage"] h3 {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #f1f5f9;
    }
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        padding-left: 1.3rem;
    }
    [data-testid="stChatMessage"] li {
        margin-bottom: 0.35rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessage"] a {
        color: #6366f1 !important;
        text-decoration: none;
        font-weight: 500;
        border-bottom: 1px solid transparent;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatMessage"] a:hover {
        border-bottom-color: #6366f1;
    }
    [data-testid="stChatMessage"] code {
        background: #f1f5f9;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.82rem;
    }
    [data-testid="stChatMessage"] strong {
        font-weight: 700;
    }
    [data-testid="stChatMessage"] hr {
        border-color: #e5e7eb;
        margin: 1.2rem 0;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 1.2rem;
        opacity: 0.4;
    }
    .empty-state-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .empty-state-hint {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""<div class="hero-header">
    <div class="hero-badge">ClinicalTrials.gov + FDA + SEC EDGAR + Claude</div>
    <div class="hero-title">Pharma Due Diligence</div>
    <div class="hero-subtitle">AI-powered pipeline analysis, regulatory review, and financial assessment for healthcare investments</div>
</div>""", unsafe_allow_html=True)


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
    sec_client = SECEdgarClient(user_agent=os.getenv("SEC_USER_AGENT"))
    embedder = Embedder(openai_api_key=openai_key)
    retriever = Retriever(embedder=embedder)
    generator = Generator(api_key=anthropic_key)
    builder = ReportBuilder(
        ct_client=ct_client,
        fda_client=fda_client,
        sec_client=sec_client,
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
if "quick_pick_generate" not in st.session_state:
    st.session_state.quick_pick_generate = None
if "report_count" not in st.session_state:
    st.session_state.report_count = 0

MAX_REPORTS_PER_SESSION = 10

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
    if st.session_state.quick_pick:
        st.session_state.company_input_field = st.session_state.quick_pick
        st.session_state.quick_pick = None
    company_input = st.text_input(
        "Company or Drug Name",
        key="company_input_field",
        placeholder="e.g., Moderna, Keytruda, Pfizer",
        label_visibility="collapsed",
    )

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
                st.session_state.quick_pick_generate = name
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
        'Data: ClinicalTrials.gov &bull; openFDA &bull; SEC EDGAR &bull; Yahoo Finance<br>'
        'AI: Claude &bull; OpenAI Embeddings &bull; ChromaDB'
        '</p>',
        unsafe_allow_html=True,
    )

# Report generation — from button or quick pick
report_target = None
if generate_btn and company_input:
    report_target = company_input
elif st.session_state.quick_pick_generate:
    report_target = st.session_state.quick_pick_generate
    st.session_state.quick_pick_generate = None

if report_target:
    if st.session_state.report_count >= MAX_REPORTS_PER_SESSION:
        st.error(f"You've reached the limit of {MAX_REPORTS_PER_SESSION} reports per session. Please refresh the page to start a new session.")
    else:
        st.session_state.messages = []
        st.session_state.current_company = report_target

        condition = therapeutic_area if therapeutic_area != "All" else None
        phases = selected_phases if len(selected_phases) < 4 else None

        with st.spinner(f"Pulling data and generating report for **{report_target}**..."):
            try:
                report = _run_async(builder.build_report(report_target, condition=condition, phases=phases))
            except Exception as e:
                report = f"Error generating report: {e}"

        st.session_state.report_count += 1
        st.session_state.collection_name = ReportBuilder.sanitize_collection_name(report_target)
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
