import asyncio
import threading
import os
import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.dependencies import verify_jwt
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.api.sec_edgar import SECEdgarClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.report.builder import ReportBuilder

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title="Pharma DD API")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://pharma-insight-engine-86.lovable.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Background event loop for async pipeline ──
_BACKGROUND_LOOP = asyncio.new_event_loop()
_LOOP_THREAD = threading.Thread(target=_BACKGROUND_LOOP.run_forever, daemon=True)
_LOOP_THREAD.start()


def _run_async(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _BACKGROUND_LOOP)
    return future.result(timeout=120)


def _init_builder():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not anthropic_key or not openai_key:
        raise RuntimeError("ANTHROPIC_API_KEY and OPENAI_API_KEY must be set")
    fda_key = os.getenv("OPENFDA_API_KEY")
    sec_agent = os.getenv("SEC_USER_AGENT")
    embedder = Embedder(openai_api_key=openai_key)
    return ReportBuilder(
        ct_client=ClinicalTrialsClient(),
        fda_client=FDAClient(api_key=fda_key),
        sec_client=SECEdgarClient(user_agent=sec_agent),
        chunker_cls=Chunker,
        embedder=embedder,
        retriever=Retriever(embedder=embedder),
        generator=Generator(api_key=anthropic_key),
    )


builder = _init_builder()


# ── Models ──
class ReportRequest(BaseModel):
    company: str
    condition: Optional[str] = None
    phases: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str
    collection_id: str
    history: Optional[List[dict]] = None


# ── Endpoints ──
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/report")
@limiter.limit("10/hour")
def generate_report(request: Request, req: ReportRequest, _user=Depends(verify_jwt)):
    try:
        report = _run_async(builder.build_report(
            req.company,
            condition=req.condition,
            phases=req.phases,
        ))
    except Exception as e:
        logger.exception("Report generation failed for company=%s", req.company)
        raise HTTPException(status_code=500, detail="Report generation failed. Please try again.")
    collection_id = ReportBuilder.sanitize_collection_name(req.company)
    return {"report": report, "collection_id": collection_id}


@app.post("/chat")
def chat(req: ChatRequest, _user=Depends(verify_jwt)):
    try:
        chunks = builder.retriever.retrieve_for_chat(req.collection_id, req.message)
        response = builder.generator.generate_chat_response(
            req.message, chunks, req.history
        )
    except Exception as e:
        logger.exception("Chat failed for collection_id=%s", req.collection_id)
        raise HTTPException(status_code=500, detail="Chat failed. Please try again.")
    return {"response": response}
