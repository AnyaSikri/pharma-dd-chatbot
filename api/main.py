import asyncio
import threading
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

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

app = FastAPI(title="Pharma DD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "placeholder")
    openai_key = os.getenv("OPENAI_API_KEY", "placeholder")
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
def generate_report(req: ReportRequest, _user=Depends(verify_jwt)):
    try:
        report = _run_async(builder.build_report(
            req.company,
            condition=req.condition,
            phases=req.phases,
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
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
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
    return {"response": response}
