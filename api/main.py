import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List
from api.dependencies import verify_jwt

load_dotenv()

app = FastAPI(title="Pharma DD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReportRequest(BaseModel):
    company: str
    condition: Optional[str] = None
    phases: Optional[List[str]] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/report")
def generate_report(req: ReportRequest, _user=Depends(verify_jwt)):
    return {"report": "stub", "collection_id": "stub"}
