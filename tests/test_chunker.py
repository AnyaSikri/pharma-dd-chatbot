import pytest
from src.ingestion.chunker import Chunker


def test_chunk_clinical_trial():
    trial = {
        "nct_id": "NCT12345678",
        "title": "A Phase 3 Study of Drug X",
        "status": "RECRUITING",
        "phase": "Phase 3",
        "enrollment": 500,
        "start_date": "2024-01-15",
        "primary_completion_date": "2026-06-01",
        "sponsor": "TestPharma Inc",
        "conditions": ["Non-Small Cell Lung Cancer"],
        "interventions": [{"name": "Drug X", "type": "DRUG"}],
        "primary_outcomes": [{"measure": "Overall Survival", "timeFrame": "24 months"}],
        "brief_summary": "This study evaluates Drug X.",
        "has_results": False,
    }
    chunks = Chunker.chunk_clinical_trial(trial)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert "NCT12345678" in chunk["text"]
    assert "Phase 3" in chunk["text"]
    assert "RECRUITING" in chunk["text"]
    assert chunk["metadata"]["source"] == "clinicaltrials"
    assert chunk["metadata"]["nct_id"] == "NCT12345678"
    assert chunk["metadata"]["company"] == "TestPharma Inc"
    assert chunk["metadata"]["phase"] == "Phase 3"


def test_chunk_fda_approval():
    approval = {
        "application_number": "NDA021821",
        "brand_name": "TYGACIL",
        "generic_name": "TIGECYCLINE",
        "manufacturer": "Pfizer Inc.",
        "products": [
            {"brand_name": "TYGACIL", "active_ingredients": [{"name": "TIGECYCLINE", "strength": "50MG"}], "dosage_form": "POWDER", "route": "INTRAVENOUS"}
        ],
        "submissions": [
            {"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "20050617", "submission_class_code_description": "New Molecular Entity"}
        ],
    }
    chunks = Chunker.chunk_fda_approval(approval)
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert "TYGACIL" in chunk["text"]
    assert "NDA021821" in chunk["text"]
    assert chunk["metadata"]["source"] == "fda_approval"
    assert chunk["metadata"]["drug_name"] == "TYGACIL"


def test_chunk_fda_label():
    label = {
        "brand_name": "TYGACIL",
        "generic_name": "TIGECYCLINE",
        "manufacturer": "Pfizer Inc.",
        "indications": "TYGACIL is indicated for complicated skin infections...",
        "boxed_warning": "WARNING: ALL-CAUSE MORTALITY...",
        "warnings": "Anaphylaxis reported...",
        "adverse_reactions": "Nausea and vomiting are common...",
    }
    chunks = Chunker.chunk_fda_label(label)
    assert len(chunks) >= 1
    texts = [c["text"] for c in chunks]
    all_text = " ".join(texts)
    assert "TYGACIL" in all_text
    assert any(c["metadata"]["source"] == "fda_label" for c in chunks)


def test_chunk_adverse_events():
    ae_summary = {
        "total_reports": 5000,
        "sample_reactions": ["Nausea", "Headache", "Vomiting"],
        "serious_count": 3,
    }
    chunks = Chunker.chunk_adverse_events("TYGACIL", ae_summary)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "5000" in chunk["text"] or "5,000" in chunk["text"]
    assert chunk["metadata"]["source"] == "fda_adverse_events"
    assert chunk["metadata"]["drug_name"] == "TYGACIL"
