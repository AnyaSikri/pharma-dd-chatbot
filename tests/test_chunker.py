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


def test_chunk_device_clearance():
    clearance = {
        "k_number": "K223456",
        "applicant": "DeepSight Medical Inc.",
        "device_name": "DeepSight AI Imaging System",
        "product_code": "QBS",
        "clearance_type": "Traditional",
        "decision_date": "20230615",
        "decision_description": "Substantially Equivalent",
        "advisory_committee_description": "Radiology",
    }
    chunks = Chunker.chunk_device_clearance(clearance)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "K223456" in chunk["text"]
    assert "DeepSight AI Imaging System" in chunk["text"]
    assert chunk["metadata"]["source"] == "fda_device_clearance"
    assert chunk["metadata"]["device_name"] == "DeepSight AI Imaging System"
    assert "cfpmn" in chunk["metadata"]["source_url"]


def test_chunk_device_recalls():
    recalls = [
        {
            "res_event_number": "Z-1234-2024",
            "recalling_firm": "DeepSight Medical Inc.",
            "product_description": "DeepSight AI Imaging System",
            "reason_for_recall": "Software error",
            "status": "Terminated",
        }
    ]
    chunks = Chunker.chunk_device_recalls("DeepSight", recalls)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "Software error" in chunk["text"]
    assert chunk["metadata"]["source"] == "fda_device_recall"


def test_chunk_device_adverse_events():
    ae_summary = {
        "total_reports": 120,
        "serious_count": 5,
        "sample_events": ["Patient experienced irritation"],
    }
    chunks = Chunker.chunk_device_adverse_events("DeepSight AI", ae_summary)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "120" in chunk["text"]
    assert chunk["metadata"]["source"] == "fda_device_events"
    assert chunk["metadata"]["device_name"] == "DeepSight AI"


def test_chunk_sec_filings():
    filings = [
        {
            "form_type": "10-K",
            "filing_date": "2024-02-22",
            "accession_number": "0001682852-24-000010",
            "primary_document": "mrna-20231231.htm",
            "description": "10-K Annual Report",
            "filing_url": "https://www.sec.gov/Archives/edgar/data/1682852/mrna-20231231.htm",
            "company_name": "Moderna, Inc.",
        }
    ]
    chunks = Chunker.chunk_sec_filings("Moderna, Inc.", filings)
    assert len(chunks) == 1
    assert "10-K" in chunks[0]["text"]
    assert "2024-02-22" in chunks[0]["text"]
    assert chunks[0]["metadata"]["source"] == "sec_filings"
    assert chunks[0]["metadata"]["company"] == "Moderna, Inc."


def test_chunk_sec_filings_empty():
    chunks = Chunker.chunk_sec_filings("Test Corp", [])
    assert chunks == []


def test_chunk_company_financials_with_market_data():
    facts = {
        "company_name": "Moderna, Inc.",
        "revenue": {"value": 6671000000, "period_end": "2023-12-31", "form": "10-K"},
    }
    market = {
        "ticker": "MRNA",
        "current_price": 95.50,
        "market_cap": 36500000000,
        "trailing_pe": None,
        "sector": "Healthcare",
        "industry": "Biotechnology",
    }
    chunks = Chunker.chunk_company_financials("Moderna, Inc.", facts, market)
    assert len(chunks) == 2
    sources = [c["metadata"]["source"] for c in chunks]
    assert "sec_financials" in sources
    assert "market_data" in sources
    assert any("MRNA" in c["text"] for c in chunks)


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


# ── Edge case tests ──


def test_chunk_clinical_trial_missing_fields():
    """Trial with minimal fields should not crash."""
    trial = {"nct_id": "NCT99999999", "title": "Minimal Trial"}
    chunks = Chunker.chunk_clinical_trial(trial)
    assert len(chunks) == 1
    assert "NCT99999999" in chunks[0]["text"]
    assert chunks[0]["metadata"]["company"] == "Unknown"
    assert chunks[0]["metadata"]["phase"] == "N/A"


def test_chunk_clinical_trial_empty_dict():
    """Completely empty trial dict should not crash."""
    chunks = Chunker.chunk_clinical_trial({})
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["nct_id"] == ""


def test_chunk_fda_approval_missing_fields():
    """Approval with missing keys should not crash."""
    approval = {"manufacturer": "TestCo"}
    chunks = Chunker.chunk_fda_approval(approval)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "fda_approval"
    assert chunks[0]["metadata"]["drug_name"] == "Unknown"


def test_chunk_adverse_events_none_reactions():
    """Adverse events with sample_reactions=None should not crash."""
    ae_summary = {"total_reports": 100, "sample_reactions": None}
    chunks = Chunker.chunk_adverse_events("TestDrug", ae_summary)
    assert len(chunks) == 1
    assert "100" in chunks[0]["text"]


def test_chunk_adverse_events_missing_total():
    """Adverse events with missing total_reports should not crash."""
    ae_summary = {}
    chunks = Chunker.chunk_adverse_events("TestDrug", ae_summary)
    assert len(chunks) == 1
    assert "0" in chunks[0]["text"]


def test_chunk_device_clearance_missing_device_name():
    """Device clearance without device_name should not crash."""
    clearance = {"k_number": "K999999"}
    chunks = Chunker.chunk_device_clearance(clearance)
    assert len(chunks) == 1
    assert "Unknown" in chunks[0]["text"]


def test_chunk_company_financials_string_value():
    """Financial data with string instead of number should not crash."""
    facts = {
        "company_name": "Test Corp",
        "revenue": {"value": "not_a_number", "period_end": "2024-01-01"},
    }
    market = {"ticker": "TEST", "current_price": "bad_value"}
    chunks = Chunker.chunk_company_financials("Test Corp", facts, market)
    assert len(chunks) >= 1


def test_chunk_company_financials_no_data():
    """Empty facts and no market data should return no chunks."""
    chunks = Chunker.chunk_company_financials("Empty Corp", {}, None)
    assert chunks == []
