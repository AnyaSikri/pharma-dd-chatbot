# tests/test_builder.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.report.builder import ReportBuilder


@pytest.fixture
def mock_deps():
    ct_client = AsyncMock()
    fda_client = AsyncMock()
    chunker_cls = MagicMock()
    embedder = MagicMock()
    retriever = MagicMock()
    generator = MagicMock()

    ct_client.search_by_sponsor.return_value = [
        {"nct_id": "NCT123", "title": "Test Trial", "sponsor": "TestPharma", "phase": "Phase 3",
         "status": "RECRUITING", "enrollment": 100, "conditions": ["Cancer"],
         "interventions": [{"name": "DrugX", "type": "DRUG"}], "primary_outcomes": [],
         "brief_summary": "A test trial", "start_date": "2024-01-01",
         "primary_completion_date": "2026-01-01", "has_results": False}
    ]
    ct_client.search_by_drug.return_value = []
    fda_client.search_approvals.return_value = [
        {"application_number": "NDA123", "brand_name": "DrugX", "generic_name": "drugx",
         "manufacturer": "TestPharma", "products": [], "submissions": []}
    ]
    fda_client.search_labels.return_value = [
        {"brand_name": "DrugX", "generic_name": "drugx", "manufacturer": "TestPharma",
         "indications": "For cancer", "boxed_warning": "", "warnings": "", "adverse_reactions": "Nausea"}
    ]
    fda_client.get_adverse_events_summary.return_value = {
        "total_reports": 100, "sample_reactions": ["Nausea"], "serious_count": 5
    }

    chunker_cls.chunk_clinical_trial.return_value = [{"text": "trial chunk", "metadata": {"source": "clinicaltrials", "company": "TestPharma"}}]
    chunker_cls.chunk_fda_approval.return_value = [{"text": "approval chunk", "metadata": {"source": "fda_approval"}}]
    chunker_cls.chunk_fda_label.return_value = [{"text": "label chunk", "metadata": {"source": "fda_label"}}]
    chunker_cls.chunk_adverse_events.return_value = [{"text": "ae chunk", "metadata": {"source": "fda_adverse_events"}}]

    retriever.retrieve_for_report.return_value = [
        {"text": "trial chunk", "metadata": {"source": "clinicaltrials"}},
        {"text": "approval chunk", "metadata": {"source": "fda_approval"}},
    ]

    generator.generate_report.return_value = "## Due Diligence Report: TestPharma\n\nGreat pipeline."

    return {
        "ct_client": ct_client, "fda_client": fda_client,
        "chunker_cls": chunker_cls, "embedder": embedder,
        "retriever": retriever, "generator": generator
    }


@pytest.mark.asyncio
async def test_build_report(mock_deps):
    builder = ReportBuilder(**mock_deps)
    report = await builder.build_report("TestPharma")

    assert "Due Diligence Report" in report
    mock_deps["ct_client"].search_by_sponsor.assert_called_once_with("TestPharma", condition=None)
    mock_deps["ct_client"].search_by_drug.assert_called_once_with("TestPharma", condition=None)
    mock_deps["fda_client"].search_approvals.assert_called_once()
    mock_deps["embedder"].embed_and_store.assert_called_once()
    mock_deps["generator"].generate_report.assert_called_once()
