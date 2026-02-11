# tests/test_integration.py
"""Integration test verifying the full pipeline with mocked external APIs."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.report.builder import ReportBuilder


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_apis():
    """Test the complete flow: fetch -> chunk -> embed -> retrieve -> generate."""
    # Mock ClinicalTrials API
    ct_client = AsyncMock(spec=ClinicalTrialsClient)
    ct_client.search_by_drug.return_value = []
    ct_client.search_by_sponsor.return_value = [
        {
            "nct_id": "NCT99999999",
            "title": "Phase 3 Study of MagicDrug in Oncology",
            "status": "ACTIVE_NOT_RECRUITING",
            "phase": "Phase 3",
            "enrollment": 1200,
            "start_date": "2023-06-01",
            "primary_completion_date": "2025-12-01",
            "completion_date": "2026-06-01",
            "sponsor": "IntegrationPharma",
            "conditions": ["Breast Cancer"],
            "interventions": [{"name": "MagicDrug", "type": "DRUG"}],
            "primary_outcomes": [{"measure": "PFS", "timeFrame": "36 months"}],
            "secondary_outcomes": [],
            "brief_summary": "Evaluating MagicDrug in breast cancer patients.",
            "has_results": False,
        }
    ]

    # Mock FDA API
    fda_client = AsyncMock(spec=FDAClient)
    fda_client.search_approvals.return_value = [
        {
            "application_number": "NDA999999",
            "brand_name": "MagicDrug",
            "generic_name": "magicdruginib",
            "manufacturer": "IntegrationPharma",
            "sponsor_name": "IntegrationPharma",
            "products": [{"brand_name": "MagicDrug", "active_ingredients": [{"name": "magicdruginib", "strength": "100mg"}], "dosage_form": "TABLET", "route": "ORAL"}],
            "submissions": [{"submission_type": "ORIG", "submission_status": "AP", "submission_status_date": "20220101", "submission_class_code_description": "New Molecular Entity"}],
        }
    ]
    fda_client.search_labels.return_value = [
        {
            "brand_name": "MagicDrug",
            "generic_name": "magicdruginib",
            "manufacturer": "IntegrationPharma",
            "indications": "For treatment of HER2+ breast cancer",
            "boxed_warning": "",
            "warnings": "Monitor liver function",
            "adverse_reactions": "Common: fatigue, nausea",
        }
    ]
    fda_client.get_adverse_events_summary.return_value = {
        "total_reports": 250,
        "sample_reactions": ["Fatigue", "Nausea", "Headache"],
        "serious_count": 12,
    }
    fda_client.search_device_clearances.return_value = []
    fda_client.get_device_adverse_events_summary.return_value = {"total_reports": 0, "serious_count": 0, "sample_events": []}
    fda_client.search_device_recalls.return_value = []

    # Mock embedder (don't actually call OpenAI)
    embedder = MagicMock(spec=Embedder)
    embedder.embed_and_store = MagicMock()
    embedder.embed_query = MagicMock(return_value=[0.1] * 1536)
    mock_collection = MagicMock()
    embedder.get_collection.return_value = mock_collection

    # Retriever gets all docs from collection (no company filter)
    mock_collection.get.return_value = {"documents": [], "metadatas": [], "ids": []}
    retriever = Retriever(embedder=embedder)

    # Mock Claude response
    generator = MagicMock(spec=Generator)
    generator.generate_report.return_value = (
        "## Due Diligence Report: IntegrationPharma\n\n"
        "### Pipeline Overview\n1 active Phase 3 program.\n\n"
        "### Clinical Trials\n- NCT99999999: Phase 3, ACTIVE_NOT_RECRUITING\n\n"
        "### FDA / Regulatory\n- MagicDrug (NDA999999) approved 2022\n\n"
        "### Risk Assessment\nSingle-asset pipeline risk.\n\n"
        "### Sources\n- https://clinicaltrials.gov/study/NCT99999999"
    )

    builder = ReportBuilder(
        ct_client=ct_client,
        fda_client=fda_client,
        chunker_cls=Chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )

    report = await builder.build_report("IntegrationPharma")

    # Verify the pipeline executed correctly
    ct_client.search_by_sponsor.assert_called_once_with("IntegrationPharma", condition=None)
    ct_client.search_by_drug.assert_called_once_with("IntegrationPharma", condition=None)
    fda_client.search_approvals.assert_called_once_with("IntegrationPharma")
    fda_client.search_labels.assert_called_once_with("MagicDrug")
    fda_client.get_adverse_events_summary.assert_called_once_with("MagicDrug")
    embedder.embed_and_store.assert_called_once()

    # Verify chunks were created and passed
    embed_call = embedder.embed_and_store.call_args
    chunks = embed_call.args[0] if embed_call.args else embed_call.kwargs.get("chunks", [])
    assert len(chunks) >= 3  # at least: trial + approval + label + AE

    generator.generate_report.assert_called_once()

    assert "Due Diligence Report" in report
    assert "IntegrationPharma" in report
