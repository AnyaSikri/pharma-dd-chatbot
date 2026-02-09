# tests/test_fda.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.fda import FDAClient


SAMPLE_DRUGSFDA_RESPONSE = {
    "meta": {"results": {"total": 1}},
    "results": [
        {
            "application_number": "NDA021821",
            "sponsor_name": "PF PRISM CV",
            "openfda": {
                "brand_name": ["TYGACIL"],
                "generic_name": ["TIGECYCLINE"],
                "manufacturer_name": ["Pfizer Inc."],
                "product_type": ["HUMAN PRESCRIPTION DRUG"],
                "route": ["INTRAVENOUS"]
            },
            "products": [
                {
                    "product_number": "001",
                    "brand_name": "TYGACIL",
                    "active_ingredients": [{"name": "TIGECYCLINE", "strength": "50MG/VIAL"}],
                    "dosage_form": "POWDER",
                    "route": "INTRAVENOUS",
                    "marketing_status": "Prescription"
                }
            ],
            "submissions": [
                {
                    "submission_type": "ORIG",
                    "submission_number": "1",
                    "submission_status": "AP",
                    "submission_status_date": "20050617",
                    "submission_class_code_description": "New Molecular Entity"
                }
            ]
        }
    ]
}


SAMPLE_LABEL_RESPONSE = {
    "meta": {"results": {"total": 1}},
    "results": [
        {
            "openfda": {
                "brand_name": ["TYGACIL"],
                "generic_name": ["TIGECYCLINE"],
                "manufacturer_name": ["Pfizer Inc."]
            },
            "indications_and_usage": ["TYGACIL is indicated for treatment of infections..."],
            "boxed_warning": ["WARNING: ALL-CAUSE MORTALITY..."],
            "warnings_and_cautions": ["Anaphylaxis has been reported..."],
            "adverse_reactions": ["The most common adverse reactions are nausea and vomiting..."]
        }
    ]
}


SAMPLE_EVENTS_RESPONSE = {
    "meta": {"results": {"total": 5000}},
    "results": [
        {
            "safetyreportid": "12345",
            "serious": "1",
            "receivedate": "20240115",
            "patient": {
                "reaction": [
                    {"reactionmeddrapt": "Nausea"},
                    {"reactionmeddrapt": "Headache"}
                ],
                "drug": [
                    {
                        "medicinalproduct": "TYGACIL",
                        "drugcharacterization": "1",
                        "openfda": {
                            "brand_name": ["TYGACIL"],
                            "generic_name": ["TIGECYCLINE"]
                        }
                    }
                ]
            }
        }
    ]
}


@pytest.mark.asyncio
async def test_search_approvals_parses_results():
    client = FDAClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_DRUGSFDA_RESPONSE)
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_approvals("Pfizer")

    assert len(results) == 1
    drug = results[0]
    assert drug["application_number"] == "NDA021821"
    assert drug["brand_name"] == "TYGACIL"
    assert drug["generic_name"] == "TIGECYCLINE"
    assert len(drug["products"]) == 1
    assert len(drug["submissions"]) == 1


@pytest.mark.asyncio
async def test_search_labels_parses_results():
    client = FDAClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_LABEL_RESPONSE)
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_labels("TYGACIL")

    assert len(results) == 1
    label = results[0]
    assert label["brand_name"] == "TYGACIL"
    assert "indications" in label
    assert "boxed_warning" in label
    assert "warnings" in label
    assert "adverse_reactions" in label


@pytest.mark.asyncio
async def test_get_adverse_event_summary():
    client = FDAClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_EVENTS_RESPONSE)
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response):
        summary = await client.get_adverse_events_summary("TYGACIL")

    assert summary["total_reports"] == 5000
    assert summary["sample_reactions"] == ["Nausea", "Headache"]
    assert summary["serious_count"] >= 0


@pytest.mark.asyncio
async def test_search_approvals_handles_no_results():
    client = FDAClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value={"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_approvals("NonexistentPharma")

    assert results == []
