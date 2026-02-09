# tests/test_clinical_trials.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.clinical_trials import ClinicalTrialsClient


SAMPLE_API_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "A Phase 3 Study of Drug X",
                    "organization": {
                        "fullName": "TestPharma Inc",
                        "class": "INDUSTRY"
                    }
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2024-01-15", "type": "ACTUAL"},
                    "primaryCompletionDateStruct": {"date": "2026-06-01", "type": "ESTIMATED"},
                    "completionDateStruct": {"date": "2026-12-01", "type": "ESTIMATED"}
                },
                "designModule": {
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 500, "type": "ESTIMATED"}
                },
                "conditionsModule": {
                    "conditions": ["Non-Small Cell Lung Cancer"]
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"name": "Drug X", "type": "DRUG", "description": "Experimental treatment"}
                    ]
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Overall Survival", "timeFrame": "24 months"}
                    ]
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "TestPharma Inc", "class": "INDUSTRY"}
                },
                "descriptionModule": {
                    "briefSummary": "This study evaluates Drug X in NSCLC patients."
                }
            },
            "hasResults": False
        }
    ],
    "nextPageToken": None
}


@pytest.mark.asyncio
async def test_search_by_sponsor_parses_studies():
    client = ClinicalTrialsClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_API_RESPONSE)
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response) as mock_get:
        results = await client.search_by_sponsor("TestPharma")

    assert len(results) == 1
    trial = results[0]
    assert trial["nct_id"] == "NCT12345678"
    assert trial["title"] == "A Phase 3 Study of Drug X"
    assert trial["status"] == "RECRUITING"
    assert trial["phase"] == "Phase 3"
    assert trial["enrollment"] == 500
    assert trial["sponsor"] == "TestPharma Inc"
    assert trial["conditions"] == ["Non-Small Cell Lung Cancer"]
    assert trial["interventions"] == [{"name": "Drug X", "type": "DRUG"}]
    assert trial["primary_outcomes"] == [{"measure": "Overall Survival", "timeFrame": "24 months"}]


@pytest.mark.asyncio
async def test_search_by_sponsor_calls_correct_url():
    client = ClinicalTrialsClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value={"studies": [], "nextPageToken": None})
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response) as mock_get:
        await client.search_by_sponsor("Moderna")

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "query.spons" in call_args.kwargs.get("params", {}) or "query.spons" in str(call_args)


@pytest.mark.asyncio
async def test_search_by_drug_returns_parsed_trials():
    client = ClinicalTrialsClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_API_RESPONSE)
    mock_response.raise_for_status = MagicMock(return_value=None)

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_by_drug("Drug X")

    assert len(results) == 1
    assert results[0]["nct_id"] == "NCT12345678"
