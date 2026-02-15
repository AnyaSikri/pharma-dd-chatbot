# tests/test_sec_edgar.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.api.sec_edgar import SECEdgarClient


SAMPLE_TICKERS = {
    "0": {"cik_str": "789019", "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "1": {"cik_str": "1682852", "ticker": "MRNA", "title": "MODERNA INC"},
}

SAMPLE_SUBMISSIONS = {
    "name": "Moderna, Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "4"],
            "filingDate": ["2024-02-22", "2024-08-01", "2024-06-15", "2024-05-01"],
            "accessionNumber": [
                "0001682852-24-000010", "0001682852-24-000020",
                "0001682852-24-000030", "0001682852-24-000040",
            ],
            "primaryDocument": [
                "mrna-20231231.htm", "mrna-20240630.htm",
                "mrna-8k.htm", "form4.htm",
            ],
            "primaryDocDescription": [
                "10-K Annual Report", "10-Q Quarterly Report",
                "8-K Current Report", "Statement of Changes",
            ],
        }
    }
}

SAMPLE_COMPANY_FACTS = {
    "entityName": "Moderna, Inc.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"val": 18435000000, "end": "2022-12-31", "form": "10-K"},
                        {"val": 6671000000, "end": "2023-12-31", "form": "10-K"},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"val": 8362000000, "end": "2022-12-31", "form": "10-K"},
                        {"val": -4714000000, "end": "2023-12-31", "form": "10-K"},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"val": 18100000000, "end": "2023-12-31", "form": "10-K"},
                    ]
                }
            },
        }
    }
}


@pytest.mark.asyncio
async def test_lookup_company_by_ticker():
    client = SECEdgarClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_TICKERS)
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        result = await client.lookup_company("MRNA")

    assert result is not None
    assert result["ticker"] == "MRNA"
    assert result["cik"] == "0001682852"


@pytest.mark.asyncio
async def test_lookup_company_by_name():
    client = SECEdgarClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_TICKERS)
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        result = await client.lookup_company("Moderna")

    assert result is not None
    assert result["ticker"] == "MRNA"


@pytest.mark.asyncio
async def test_lookup_company_not_found():
    client = SECEdgarClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_TICKERS)
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        result = await client.lookup_company("NonexistentCorp")

    assert result is None


@pytest.mark.asyncio
async def test_get_filings_filters_and_parses():
    client = SECEdgarClient()
    client._tickers_cache = SAMPLE_TICKERS
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_SUBMISSIONS)
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        filings = await client.get_filings("0001682852")

    # Should exclude form "4", keep 10-K, 10-Q, 8-K
    assert len(filings) == 3
    assert all(f["form_type"] in ("10-K", "10-Q", "8-K") for f in filings)
    assert filings[0]["filing_date"] == "2024-02-22"
    assert "sec.gov" in filings[0]["filing_url"]


@pytest.mark.asyncio
async def test_get_company_facts_extracts_latest():
    client = SECEdgarClient()
    mock_response = AsyncMock()
    mock_response.json = MagicMock(return_value=SAMPLE_COMPANY_FACTS)
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        facts = await client.get_company_facts("0001682852")

    assert facts["company_name"] == "Moderna, Inc."
    assert facts["revenue"]["value"] == 6671000000
    assert facts["revenue"]["period_end"] == "2023-12-31"
    assert facts["net_income"]["value"] == -4714000000
    assert facts["total_assets"]["value"] == 18100000000


@pytest.mark.asyncio
async def test_get_market_data_returns_none_on_failure():
    client = SECEdgarClient()
    with patch.object(client, "_fetch_yfinance", return_value=None):
        result = await client.get_market_data("INVALID")
    assert result is None
