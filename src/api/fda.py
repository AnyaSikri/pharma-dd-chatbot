# src/api/fda.py
from __future__ import annotations

import httpx
from typing import Optional


class FDAClient:
    """Async client for the openFDA API (drug approvals, labels, adverse events)."""

    DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
    LABEL_URL = "https://api.fda.gov/drug/label.json"
    EVENTS_URL = "https://api.fda.gov/drug/event.json"

    def __init__(self, api_key: Optional[str] = None):
        self._client = httpx.AsyncClient(timeout=30.0)
        self._api_key = api_key

    def _base_params(self) -> dict:
        if self._api_key:
            return {"api_key": self._api_key}
        return {}

    async def search_approvals(self, company_or_drug: str, limit: int = 50) -> list[dict]:
        """Search drug approval records by company or drug name."""
        params = {
            **self._base_params(),
            "search": f'openfda.manufacturer_name:"{company_or_drug}"+openfda.brand_name:"{company_or_drug}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.DRUGSFDA_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return []

        results = []
        for record in data.get("results", []):
            openfda = record.get("openfda", {})
            results.append({
                "application_number": record.get("application_number", ""),
                "sponsor_name": record.get("sponsor_name", ""),
                "brand_name": (openfda.get("brand_name") or [""])[0],
                "generic_name": (openfda.get("generic_name") or [""])[0],
                "manufacturer": (openfda.get("manufacturer_name") or [""])[0],
                "product_type": (openfda.get("product_type") or [""])[0],
                "route": (openfda.get("route") or [""])[0],
                "products": record.get("products", []),
                "submissions": record.get("submissions", []),
            })
        return results

    async def search_labels(self, drug_name: str, limit: int = 10) -> list[dict]:
        """Search drug labeling information by drug name."""
        params = {
            **self._base_params(),
            "search": f'openfda.brand_name:"{drug_name}"+openfda.generic_name:"{drug_name}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.LABEL_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return []

        results = []
        for record in data.get("results", []):
            openfda = record.get("openfda", {})
            results.append({
                "brand_name": (openfda.get("brand_name") or [""])[0],
                "generic_name": (openfda.get("generic_name") or [""])[0],
                "manufacturer": (openfda.get("manufacturer_name") or [""])[0],
                "indications": (record.get("indications_and_usage") or [""])[0],
                "boxed_warning": (record.get("boxed_warning") or [""])[0],
                "warnings": (record.get("warnings_and_cautions") or [""])[0],
                "adverse_reactions": (record.get("adverse_reactions") or [""])[0],
            })
        return results

    async def get_adverse_events_summary(self, drug_name: str, limit: int = 10) -> dict:
        """Get a summary of adverse event reports for a given drug."""
        params = {
            **self._base_params(),
            "search": f'patient.drug.openfda.brand_name:"{drug_name}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.EVENTS_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return {"total_reports": 0, "sample_reactions": [], "serious_count": 0}

        total = data.get("meta", {}).get("results", {}).get("total", 0)
        reactions = []
        serious_count = 0

        for report in data.get("results", []):
            if report.get("serious") == "1":
                serious_count += 1
            for reaction in report.get("patient", {}).get("reaction", []):
                name = reaction.get("reactionmeddrapt")
                if name and name not in reactions:
                    reactions.append(name)

        return {
            "total_reports": total,
            "sample_reactions": reactions[:20],
            "serious_count": serious_count,
        }

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
