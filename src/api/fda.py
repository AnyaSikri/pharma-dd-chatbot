# src/api/fda.py
from __future__ import annotations

import re
import httpx
from typing import Optional

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def _escape_lucene(value: str) -> str:
    return _LUCENE_SPECIAL.sub(r'\\\1', value)


class FDAClient:
    """Async client for the openFDA API (drugs, devices, labels, adverse events)."""

    DRUGSFDA_URL = "https://api.fda.gov/drug/drugsfda.json"
    LABEL_URL = "https://api.fda.gov/drug/label.json"
    EVENTS_URL = "https://api.fda.gov/drug/event.json"
    DEVICE_510K_URL = "https://api.fda.gov/device/510k.json"
    DEVICE_EVENT_URL = "https://api.fda.gov/device/event.json"
    DEVICE_RECALL_URL = "https://api.fda.gov/device/recall.json"

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
            "search": f'openfda.manufacturer_name:"{_escape_lucene(company_or_drug)}"+openfda.brand_name:"{_escape_lucene(company_or_drug)}"',
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
            "search": f'openfda.brand_name:"{_escape_lucene(drug_name)}"+openfda.generic_name:"{_escape_lucene(drug_name)}"',
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
            "search": f'patient.drug.openfda.brand_name:"{_escape_lucene(drug_name)}"',
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

    async def search_device_clearances(self, company_or_device: str, limit: int = 50) -> list[dict]:
        """Search 510(k) device clearances by company or device name."""
        escaped = _escape_lucene(company_or_device)
        params = {
            **self._base_params(),
            "search": f'applicant:"{escaped}"+device_name:"{escaped}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.DEVICE_510K_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return []

        results = []
        for record in data.get("results", []):
            results.append({
                "k_number": record.get("k_number", ""),
                "applicant": record.get("applicant", ""),
                "device_name": record.get("device_name", ""),
                "product_code": record.get("product_code", ""),
                "clearance_type": record.get("clearance_type", ""),
                "decision_date": record.get("decision_date", ""),
                "decision_description": record.get("decision_description", ""),
                "advisory_committee_description": record.get("advisory_committee_description", ""),
            })
        return results

    async def get_device_adverse_events_summary(self, device_name: str, limit: int = 10) -> dict:
        """Get a summary of MAUDE adverse event reports for a device."""
        escaped = _escape_lucene(device_name)
        params = {
            **self._base_params(),
            "search": f'device.generic_name:"{escaped}"+device.brand_name:"{escaped}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.DEVICE_EVENT_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return {"total_reports": 0, "serious_count": 0, "sample_events": []}

        total = data.get("meta", {}).get("results", {}).get("total", 0)
        serious_count = 0
        sample_events = []

        for report in data.get("results", []):
            if report.get("event_type", "").upper() in ("DEATH", "INJURY"):
                serious_count += 1
            for text_entry in report.get("mdr_text", []):
                narrative = text_entry.get("text", "")
                if narrative and len(sample_events) < 5:
                    sample_events.append(narrative[:300])

        return {
            "total_reports": total,
            "serious_count": serious_count,
            "sample_events": sample_events,
        }

    async def search_device_recalls(self, company_or_device: str, limit: int = 20) -> list[dict]:
        """Search device recall records by company or product description."""
        escaped = _escape_lucene(company_or_device)
        params = {
            **self._base_params(),
            "search": f'recalling_firm:"{escaped}"+product_description:"{escaped}"',
            "limit": min(limit, 99),
        }
        response = await self._client.get(self.DEVICE_RECALL_URL, params=params)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return []

        results = []
        for record in data.get("results", []):
            results.append({
                "res_event_number": record.get("res_event_number", ""),
                "recalling_firm": record.get("recalling_firm", ""),
                "product_description": record.get("product_description", ""),
                "reason_for_recall": record.get("reason_for_recall", ""),
                "classification": record.get("event_date_terminated", "") and "Class " + str(record.get("product_res_number", "")),
                "event_date_terminated": record.get("event_date_terminated", ""),
                "status": record.get("status", ""),
            })
        return results

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
