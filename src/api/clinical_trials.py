# src/api/clinical_trials.py
from __future__ import annotations

import httpx
from typing import Optional


PHASE_MAP = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "N/A",
}


class ClinicalTrialsClient:
    """Async client for the ClinicalTrials.gov v2 API."""

    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search_by_sponsor(self, sponsor: str, max_results: int = 100) -> list[dict]:
        """Search clinical trials by sponsor name."""
        return await self._search(params={"query.spons": sponsor}, max_results=max_results)

    async def search_by_drug(self, drug_name: str, max_results: int = 100) -> list[dict]:
        """Search clinical trials by drug / intervention name."""
        return await self._search(params={"query.intr": drug_name}, max_results=max_results)

    async def _search(self, params: dict, max_results: int) -> list[dict]:
        """Internal paginated search against the ClinicalTrials.gov v2 API."""
        all_studies: list[dict] = []
        page_token: Optional[str] = None

        while len(all_studies) < max_results:
            request_params = {
                **params,
                "pageSize": min(max_results - len(all_studies), 100),
                "format": "json",
            }
            if page_token:
                request_params["pageToken"] = page_token

            response = await self._client.get(self.BASE_URL, params=request_params)
            response.raise_for_status()
            data = response.json()

            for study in data.get("studies", []):
                parsed = self._parse_study(study)
                if parsed:
                    all_studies.append(parsed)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return all_studies

    def _parse_study(self, study: dict) -> Optional[dict]:
        """Parse a raw ClinicalTrials.gov study object into a flat dict."""
        protocol = study.get("protocolSection", {})
        id_mod = protocol.get("identificationModule", {})
        status_mod = protocol.get("statusModule", {})
        design_mod = protocol.get("designModule", {})
        conditions_mod = protocol.get("conditionsModule", {})
        arms_mod = protocol.get("armsInterventionsModule", {})
        outcomes_mod = protocol.get("outcomesModule", {})
        sponsor_mod = protocol.get("sponsorCollaboratorsModule", {})
        desc_mod = protocol.get("descriptionModule", {})

        phases_raw = design_mod.get("phases", [])
        phase = ", ".join(PHASE_MAP.get(p, p) for p in phases_raw) if phases_raw else "N/A"

        interventions = [
            {"name": i["name"], "type": i.get("type", "UNKNOWN")}
            for i in arms_mod.get("interventions", [])
        ]

        return {
            "nct_id": id_mod.get("nctId", ""),
            "title": id_mod.get("briefTitle", ""),
            "official_title": id_mod.get("officialTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": phase,
            "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
            "start_date": status_mod.get("startDateStruct", {}).get("date"),
            "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date"),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date"),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "conditions": conditions_mod.get("conditions", []),
            "interventions": interventions,
            "primary_outcomes": outcomes_mod.get("primaryOutcomes", []),
            "secondary_outcomes": outcomes_mod.get("secondaryOutcomes", []),
            "brief_summary": desc_mod.get("briefSummary", ""),
            "has_results": study.get("hasResults", False),
        }

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
