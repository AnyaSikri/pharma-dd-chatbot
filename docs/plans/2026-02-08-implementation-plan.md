# Pharma DD Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a RAG-based Streamlit chatbot that queries ClinicalTrials.gov and openFDA APIs to generate pharma due diligence reports and answer investor follow-up questions using Claude.

**Architecture:** Data flows from two public APIs (ClinicalTrials.gov v2, openFDA) through an ingestion layer that chunks and embeds documents into ChromaDB. A retrieval layer pulls relevant chunks and feeds them to Claude API for report generation and conversational Q&A. Streamlit provides the UI with two modes: report generation and chat.

**Tech Stack:** Python 3.12, Streamlit, Anthropic SDK (Claude), OpenAI SDK (embeddings only), ChromaDB, httpx

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/api/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/rag/__init__.py`
- Create: `src/report/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```
streamlit>=1.30.0
anthropic>=0.40.0
openai>=1.10.0
chromadb>=0.4.22
httpx>=0.26.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
OPENFDA_API_KEY=your-openfda-api-key-optional
```

**Step 3: Create all `__init__.py` files**

All empty files.

**Step 4: Install dependencies**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && pip install -r requirements.txt`

**Step 5: Commit**

```bash
git add requirements.txt .env.example src/ tests/
git commit -m "feat: project scaffolding with dependencies"
```

---

## Task 2: ClinicalTrials.gov API Client

**Files:**
- Create: `src/api/clinical_trials.py`
- Create: `tests/test_clinical_trials.py`

**Step 1: Write the failing test**

```python
# tests/test_clinical_trials.py
import pytest
from unittest.mock import patch, AsyncMock
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
    mock_response.json.return_value = SAMPLE_API_RESPONSE
    mock_response.raise_for_status = lambda: None

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
    mock_response.json.return_value = {"studies": [], "nextPageToken": None}
    mock_response.raise_for_status = lambda: None

    with patch.object(client._client, "get", return_value=mock_response) as mock_get:
        await client.search_by_sponsor("Moderna")

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "query.spons" in call_args.kwargs.get("params", {}) or "query.spons" in str(call_args)


@pytest.mark.asyncio
async def test_search_by_drug_returns_parsed_trials():
    client = ClinicalTrialsClient()
    mock_response = AsyncMock()
    mock_response.json.return_value = SAMPLE_API_RESPONSE
    mock_response.raise_for_status = lambda: None

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_by_drug("Drug X")

    assert len(results) == 1
    assert results[0]["nct_id"] == "NCT12345678"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_clinical_trials.py -v`
Expected: FAIL with ImportError (module not found)

**Step 3: Write the implementation**

```python
# src/api/clinical_trials.py
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
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search_by_sponsor(self, sponsor: str, max_results: int = 100) -> list[dict]:
        return await self._search(params={"query.spons": sponsor}, max_results=max_results)

    async def search_by_drug(self, drug_name: str, max_results: int = 100) -> list[dict]:
        return await self._search(params={"query.intr": drug_name}, max_results=max_results)

    async def _search(self, params: dict, max_results: int) -> list[dict]:
        all_studies = []
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
        await self._client.aclose()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_clinical_trials.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/api/clinical_trials.py tests/test_clinical_trials.py
git commit -m "feat: ClinicalTrials.gov API client with sponsor and drug search"
```

---

## Task 3: openFDA API Client

**Files:**
- Create: `src/api/fda.py`
- Create: `tests/test_fda.py`

**Step 1: Write the failing tests**

```python
# tests/test_fda.py
import pytest
from unittest.mock import patch, AsyncMock
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
    mock_response.json.return_value = SAMPLE_DRUGSFDA_RESPONSE
    mock_response.raise_for_status = lambda: None

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
    mock_response.json.return_value = SAMPLE_LABEL_RESPONSE
    mock_response.raise_for_status = lambda: None

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
    mock_response.json.return_value = SAMPLE_EVENTS_RESPONSE
    mock_response.raise_for_status = lambda: None

    with patch.object(client._client, "get", return_value=mock_response):
        summary = await client.get_adverse_events_summary("TYGACIL")

    assert summary["total_reports"] == 5000
    assert summary["sample_reactions"] == ["Nausea", "Headache"]
    assert summary["serious_count"] >= 0


@pytest.mark.asyncio
async def test_search_approvals_handles_no_results():
    client = FDAClient()
    mock_response = AsyncMock()
    mock_response.json.return_value = {"error": {"code": "NOT_FOUND", "message": "No matches found!"}}
    mock_response.raise_for_status = lambda: None

    with patch.object(client._client, "get", return_value=mock_response):
        results = await client.search_approvals("NonexistentPharma")

    assert results == []
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_fda.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/api/fda.py
import httpx
from typing import Optional


class FDAClient:
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
        await self._client.aclose()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_fda.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/api/fda.py tests/test_fda.py
git commit -m "feat: openFDA API client for approvals, labels, and adverse events"
```

---

## Task 4: Document Chunker

**Files:**
- Create: `src/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
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
    # Should create separate chunks for key sections
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_chunker.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/ingestion/chunker.py


class Chunker:
    @staticmethod
    def chunk_clinical_trial(trial: dict) -> list[dict]:
        outcomes_text = ""
        for outcome in trial.get("primary_outcomes", []):
            outcomes_text += f"  - {outcome['measure']} ({outcome.get('timeFrame', 'N/A')})\n"

        interventions_text = ", ".join(
            i["name"] for i in trial.get("interventions", [])
        )

        text = (
            f"Clinical Trial: {trial['title']}\n"
            f"NCT ID: {trial['nct_id']}\n"
            f"Phase: {trial['phase']}\n"
            f"Status: {trial['status']}\n"
            f"Sponsor: {trial['sponsor']}\n"
            f"Enrollment: {trial.get('enrollment', 'N/A')}\n"
            f"Conditions: {', '.join(trial.get('conditions', []))}\n"
            f"Interventions: {interventions_text}\n"
            f"Start Date: {trial.get('start_date', 'N/A')}\n"
            f"Primary Completion Date: {trial.get('primary_completion_date', 'N/A')}\n"
            f"Primary Outcomes:\n{outcomes_text}"
            f"Summary: {trial.get('brief_summary', '')}"
        )

        return [{
            "text": text,
            "metadata": {
                "source": "clinicaltrials",
                "nct_id": trial["nct_id"],
                "company": trial["sponsor"],
                "drug_name": interventions_text,
                "phase": trial["phase"],
                "status": trial["status"],
                "date": trial.get("start_date", ""),
            }
        }]

    @staticmethod
    def chunk_fda_approval(approval: dict) -> list[dict]:
        products_text = ""
        for p in approval.get("products", []):
            ingredients = ", ".join(
                f"{i['name']} {i.get('strength', '')}" for i in p.get("active_ingredients", [])
            )
            products_text += f"  - {p.get('brand_name', '')} ({p.get('dosage_form', '')}, {p.get('route', '')}): {ingredients}\n"

        submissions_text = ""
        for s in approval.get("submissions", []):
            date_raw = s.get("submission_status_date", "")
            date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}" if len(date_raw) == 8 else date_raw
            submissions_text += f"  - {s.get('submission_type', '')} ({s.get('submission_class_code_description', '')}): {s.get('submission_status', '')} on {date_fmt}\n"

        text = (
            f"FDA Drug Application: {approval['brand_name']} ({approval['generic_name']})\n"
            f"Application Number: {approval['application_number']}\n"
            f"Manufacturer: {approval.get('manufacturer', '')}\n"
            f"Products:\n{products_text}"
            f"Submissions:\n{submissions_text}"
        )

        return [{
            "text": text,
            "metadata": {
                "source": "fda_approval",
                "drug_name": approval["brand_name"],
                "company": approval.get("manufacturer", ""),
                "application_number": approval["application_number"],
            }
        }]

    @staticmethod
    def chunk_fda_label(label: dict) -> list[dict]:
        chunks = []
        drug_name = label.get("brand_name", "Unknown")
        company = label.get("manufacturer", "")

        sections = {
            "indications": label.get("indications", ""),
            "boxed_warning": label.get("boxed_warning", ""),
            "warnings": label.get("warnings", ""),
            "adverse_reactions": label.get("adverse_reactions", ""),
        }

        for section_name, content in sections.items():
            if not content:
                continue
            text = (
                f"FDA Label - {drug_name} ({label.get('generic_name', '')}) - {section_name.replace('_', ' ').title()}\n\n"
                f"{content}"
            )
            chunks.append({
                "text": text,
                "metadata": {
                    "source": "fda_label",
                    "drug_name": drug_name,
                    "company": company,
                    "section": section_name,
                }
            })

        return chunks if chunks else [{
            "text": f"FDA Label for {drug_name}: No detailed label information available.",
            "metadata": {"source": "fda_label", "drug_name": drug_name, "company": company, "section": "summary"}
        }]

    @staticmethod
    def chunk_adverse_events(drug_name: str, ae_summary: dict) -> list[dict]:
        reactions_text = ", ".join(ae_summary.get("sample_reactions", []))
        text = (
            f"Adverse Events Summary for {drug_name}\n"
            f"Total FAERS Reports: {ae_summary['total_reports']:,}\n"
            f"Serious Reports in Sample: {ae_summary.get('serious_count', 0)}\n"
            f"Common Reactions: {reactions_text}"
        )

        return [{
            "text": text,
            "metadata": {
                "source": "fda_adverse_events",
                "drug_name": drug_name,
            }
        }]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_chunker.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/ingestion/chunker.py tests/test_chunker.py
git commit -m "feat: document chunker for clinical trials, FDA approvals, labels, and adverse events"
```

---

## Task 5: Embedder & ChromaDB Storage

**Files:**
- Create: `src/ingestion/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.embedder import Embedder


@pytest.fixture
def mock_openai():
    with patch("src.ingestion.embedder.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_chroma():
    with patch("src.ingestion.embedder.chromadb") as mock_mod:
        mock_client = MagicMock()
        mock_mod.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        yield mock_collection


def test_embed_and_store(mock_openai, mock_chroma):
    embedder = Embedder(openai_api_key="test-key", chroma_path="/tmp/test_chroma")

    chunks = [
        {"text": "Clinical trial NCT123", "metadata": {"source": "clinicaltrials", "nct_id": "NCT123"}},
        {"text": "FDA approval NDA456", "metadata": {"source": "fda_approval", "application_number": "NDA456"}},
    ]

    embedder.embed_and_store(chunks, collection_name="test_company")

    mock_openai.embeddings.create.assert_called()
    mock_chroma.add.assert_called_once()
    call_args = mock_chroma.add.call_args
    assert len(call_args.kwargs["documents"]) == 2
    assert len(call_args.kwargs["ids"]) == 2


def test_embed_and_store_batches_large_inputs(mock_openai, mock_chroma):
    embedder = Embedder(openai_api_key="test-key", chroma_path="/tmp/test_chroma")

    # Create 150 chunks (should be batched)
    chunks = [
        {"text": f"Chunk {i}", "metadata": {"source": "test"}}
        for i in range(150)
    ]

    # Make embeddings.create return enough embeddings each call
    def side_effect(**kwargs):
        n = len(kwargs["input"])
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
        return mock_resp

    mock_openai.embeddings.create.side_effect = side_effect

    embedder.embed_and_store(chunks, collection_name="test_company")

    # Should have called embeddings.create multiple times for batching
    assert mock_openai.embeddings.create.call_count >= 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_embedder.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/ingestion/embedder.py
import hashlib
import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


class Embedder:
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_db"):
        self._openai = OpenAI(api_key=openai_api_key)
        self._chroma = chromadb.PersistentClient(path=chroma_path)

    def embed_and_store(self, chunks: list[dict], collection_name: str) -> None:
        if not chunks:
            return

        collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        all_embeddings = []
        texts = [c["text"] for c in chunks]

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])

        ids = [
            hashlib.md5(chunk["text"].encode()).hexdigest()
            for chunk in chunks
        ]
        metadatas = [chunk["metadata"] for chunk in chunks]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
        )

    def get_collection(self, collection_name: str):
        return self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_embedder.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/ingestion/embedder.py tests/test_embedder.py
git commit -m "feat: embedder with OpenAI embeddings and ChromaDB storage"
```

---

## Task 6: RAG Retriever

**Files:**
- Create: `src/rag/retriever.py`
- Create: `tests/test_retriever.py`

**Step 1: Write the failing tests**

```python
# tests/test_retriever.py
import pytest
from unittest.mock import patch, MagicMock
from src.rag.retriever import Retriever


@pytest.fixture
def mock_embedder():
    with patch("src.rag.retriever.Embedder") as MockEmbedder:
        instance = MagicMock()
        MockEmbedder.return_value = instance

        mock_collection = MagicMock()
        instance.get_collection.return_value = mock_collection
        instance._openai = MagicMock()

        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 1536)]
        instance._openai.embeddings.create.return_value = mock_embed_response

        yield instance, mock_collection


def test_retrieve_for_report_uses_metadata_filter(mock_embedder):
    embedder_instance, mock_collection = mock_embedder
    mock_collection.get.return_value = {
        "documents": ["Trial NCT123 Phase 3", "FDA Approval NDA456"],
        "metadatas": [
            {"source": "clinicaltrials", "company": "TestPharma"},
            {"source": "fda_approval", "company": "TestPharma"}
        ],
        "ids": ["id1", "id2"]
    }

    retriever = Retriever(embedder=embedder_instance)
    results = retriever.retrieve_for_report("test_collection", "TestPharma")

    mock_collection.get.assert_called_once()
    assert len(results) == 2
    assert results[0]["text"] == "Trial NCT123 Phase 3"


def test_retrieve_for_chat_uses_similarity_search(mock_embedder):
    embedder_instance, mock_collection = mock_embedder
    mock_collection.query.return_value = {
        "documents": [["Trial NCT123 Phase 3"]],
        "metadatas": [[{"source": "clinicaltrials", "company": "TestPharma"}]],
        "distances": [[0.15]],
        "ids": [["id1"]]
    }

    retriever = Retriever(embedder=embedder_instance)
    results = retriever.retrieve_for_chat("test_collection", "What phase is Drug X in?")

    mock_collection.query.assert_called_once()
    assert len(results) == 1
    assert results[0]["text"] == "Trial NCT123 Phase 3"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_retriever.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/rag/retriever.py
from src.ingestion.embedder import Embedder, EMBEDDING_MODEL


class Retriever:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def retrieve_for_report(self, collection_name: str, company: str) -> list[dict]:
        collection = self._embedder.get_collection(collection_name)
        results = collection.get(
            where={"company": company},
            include=["documents", "metadatas"],
        )

        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({"text": doc, "metadata": meta})
        return chunks

    def retrieve_for_chat(self, collection_name: str, query: str, n_results: int = 10) -> list[dict]:
        collection = self._embedder.get_collection(collection_name)

        response = self._embedder._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        query_embedding = response.data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_retriever.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/rag/retriever.py tests/test_retriever.py
git commit -m "feat: RAG retriever with metadata filtering and similarity search"
```

---

## Task 7: Claude Generator

**Files:**
- Create: `src/rag/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write the failing tests**

```python
# tests/test_generator.py
import pytest
from unittest.mock import patch, MagicMock
from src.rag.generator import Generator


@pytest.fixture
def mock_anthropic():
    with patch("src.rag.generator.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        MockAnthropic.return_value = mock_client

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is a generated report about TestPharma.")]
        mock_client.messages.create.return_value = mock_message

        yield mock_client


def test_generate_report(mock_anthropic):
    generator = Generator(api_key="test-key")

    chunks = [
        {"text": "Clinical Trial NCT123 Phase 3 RECRUITING", "metadata": {"source": "clinicaltrials"}},
        {"text": "FDA Approval NDA456 TYGACIL", "metadata": {"source": "fda_approval"}},
    ]

    result = generator.generate_report("TestPharma", chunks)

    assert isinstance(result, str)
    assert len(result) > 0
    mock_anthropic.messages.create.assert_called_once()

    call_kwargs = mock_anthropic.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert any("TestPharma" in msg.get("content", "") for msg in call_kwargs["messages"])


def test_generate_chat_response(mock_anthropic):
    generator = Generator(api_key="test-key")

    chunks = [
        {"text": "Clinical Trial NCT123 Phase 3 RECRUITING", "metadata": {"source": "clinicaltrials"}},
    ]
    history = [
        {"role": "user", "content": "Tell me about TestPharma's pipeline"},
        {"role": "assistant", "content": "TestPharma has several trials..."},
    ]

    result = generator.generate_chat_response("What phase is Drug X?", chunks, history)

    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_report_with_no_data(mock_anthropic):
    generator = Generator(api_key="test-key")
    result = generator.generate_report("UnknownCo", [])

    assert isinstance(result, str)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_generator.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/rag/generator.py
from anthropic import Anthropic

MODEL = "claude-sonnet-4-5-20250929"

REPORT_SYSTEM_PROMPT = """You are a pharma due diligence analyst helping VC/PE investors evaluate pharmaceutical and biotech companies.

Given data from ClinicalTrials.gov and FDA databases, generate a structured due diligence report. Be factual, cite specific data points, and flag risks clearly.

Format the report as follows:
## Due Diligence Report: [Company/Drug]

### Pipeline Overview
Total active programs, breakdown by phase.

### Clinical Trials
Per-trial summary with phase, status, enrollment, key dates, endpoints.
Flag risks: terminated/suspended trials, delayed timelines.

### FDA / Regulatory
Approved products and indications.
Recent FDA actions.
Adverse event signal summary.

### Risk Assessment
Pipeline concentration risk, regulatory risks, competitive positioning.

### Sources
Links to ClinicalTrials.gov entries (https://clinicaltrials.gov/study/NCTXXXXXXXX) and FDA records.

If data is limited, say so clearly. Never fabricate information not present in the provided data."""

CHAT_SYSTEM_PROMPT = """You are a pharma due diligence analyst helping VC/PE investors. Answer questions based on the provided clinical trial and FDA data.

Rules:
- Only use information from the provided context
- Cite specific NCT IDs, application numbers, or data points
- If you don't have data to answer a question, say "I don't have data on that in the current dataset"
- Be concise and factual"""


class Generator:
    def __init__(self, api_key: str):
        self._client = Anthropic(api_key=api_key)

    def generate_report(self, company_or_drug: str, chunks: list[dict]) -> str:
        if not chunks:
            context = "No data was found for this query in ClinicalTrials.gov or FDA databases."
        else:
            context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=REPORT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a due diligence report for: {company_or_drug}\n\nData:\n{context}"
                }
            ],
        )
        return response.content[0].text

    def generate_chat_response(
        self,
        question: str,
        chunks: list[dict],
        history: list[dict],
    ) -> str:
        context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks) if chunks else "No relevant data found."

        messages = list(history) + [
            {
                "role": "user",
                "content": f"Context from database:\n{context}\n\nQuestion: {question}"
            }
        ]

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=CHAT_SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_generator.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/rag/generator.py tests/test_generator.py
git commit -m "feat: Claude-powered report and chat response generator"
```

---

## Task 8: Report Builder (Orchestrator)

**Files:**
- Create: `src/report/builder.py`
- Create: `tests/test_builder.py`

**Step 1: Write the failing tests**

```python
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
    mock_deps["ct_client"].search_by_sponsor.assert_called_once_with("TestPharma")
    mock_deps["fda_client"].search_approvals.assert_called_once()
    mock_deps["embedder"].embed_and_store.assert_called_once()
    mock_deps["generator"].generate_report.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_builder.py -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

```python
# src/report/builder.py
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator


def _sanitize_collection_name(name: str) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in name.lower())
    sanitized = sanitized.strip("_")[:50]
    if len(sanitized) < 3:
        sanitized = sanitized + "_co"
    return sanitized


class ReportBuilder:
    def __init__(
        self,
        ct_client: ClinicalTrialsClient,
        fda_client: FDAClient,
        chunker_cls: type = Chunker,
        embedder: Embedder = None,
        retriever: Retriever = None,
        generator: Generator = None,
    ):
        self.ct_client = ct_client
        self.fda_client = fda_client
        self.chunker_cls = chunker_cls
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator

    async def build_report(self, company_or_drug: str) -> str:
        collection_name = _sanitize_collection_name(company_or_drug)

        # 1. Fetch data from APIs
        trials = await self.ct_client.search_by_sponsor(company_or_drug)
        approvals = await self.fda_client.search_approvals(company_or_drug)

        # Get labels and adverse events for each approved drug
        drug_names = list({a["brand_name"] for a in approvals if a["brand_name"]})
        labels = []
        ae_summaries = []
        for drug_name in drug_names:
            drug_labels = await self.fda_client.search_labels(drug_name)
            labels.extend(drug_labels)
            ae_summary = await self.fda_client.get_adverse_events_summary(drug_name)
            ae_summaries.append((drug_name, ae_summary))

        # 2. Chunk all data
        all_chunks = []
        for trial in trials:
            all_chunks.extend(self.chunker_cls.chunk_clinical_trial(trial))
        for approval in approvals:
            all_chunks.extend(self.chunker_cls.chunk_fda_approval(approval))
        for label in labels:
            all_chunks.extend(self.chunker_cls.chunk_fda_label(label))
        for drug_name, ae_summary in ae_summaries:
            all_chunks.extend(self.chunker_cls.chunk_adverse_events(drug_name, ae_summary))

        if not all_chunks:
            return f"No data found for '{company_or_drug}' in ClinicalTrials.gov or FDA databases."

        # 3. Embed and store
        self.embedder.embed_and_store(all_chunks, collection_name=collection_name)

        # 4. Retrieve all chunks for report
        report_chunks = self.retriever.retrieve_for_report(collection_name, company_or_drug)

        # If metadata filter returns nothing, fall back to all chunks
        if not report_chunks:
            report_chunks = all_chunks

        # 5. Generate report
        return self.generator.generate_report(company_or_drug, report_chunks)

    @property
    def collection_name_for(self):
        return _sanitize_collection_name
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/test_builder.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add src/report/builder.py tests/test_builder.py
git commit -m "feat: report builder orchestrating API fetch, chunking, embedding, and generation"
```

---

## Task 9: Streamlit App

**Files:**
- Create: `app.py`

**Step 1: Write the Streamlit app**

```python
# app.py
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.report.builder import ReportBuilder

load_dotenv()

st.set_page_config(
    page_title="Pharma Due Diligence",
    page_icon="💊",
    layout="wide",
)

st.title("Pharma Due Diligence Chatbot")
st.caption("RAG-powered analysis using ClinicalTrials.gov and FDA data")


@st.cache_resource
def get_components():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    fda_key = os.getenv("OPENFDA_API_KEY")

    if not anthropic_key or not openai_key:
        st.error("Please set ANTHROPIC_API_KEY and OPENAI_API_KEY in your .env file")
        st.stop()

    ct_client = ClinicalTrialsClient()
    fda_client = FDAClient(api_key=fda_key)
    embedder = Embedder(openai_api_key=openai_key)
    retriever = Retriever(embedder=embedder)
    generator = Generator(api_key=anthropic_key)
    builder = ReportBuilder(
        ct_client=ct_client,
        fda_client=fda_client,
        chunker_cls=Chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
    return builder, retriever, generator


builder, retriever, generator = get_components()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Sidebar - Report Generation
with st.sidebar:
    st.header("Generate Report")
    company_input = st.text_input(
        "Company or Drug Name",
        placeholder="e.g., Moderna, Keytruda, Pfizer",
    )
    generate_btn = st.button("Generate Due Diligence Report", type="primary", use_container_width=True)

    if st.session_state.current_company:
        st.divider()
        st.success(f"Active: {st.session_state.current_company}")
        st.caption("Ask follow-up questions in the chat below.")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_company = None
        st.session_state.collection_name = None
        st.rerun()

# Report generation
if generate_btn and company_input:
    st.session_state.messages = []
    st.session_state.current_company = company_input

    with st.spinner(f"Fetching data and generating report for {company_input}..."):
        report = asyncio.run(builder.build_report(company_input))

    sanitized = "".join(c if c.isalnum() else "_" for c in company_input.lower()).strip("_")[:50]
    if len(sanitized) < 3:
        sanitized = sanitized + "_co"
    st.session_state.collection_name = sanitized

    st.session_state.messages.append({"role": "assistant", "content": report})
    st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a follow-up question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.collection_name:
            chunks = retriever.retrieve_for_chat(st.session_state.collection_name, prompt)
        else:
            chunks = []

        history = [
            msg for msg in st.session_state.messages[:-1]
            if msg["role"] in ("user", "assistant")
        ]

        with st.spinner("Thinking..."):
            response = generator.generate_chat_response(prompt, chunks, history)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
```

**Step 2: Manual smoke test**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && streamlit run app.py`

Verify:
- App loads without errors
- Sidebar has company input and generate button
- Chat input is visible at the bottom

(Stop the server with Ctrl+C after verifying.)

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit app with report generation and chat interface"
```

---

## Task 10: Integration Test & Final Verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
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

    # Mock embedder (don't actually call OpenAI)
    embedder = MagicMock(spec=Embedder)
    embedder.embed_and_store = MagicMock()
    mock_collection = MagicMock()
    embedder.get_collection.return_value = mock_collection

    # Retriever will use all_chunks fallback since mock metadata filter returns empty
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
    ct_client.search_by_sponsor.assert_called_once_with("IntegrationPharma")
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
```

**Step 2: Run all tests**

Run: `cd /Users/anyasikri/pharma-dd-chatbot && python -m pytest tests/ -v`
Expected: All tests pass (11+ tests)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: integration test for full report pipeline"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project scaffolding | - |
| 2 | ClinicalTrials.gov API client | 3 tests |
| 3 | openFDA API client | 4 tests |
| 4 | Document chunker | 4 tests |
| 5 | Embedder + ChromaDB | 2 tests |
| 6 | RAG retriever | 2 tests |
| 7 | Claude generator | 3 tests |
| 8 | Report builder (orchestrator) | 1 test |
| 9 | Streamlit app | manual smoke test |
| 10 | Integration test | 1 test |

**Total: 10 tasks, ~20 tests, 10 commits**
