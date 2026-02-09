# Pharma Due Diligence Chatbot - Design Document

## Overview

A RAG-based Streamlit chatbot for VC/PE investors evaluating pharma/biotech companies for investment. Connects to ClinicalTrials.gov and openFDA APIs to provide structured due diligence reports and conversational follow-up with cited sources.

## User & Use Case

**Primary user:** VC/PE investor evaluating pharma/biotech companies for investment.

**Key questions they need answered:**
- What does this company's clinical pipeline look like?
- What phase are their trials in? Any terminated or delayed?
- What FDA approvals/actions exist for their drugs?
- Are there safety signals in adverse event data?
- What's the overall risk profile?

## Interaction Modes

1. **Report Mode** - User enters a company or drug name and receives a structured due diligence report covering pipeline, trials, FDA status, and risk assessment.
2. **Chat Mode** - User asks follow-up questions conversationally. RAG retrieves relevant context and Claude generates cited answers.

## Architecture

```
User (Streamlit)
  -> Query Router (report vs. conversational)
  -> Data Ingestion Layer (ClinicalTrials.gov API + openFDA API)
  -> Chunking + Embedding (OpenAI embeddings -> ChromaDB)
  -> RAG Retrieval (ChromaDB similarity search + metadata filtering)
  -> Claude API (answer generation with retrieved context)
  -> Response with citations -> User
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit (chat interface + report display) |
| LLM | Claude API (Anthropic) for generation |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local, persistent) |
| Data sources | ClinicalTrials.gov API v2, openFDA API |
| HTTP client | httpx (async) |
| Language | Python 3.12 |

## Data Sources

### ClinicalTrials.gov API (v2)

- **Base URL:** `https://clinicaltrials.gov/api/v2/studies`
- **Search by:** company name (sponsor), drug name, NCT ID, condition/disease area
- **Key fields:** NCT ID, title, phase, status, enrollment, start/completion dates, primary endpoints, sponsor, conditions, interventions, results
- **Pagination:** up to 1000 results per request
- **Auth:** none required

### openFDA API

- **Drug labeling:** `https://api.fda.gov/drug/label.json` - approved indications, warnings, boxed warnings
- **Adverse events:** `https://api.fda.gov/drug/event.json` - FAERS safety signals
- **Drug approvals:** `https://api.fda.gov/drug/drugsfda.json` - NDA/BLA approval history
- **Key fields:** brand/generic name, approval date, application number, active ingredients, adverse event counts by seriousness
- **Rate limit:** 240 req/min with free API key, 40/min without

### Ingestion Flow

1. User enters a company or drug name
2. System queries both APIs in parallel
3. Raw responses are parsed into structured documents (one per trial, one per FDA record)
4. Documents are chunked by logical sections and embedded
5. Chunks stored in ChromaDB with metadata (source, company, drug, phase, date)

## RAG Pipeline

### Chunking Strategy

- **Clinical trials:** one chunk per trial (title + phase + status + endpoints + results). Large trials with results split into overview + results chunks.
- **FDA records:** one chunk per label section (indications, warnings, adverse events). One chunk per approval action.
- **Metadata per chunk:** `{source, company, drug_name, phase, status, date, nct_id | application_number}`

### Retrieval Strategy

- **Report generation:** retrieve ALL chunks for the target company/drug via metadata filter (not just similarity)
- **Chat follow-ups:** hybrid - metadata filter to current company context + similarity search on the question
- Claude is prompted to say "I don't have data on that" rather than hallucinate

## Report Template

```
## Due Diligence Report: [Company/Drug]

### Pipeline Overview
- Total active programs, breakdown by phase

### Clinical Trials
- Per-trial summary: phase, status, enrollment, key dates, endpoints
- Flagged risks: trials terminated/suspended, delayed timelines

### FDA / Regulatory
- Approved products and indications
- Recent FDA actions (approvals, supplements, warnings)
- Adverse event signal summary

### Risk Assessment
- Pipeline concentration risk
- Regulatory risks
- Competitive positioning based on available data

### Sources
- Links to ClinicalTrials.gov entries and FDA records
```

## Project Structure

```
pharma-dd-chatbot/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── clinical_trials.py   # ClinicalTrials.gov API client
│   │   └── fda.py               # openFDA API client
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py           # Document chunking logic
│   │   └── embedder.py          # Embedding + ChromaDB storage
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py         # ChromaDB retrieval
│   │   └── generator.py         # Claude API calls with RAG context
│   └── report/
│       ├── __init__.py
│       └── builder.py           # Structured report generation
├── chroma_db/                   # Local ChromaDB persistence
└── tests/
    ├── test_clinical_trials.py
    ├── test_fda.py
    └── test_rag.py
```

## Key Dependencies

- `streamlit` - UI
- `anthropic` - Claude API
- `openai` - embeddings only
- `chromadb` - vector store
- `httpx` - async API calls
- `python-dotenv` - env management

## Error Handling

- **API failures:** retry with exponential backoff, surface clear error messages
- **No results:** tell user clearly rather than generating empty reports
- **ChromaDB:** persist to disk, cache results per company to avoid redundant API calls

## Future Enhancements (Not V1)

- News sources (NewsAPI, SEC EDGAR filings)
- PitchBook/Crunchbase integration for funding data
- Comparative analysis across multiple companies
- Export reports to PDF
