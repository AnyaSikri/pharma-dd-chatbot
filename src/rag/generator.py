from anthropic import Anthropic

MODEL = "claude-sonnet-4-5-20250929"
MAX_HISTORY_MESSAGES = 20

REPORT_SYSTEM_PROMPT = """You are a pharma and medtech due diligence analyst helping VC/PE investors evaluate pharmaceutical, biotech, and medical device companies.

Given data from ClinicalTrials.gov and FDA databases (including device databases), generate a structured due diligence report. Be factual, cite specific data points, and flag risks clearly.

IMPORTANT: Each data chunk contains a "Source:" URL. You MUST include these exact URLs as inline links throughout the report wherever you reference that data. Every claim must be traceable to its source.

Format the report as follows:
## Due Diligence Report: [Company/Drug/Device]

### Pipeline Overview
Total active programs, breakdown by phase. Include both drug and device programs if present.

### Clinical Trials
Per-trial summary with phase, status, enrollment, key dates, endpoints.
Include the ClinicalTrials.gov link for each trial (e.g., [NCT12345678](https://clinicaltrials.gov/study/NCT12345678)).
Flag risks: terminated/suspended trials, delayed timelines.

### FDA / Regulatory — Drugs
Approved products and indications with FDA Drugs@FDA links.
Recent FDA actions.
Adverse event signal summary with FAERS link.
Include DailyMed links for label information.
Only include this section if drug data is present.

### FDA / Regulatory — Devices
510(k) clearances and PMA approvals with FDA links.
Device classification and advisory committee information.
MAUDE adverse event summary with links.
Recall history with status and reason.
Only include this section if device data is present.

### Risk Assessment
Pipeline concentration risk, regulatory risks, competitive positioning.
For devices: recall history risk, classification risk, post-market surveillance concerns.

### Sources
Consolidated list of all source URLs referenced in the report.

If data is limited, say so clearly. Never fabricate information not present in the provided data."""

CHAT_SYSTEM_PROMPT = """You are a pharma and medtech due diligence analyst helping VC/PE investors. Answer questions based on the provided clinical trial, FDA drug, and FDA device data.

Rules:
- Only use information from the provided context
- Cite specific NCT IDs, application numbers, and 510(k) numbers as clickable links using the Source URLs from the data
- Every factual claim must include its source link
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
            messages=[{
                "role": "user",
                "content": f"Generate a due diligence report for: {company_or_drug}\n\nData:\n{context}"
            }],
        )
        return response.content[0].text

    def generate_chat_response(self, question: str, chunks: list[dict], history: list[dict]) -> str:
        context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks) if chunks else "No relevant data found."
        recent_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
        messages = list(recent_history) + [{
            "role": "user",
            "content": f"Context from database:\n{context}\n\nQuestion: {question}"
        }]
        response = self._client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=CHAT_SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
