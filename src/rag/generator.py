from anthropic import Anthropic

MODEL = "claude-sonnet-4-5-20250929"
MAX_HISTORY_MESSAGES = 20

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
