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
