# src/report/builder.py
import logging
from src.api.clinical_trials import ClinicalTrialsClient
from src.api.fda import FDAClient
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator

logger = logging.getLogger(__name__)


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

    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        return _sanitize_collection_name(name)

    async def build_report(self, company_or_drug: str) -> str:
        collection_name = _sanitize_collection_name(company_or_drug)
        errors = []

        # 1. Fetch data from APIs (continue on partial failure)
        # Search by both sponsor name and drug/intervention name for broader coverage
        trials = []
        try:
            sponsor_trials = await self.ct_client.search_by_sponsor(company_or_drug)
            trials.extend(sponsor_trials)
        except Exception as e:
            logger.error("ClinicalTrials.gov sponsor search error: %s", e)
            errors.append(f"ClinicalTrials.gov sponsor lookup failed: {e}")

        try:
            drug_trials = await self.ct_client.search_by_drug(company_or_drug)
            # Deduplicate by NCT ID
            existing_ids = {t["nct_id"] for t in trials}
            for t in drug_trials:
                if t["nct_id"] not in existing_ids:
                    trials.append(t)
                    existing_ids.add(t["nct_id"])
        except Exception as e:
            logger.error("ClinicalTrials.gov drug search error: %s", e)
            errors.append(f"ClinicalTrials.gov drug lookup failed: {e}")

        approvals = []
        try:
            approvals = await self.fda_client.search_approvals(company_or_drug)
        except Exception as e:
            logger.error("openFDA approvals API error: %s", e)
            errors.append(f"FDA approvals lookup failed: {e}")

        # Get labels and adverse events for each approved drug
        drug_names = list({a["brand_name"] for a in approvals if a["brand_name"]})
        labels = []
        ae_summaries = []
        for drug_name in drug_names:
            try:
                drug_labels = await self.fda_client.search_labels(drug_name)
                labels.extend(drug_labels)
            except Exception as e:
                logger.error("openFDA labels API error for %s: %s", drug_name, e)
                errors.append(f"FDA labels lookup failed for {drug_name}: {e}")
            try:
                ae_summary = await self.fda_client.get_adverse_events_summary(drug_name)
                ae_summaries.append((drug_name, ae_summary))
            except Exception as e:
                logger.error("openFDA adverse events API error for %s: %s", drug_name, e)
                errors.append(f"FDA adverse events lookup failed for {drug_name}: {e}")

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
            error_detail = "\n".join(errors) if errors else ""
            msg = f"No data found for '{company_or_drug}' in ClinicalTrials.gov or FDA databases."
            if error_detail:
                msg += f"\n\nErrors encountered:\n{error_detail}"
            return msg

        # 3. Embed and store
        self.embedder.embed_and_store(all_chunks, collection_name=collection_name)

        # 4. Retrieve all chunks for report
        report_chunks = self.retriever.retrieve_for_report(collection_name, company_or_drug)

        # If retrieval returns nothing, fall back to all chunks
        if not report_chunks:
            report_chunks = all_chunks

        # 5. Generate report
        report = self.generator.generate_report(company_or_drug, report_chunks)

        if errors:
            report += "\n\n---\n*Note: Some data sources were unavailable: " + "; ".join(errors) + "*"

        return report

    async def close(self):
        await self.ct_client.close()
        await self.fda_client.close()
