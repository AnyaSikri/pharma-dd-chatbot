from __future__ import annotations


class Chunker:
    @staticmethod
    def chunk_clinical_trial(trial: dict) -> list[dict]:
        outcomes_text = ""
        for outcome in trial.get("primary_outcomes", []):
            outcomes_text += f"  - {outcome['measure']} ({outcome.get('timeFrame', 'N/A')})\n"
        interventions_text = ", ".join(i["name"] for i in trial.get("interventions", []))
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
        return [{"text": text, "metadata": {
            "source": "clinicaltrials",
            "nct_id": trial["nct_id"],
            "company": trial["sponsor"],
            "drug_name": interventions_text,
            "phase": trial["phase"],
            "status": trial["status"],
            "date": trial.get("start_date", ""),
        }}]

    @staticmethod
    def chunk_fda_approval(approval: dict) -> list[dict]:
        products_text = ""
        for p in approval.get("products", []):
            ingredients = ", ".join(
                f"{i['name']} {i.get('strength', '')}"
                for i in p.get("active_ingredients", [])
            )
            products_text += (
                f"  - {p.get('brand_name', '')} "
                f"({p.get('dosage_form', '')}, {p.get('route', '')}): "
                f"{ingredients}\n"
            )
        submissions_text = ""
        for s in approval.get("submissions", []):
            date_raw = s.get("submission_status_date", "")
            date_fmt = (
                f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
                if len(date_raw) == 8
                else date_raw
            )
            submissions_text += (
                f"  - {s.get('submission_type', '')} "
                f"({s.get('submission_class_code_description', '')}): "
                f"{s.get('submission_status', '')} on {date_fmt}\n"
            )
        text = (
            f"FDA Drug Application: {approval['brand_name']} ({approval['generic_name']})\n"
            f"Application Number: {approval['application_number']}\n"
            f"Manufacturer: {approval.get('manufacturer', '')}\n"
            f"Products:\n{products_text}"
            f"Submissions:\n{submissions_text}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_approval",
            "drug_name": approval["brand_name"],
            "company": approval.get("manufacturer", ""),
            "application_number": approval["application_number"],
        }}]

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
                f"FDA Label - {drug_name} ({label.get('generic_name', '')}) "
                f"- {section_name.replace('_', ' ').title()}\n\n{content}"
            )
            chunks.append({"text": text, "metadata": {
                "source": "fda_label",
                "drug_name": drug_name,
                "company": company,
                "section": section_name,
            }})
        if not chunks:
            chunks = [{"text": (
                f"FDA Label for {drug_name}: "
                "No detailed label information available."
            ), "metadata": {
                "source": "fda_label",
                "drug_name": drug_name,
                "company": company,
                "section": "summary",
            }}]
        return chunks

    @staticmethod
    def chunk_adverse_events(drug_name: str, ae_summary: dict) -> list[dict]:
        reactions_text = ", ".join(ae_summary.get("sample_reactions", []))
        text = (
            f"Adverse Events Summary for {drug_name}\n"
            f"Total FAERS Reports: {ae_summary['total_reports']:,}\n"
            f"Serious Reports in Sample: {ae_summary.get('serious_count', 0)}\n"
            f"Common Reactions: {reactions_text}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_adverse_events",
            "drug_name": drug_name,
        }}]
