from __future__ import annotations


class Chunker:
    @staticmethod
    def chunk_clinical_trial(trial: dict) -> list[dict]:
        outcomes_text = ""
        for outcome in trial.get("primary_outcomes", []):
            outcomes_text += f"  - {outcome['measure']} ({outcome.get('timeFrame', 'N/A')})\n"
        interventions_text = ", ".join(i["name"] for i in trial.get("interventions", []))
        nct_id = trial["nct_id"]
        source_url = f"https://clinicaltrials.gov/study/{nct_id}"
        text = (
            f"Clinical Trial: {trial['title']}\n"
            f"NCT ID: {nct_id}\n"
            f"Source: {source_url}\n"
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
            "source_url": source_url,
            "nct_id": nct_id,
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
        app_num = approval["application_number"]
        app_digits = "".join(c for c in app_num if c.isdigit())
        source_url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={app_digits}"
        text = (
            f"FDA Drug Application: {approval['brand_name']} ({approval['generic_name']})\n"
            f"Application Number: {app_num}\n"
            f"Source: {source_url}\n"
            f"Manufacturer: {approval.get('manufacturer', '')}\n"
            f"Products:\n{products_text}"
            f"Submissions:\n{submissions_text}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_approval",
            "source_url": source_url,
            "drug_name": approval["brand_name"],
            "company": approval.get("manufacturer", ""),
            "application_number": app_num,
        }}]

    @staticmethod
    def chunk_fda_label(label: dict) -> list[dict]:
        chunks = []
        drug_name = label.get("brand_name", "Unknown")
        company = label.get("manufacturer", "")
        source_url = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={drug_name.replace(' ', '+')}"
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
                f"- {section_name.replace('_', ' ').title()}\n"
                f"Source: {source_url}\n\n{content}"
            )
            chunks.append({"text": text, "metadata": {
                "source": "fda_label",
                "source_url": source_url,
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
                "source_url": source_url,
                "drug_name": drug_name,
                "company": company,
                "section": "summary",
            }}]
        return chunks

    @staticmethod
    def chunk_device_clearance(clearance: dict) -> list[dict]:
        k_number = clearance.get("k_number", "")
        source_url = f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID={k_number}"
        decision_raw = clearance.get("decision_date", "")
        decision_fmt = (
            f"{decision_raw[:4]}-{decision_raw[4:6]}-{decision_raw[6:]}"
            if len(decision_raw) == 8
            else decision_raw
        )
        text = (
            f"FDA Device 510(k) Clearance: {clearance['device_name']}\n"
            f"510(k) Number: {k_number}\n"
            f"Source: {source_url}\n"
            f"Applicant: {clearance.get('applicant', '')}\n"
            f"Decision: {clearance.get('decision_description', '')} on {decision_fmt}\n"
            f"Clearance Type: {clearance.get('clearance_type', '')}\n"
            f"Product Code: {clearance.get('product_code', '')}\n"
            f"Advisory Committee: {clearance.get('advisory_committee_description', '')}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_device_clearance",
            "source_url": source_url,
            "device_name": clearance.get("device_name", ""),
            "company": clearance.get("applicant", ""),
            "clearance_number": k_number,
        }}]

    @staticmethod
    def chunk_device_recalls(company: str, recalls: list[dict]) -> list[dict]:
        source_url = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm"
        recall_lines = []
        for r in recalls:
            recall_lines.append(
                f"  - {r.get('product_description', 'Unknown product')}: "
                f"{r.get('reason_for_recall', 'No reason listed')} "
                f"(Status: {r.get('status', 'N/A')})"
            )
        text = (
            f"FDA Device Recalls for {company}\n"
            f"Source: {source_url}\n"
            f"Total Recalls: {len(recalls)}\n"
            + "\n".join(recall_lines)
        )
        return [{"text": text, "metadata": {
            "source": "fda_device_recall",
            "source_url": source_url,
            "company": company,
        }}]

    @staticmethod
    def chunk_device_adverse_events(device_name: str, ae_summary: dict) -> list[dict]:
        source_url = "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm"
        events_text = ""
        for i, event in enumerate(ae_summary.get("sample_events", []), 1):
            events_text += f"  {i}. {event}\n"
        text = (
            f"MAUDE Adverse Events Summary for {device_name}\n"
            f"Source: {source_url}\n"
            f"Total MAUDE Reports: {ae_summary.get('total_reports', 0):,}\n"
            f"Serious Reports (Death/Injury) in Sample: {ae_summary.get('serious_count', 0)}\n"
            f"Sample Event Narratives:\n{events_text}" if events_text else
            f"MAUDE Adverse Events Summary for {device_name}\n"
            f"Source: {source_url}\n"
            f"Total MAUDE Reports: {ae_summary.get('total_reports', 0):,}\n"
            f"Serious Reports (Death/Injury) in Sample: {ae_summary.get('serious_count', 0)}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_device_events",
            "source_url": source_url,
            "device_name": device_name,
        }}]

    @staticmethod
    def chunk_sec_filings(company_name: str, filings: list[dict]) -> list[dict]:
        if not filings:
            return []
        filing_lines = []
        for f in filings:
            filing_lines.append(
                f"  - {f['form_type']} filed {f['filing_date']}: "
                f"{f.get('description', 'N/A')} "
                f"[View Filing]({f['filing_url']})"
            )
        source_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={company_name.replace(' ', '+')}&CIK=&type=&dateb=&owner=include&count=40&search_text=&action=getcompany"
        text = (
            f"SEC EDGAR Filings for {company_name}\n"
            f"Source: {source_url}\n"
            f"Recent Filings ({len(filings)}):\n"
            + "\n".join(filing_lines)
        )
        return [{"text": text, "metadata": {
            "source": "sec_filings",
            "source_url": source_url,
            "company": company_name,
        }}]

    @staticmethod
    def chunk_company_financials(company_name: str, facts: dict, market_data: dict = None) -> list[dict]:
        chunks = []

        def _fmt_dollars(val_dict):
            if val_dict is None:
                return "N/A"
            val = val_dict.get("value")
            period = val_dict.get("period_end", "")
            if val is None:
                return "N/A"
            if abs(val) >= 1_000_000_000:
                return f"${val / 1_000_000_000:.2f}B (period ending {period})"
            elif abs(val) >= 1_000_000:
                return f"${val / 1_000_000:.1f}M (period ending {period})"
            return f"${val:,.0f} (period ending {period})"

        xbrl_lines = []
        for label, key in [
            ("Revenue", "revenue"),
            ("Net Income", "net_income"),
            ("Total Assets", "total_assets"),
            ("Total Liabilities", "total_liabilities"),
            ("Stockholders' Equity", "stockholders_equity"),
            ("Cash & Equivalents", "cash_and_equivalents"),
            ("Total Debt", "total_debt"),
            ("R&D Expense", "research_and_development"),
            ("Operating Income", "operating_income"),
        ]:
            if key in facts:
                xbrl_lines.append(f"  {label}: {_fmt_dollars(facts[key])}")

        if facts.get("eps"):
            eps_val = facts["eps"].get("value")
            eps_period = facts["eps"].get("period_end", "")
            if eps_val is not None:
                xbrl_lines.append(f"  EPS (Diluted): ${eps_val:.2f} (period ending {eps_period})")

        if xbrl_lines:
            text = (
                f"SEC XBRL Financial Data for {company_name}\n"
                f"Source: https://data.sec.gov/api/xbrl/companyfacts/\n"
                f"Key Financial Metrics (from annual filings):\n"
                + "\n".join(xbrl_lines)
            )
            chunks.append({"text": text, "metadata": {
                "source": "sec_financials",
                "source_url": "https://data.sec.gov/api/xbrl/companyfacts/",
                "company": company_name,
            }})

        if market_data:
            def _fmt(val, fmt="dollar"):
                if val is None:
                    return "N/A"
                if fmt == "dollar":
                    if abs(val) >= 1_000_000_000:
                        return f"${val / 1_000_000_000:.2f}B"
                    elif abs(val) >= 1_000_000:
                        return f"${val / 1_000_000:.1f}M"
                    return f"${val:,.2f}"
                elif fmt == "percent":
                    return f"{val * 100:.1f}%"
                elif fmt == "ratio":
                    return f"{val:.2f}"
                return str(val)

            ticker = market_data.get("ticker", "")
            source_url = f"https://finance.yahoo.com/quote/{ticker}"
            lines = [
                f"  Ticker: {ticker}",
                f"  Current Price: {_fmt(market_data.get('current_price'))}",
                f"  Market Cap: {_fmt(market_data.get('market_cap'))}",
                f"  Enterprise Value: {_fmt(market_data.get('enterprise_value'))}",
                f"  Trailing P/E: {_fmt(market_data.get('trailing_pe'), 'ratio')}",
                f"  Forward P/E: {_fmt(market_data.get('forward_pe'), 'ratio')}",
                f"  Revenue Growth: {_fmt(market_data.get('revenue_growth'), 'percent')}",
                f"  Gross Margins: {_fmt(market_data.get('gross_margins'), 'percent')}",
                f"  Operating Margins: {_fmt(market_data.get('operating_margins'), 'percent')}",
                f"  Total Cash: {_fmt(market_data.get('total_cash'))}",
                f"  Total Debt: {_fmt(market_data.get('total_debt'))}",
                f"  Free Cash Flow: {_fmt(market_data.get('free_cash_flow'))}",
                f"  52-Week High: {_fmt(market_data.get('fifty_two_week_high'))}",
                f"  52-Week Low: {_fmt(market_data.get('fifty_two_week_low'))}",
                f"  Beta: {_fmt(market_data.get('beta'), 'ratio')}",
                f"  Sector: {market_data.get('sector', 'N/A')}",
                f"  Industry: {market_data.get('industry', 'N/A')}",
                f"  Employees: {market_data.get('full_time_employees', 'N/A')}",
            ]
            text = (
                f"Market Data for {company_name} ({ticker})\n"
                f"Source: {source_url}\n"
                + "\n".join(lines)
            )
            chunks.append({"text": text, "metadata": {
                "source": "market_data",
                "source_url": source_url,
                "company": company_name,
                "ticker": ticker,
            }})

        return chunks

    @staticmethod
    def chunk_adverse_events(drug_name: str, ae_summary: dict) -> list[dict]:
        reactions_text = ", ".join(ae_summary.get("sample_reactions", []))
        source_url = f"https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
        text = (
            f"Adverse Events Summary for {drug_name}\n"
            f"Source: {source_url}\n"
            f"Total FAERS Reports: {ae_summary['total_reports']:,}\n"
            f"Serious Reports in Sample: {ae_summary.get('serious_count', 0)}\n"
            f"Common Reactions: {reactions_text}"
        )
        return [{"text": text, "metadata": {
            "source": "fda_adverse_events",
            "source_url": source_url,
            "drug_name": drug_name,
        }}]
