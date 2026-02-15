# src/api/sec_edgar.py
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


class SECEdgarClient:
    """Async client for SEC EDGAR filings and yfinance market data."""

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, user_agent: Optional[str] = None):
        ua = user_agent or "Pharma DD Chatbot contact@example.com"
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
        )
        self._tickers_cache: Optional[dict] = None

    async def _load_tickers(self) -> dict:
        if self._tickers_cache is not None:
            return self._tickers_cache
        response = await self._client.get(self.TICKERS_URL)
        response.raise_for_status()
        self._tickers_cache = response.json()
        return self._tickers_cache

    async def lookup_company(self, query: str) -> Optional[dict]:
        """Find a company by ticker or name. Returns {cik, ticker, name} or None."""
        tickers = await self._load_tickers()
        query_upper = query.upper().strip()
        query_lower = query.lower().strip()

        # Pass 1: exact ticker match
        for entry in tickers.values():
            if entry.get("ticker", "").upper() == query_upper:
                cik = str(entry["cik_str"]).zfill(10)
                return {"cik": cik, "ticker": entry["ticker"], "name": entry["title"]}

        # Pass 2: company name contains query
        for entry in tickers.values():
            if query_lower in entry.get("title", "").lower():
                cik = str(entry["cik_str"]).zfill(10)
                return {"cik": cik, "ticker": entry["ticker"], "name": entry["title"]}

        return None

    async def get_filings(self, cik: str, filing_types: list[str] = None,
                          limit: int = 10) -> list[dict]:
        """Get recent filings for a CIK. Filters to 10-K, 10-Q, 8-K by default."""
        if filing_types is None:
            filing_types = ["10-K", "10-Q", "8-K"]

        url = self.SUBMISSIONS_URL.format(cik=cik.zfill(10))
        response = await self._client.get(url)
        response.raise_for_status()
        data = response.json()

        company_name = data.get("name", "")
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        results = []
        for i in range(len(forms)):
            if forms[i] not in filing_types:
                continue
            accession_clean = accessions[i].replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}"
            )
            results.append({
                "form_type": forms[i],
                "filing_date": dates[i],
                "accession_number": accessions[i],
                "primary_document": primary_docs[i],
                "description": descriptions[i] if i < len(descriptions) else "",
                "filing_url": filing_url,
                "company_name": company_name,
            })
            if len(results) >= limit:
                break

        return results

    async def get_company_facts(self, cik: str) -> dict:
        """Get XBRL financial facts: revenue, net income, assets, etc."""
        url = self.COMPANY_FACTS_URL.format(cik=cik.zfill(10))
        response = await self._client.get(url)
        response.raise_for_status()
        data = response.json()

        company_name = data.get("entityName", "")
        us_gaap = data.get("facts", {}).get("us-gaap", {})

        def _latest_annual(concept_name: str) -> Optional[dict]:
            concept = us_gaap.get(concept_name, {})
            units = concept.get("units", {})
            values = units.get("USD", units.get("USD/shares", units.get("shares", [])))
            if not values:
                return None
            annual = [v for v in values if v.get("form") == "10-K"]
            if not annual:
                annual = values
            annual.sort(key=lambda v: v.get("end", ""), reverse=True)
            if annual:
                return {"value": annual[0].get("val"), "period_end": annual[0].get("end"),
                        "form": annual[0].get("form", "")}
            return None

        concepts = {
            "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "total_assets": ["Assets"],
            "total_liabilities": ["Liabilities"],
            "stockholders_equity": ["StockholdersEquity",
                                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
            "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue",
                                     "CashCashEquivalentsAndShortTermInvestments"],
            "total_debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
            "research_and_development": ["ResearchAndDevelopmentExpense"],
            "operating_income": ["OperatingIncomeLoss"],
            "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
        }

        facts = {"company_name": company_name}
        for key, concept_names in concepts.items():
            for name in concept_names:
                result = _latest_annual(name)
                if result is not None:
                    facts[key] = result
                    break

        return facts

    async def get_market_data(self, ticker: str) -> Optional[dict]:
        """Fetch real-time market data from yfinance (runs in thread executor)."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(_EXECUTOR, self._fetch_yfinance, ticker)
        except Exception:
            return None

    @staticmethod
    def _fetch_yfinance(ticker: str) -> Optional[dict]:
        try:
            import yfinance as yf
        except ImportError:
            return None

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or info.get("regularMarketPrice") is None:
                return None
            return {
                "ticker": ticker.upper(),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "total_revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margins": info.get("grossMargins"),
                "operating_margins": info.get("operatingMargins"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "free_cash_flow": info.get("freeCashflow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "full_time_employees": info.get("fullTimeEmployees"),
                "beta": info.get("beta"),
            }
        except Exception:
            return None

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
