"""
Module B: LLM Prospecting Agent (LangChain)
Synthesizes structured corporate balance sheet signals and unstructured market news
into executive-ready prospecting dossiers and high-impact talking points.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL
)

class StrategicProspectingDossier(BaseModel):
    """Structured Pydantic schema for LangChain prospecting synthesis output."""
    executive_summary: str = Field(
        description="2-3 sentence strategic executive brief synthesizing financial trajectory, solvency, and market momentum."
    )
    market_signals: List[str] = Field(
        description="2-3 actionable market events or catalyst signals extracted from news and corporate filings."
    )
    outreach_talking_points: List[str] = Field(
        description="3 hyper-personalized conversation openers and strategic questions tailored for C-suite buyers."
    )
    recommended_pitch_angle: str = Field(
        description="The primary value proposition angle (e.g., Cost Optimization, Scale Enablement, Risk Mitigation)."
    )
    potential_objection_handler: str = Field(
        description="Anticipated buyer hesitation given their financial posture and the proactive rebuttal strategy."
    )

SYSTEM_PROMPT = """You are a Principal Enterprise Prospecting Strategist and Corporate Intelligence Specialist at a tier-1 institutional B2B sales advisory firm.

Your mission is to evaluate corporate leads by synthesizing:
1. Hard quantitative financial health metrics (Revenue velocity, debt-to-equity leverage, liquidity, Altman Z-Score credit proxy).
2. Soft qualitative market intelligence (Recent news events, leadership transitions, capital raises, M&A filings).
3. The Machine Learning Lead Probability Score (0-100) indicating propensity to convert.

You must generate an elite, concise, and razor-sharp 'Prospecting Summary' that equips sales directors with immediate strategic context and high-conviction outreach talking points.

Guidelines:
- Never regurgitate raw metrics without business interpretation. If Debt-to-Equity is elevated, identify the operational pressure. If Revenue Velocity is surging, highlight capacity bottlenecks.
- Ground talking points in the executive's specific mandate (e.g., newly appointed CRO, M&A integration, federal compliance deadlines).
- Ensure outreach talking points sound like a consultative peer advisor, not a cold salesperson.
- Return output strictly formatted according to the provided schema.
"""

HUMAN_PROMPT_TEMPLATE = """Analyze the following institutional target account and generate a strategic prospecting dossier:

### TARGET PROFILE:
- Company Name: {company_name} ({ticker})
- Sector / Sub-Sector: {sector} | {sub_sector}
- Headquarters: {headquarters} | Employees: {employee_count:,}
- Machine Learning Lead Score: {lead_probability_score}/100 ({priority_tier})

### FINANCIAL HEALTH & BALANCE SHEET DIAGNOSTICS:
- Annual Revenue: ${annual_revenue_m:,.2f}M (YoY Velocity: {revenue_velocity_pct:.1f}%, 3-Yr CAGR: {revenue_cagr_pct:.1f}%)
- EBITDA: ${ebitda_m:,.2f}M | Net Margin: {net_margin_pct:.1f}% | Free Cash Flow: ${free_cash_flow_m:,.2f}M
- Debt-to-Equity Ratio: {debt_to_equity:.2f} | Current Ratio: {current_ratio:.2f} | Quick Ratio: {quick_ratio:.2f}
- Altman Z-Score Credit Proxy: {altman_z_score:.2f} ({credit_risk_tier})

### MARKET INTELLIGENCE & EVENT TRIGGERS:
- Event Catalyst: {event_type}
- News Headline: "{headline}"
- Event Context: {narrative_summary}
- Signal Sentiment Score: {signal_sentiment_score:.2f}

{format_instructions}
"""

class LLMProspectingAgent:
    """
    LangChain-powered prospecting agent with multi-backend orchestration
    and resilient deterministic fallback synthesis.
    """

    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=StrategicProspectingDossier)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT_TEMPLATE),
        ])
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initializes LangChain LLM backend based on available API keys."""
        # 1. Google Gemini via langchain_google_genai
        if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("your-"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.2,
                    max_retries=2
                )
            except Exception as e:
                print(f"[WARN] Failed to initialize Gemini LLM: {e}")

        # 2. Groq via langchain_groq
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name=GROQ_MODEL,
                    groq_api_key=GROQ_API_KEY,
                    temperature=0.2
                )
            except Exception as e:
                print(f"[WARN] Failed to initialize Groq LLM: {e}")

        print("[INFO] Operating with resilient institutional heuristic synthesizer.")
        return None

    def synthesize_lead(self, row: Dict[str, Any]) -> StrategicProspectingDossier:
        """
        Synthesizes a single corporate lead into a strategic dossier.
        """
        prompt_inputs = {
            "company_name": row.get("company_name", "Enterprise Corp"),
            "ticker": row.get("ticker", "N/A"),
            "sector": row.get("sector", "Enterprise"),
            "sub_sector": row.get("sub_sector", "General"),
            "headquarters": row.get("headquarters", "Global"),
            "employee_count": int(row.get("employee_count", 1000)),
            "lead_probability_score": row.get("lead_probability_score", 50.0),
            "priority_tier": row.get("priority_tier", "Tier 2"),
            "annual_revenue_m": float(row.get("annual_revenue_m", 100.0)),
            "revenue_velocity_pct": float(row.get("revenue_velocity_yoy", 0.0)) * 100.0,
            "revenue_cagr_pct": float(row.get("revenue_cagr_3yr", 0.0)) * 100.0,
            "ebitda_m": float(row.get("ebitda_m", 15.0)),
            "net_margin_pct": float(row.get("net_margin", 0.08)) * 100.0,
            "free_cash_flow_m": float(row.get("free_cash_flow_m", 10.0)),
            "debt_to_equity": float(row.get("debt_to_equity", 1.0)),
            "current_ratio": float(row.get("current_ratio", 1.5)),
            "quick_ratio": float(row.get("quick_ratio", 1.2)),
            "altman_z_score": float(row.get("altman_z_score", 3.0)),
            "credit_risk_tier": str(row.get("credit_risk_tier", "Moderate Risk")),
            "event_type": row.get("event_type", "Operational Update"),
            "headline": row.get("headline", "Corporate developments"),
            "narrative_summary": row.get("narrative_summary", "Company continues strategic initiatives."),
            "signal_sentiment_score": float(row.get("signal_sentiment_score", 0.5)),
            "format_instructions": self.parser.get_format_instructions()
        }

        # Attempt LLM call if client initialized
        if self.llm is not None:
            try:
                formatted_prompt = self.prompt.format_prompt(**prompt_inputs)
                response = self.llm.invoke(formatted_prompt.to_messages())
                parsed = self.parser.parse(response.content)
                return parsed
            except Exception as e:
                print(f"[WARN] LLM invocation failed for {row.get('company_name')}: {e}. Falling back to heuristic synthesizer.")

        # Resilient institutional deterministic synthesis
        return self._generate_heuristic_dossier(prompt_inputs)

    def _generate_heuristic_dossier(self, p: Dict[str, Any]) -> StrategicProspectingDossier:
        """
        Deterministic, rule-based institutional synthesis engine ensuring 100% reliability.
        """
        comp = p["company_name"]
        score = p["lead_probability_score"]
        rev_vel = p["revenue_velocity_pct"]
        z_score = p["altman_z_score"]
        event = p["event_type"]
        headline = p["headline"]

        if score >= 75.0:
            summary = (
                f"{comp} presents an institutional Tier-1 target exhibiting robust top-line momentum "
                f"({rev_vel:.1f}% YoY growth) supported by an elite Altman Z-Score of {z_score:.2f}. "
                f"Recent catalyst '{headline}' signals active capital allocation toward technology modernization."
            )
            pitch_angle = "Scalable Enterprise Infrastructure & Rapid Time-to-Value Acceleration"
            talking_points = [
                f"Congratulate executive leadership on recent milestone: '{event}', referencing their aggressive {rev_vel:.1f}% YoY expansion.",
                f"Inquire how their architecture is absorbing surging transaction volumes while maintaining {p['net_margin_pct']:.1f}% net margins.",
                f"Propose our benchmark framework to de-risk enterprise rollout and shorten procurement lead-times by 40%."
            ]
            objection = "Prioritizing existing internal engineering bandwidth over external vendors; position as an immediate capability multiplier with zero core-refactoring."
        elif score >= 55.0:
            summary = (
                f"{comp} exhibits solid mid-tier solvency (Z-Score: {z_score:.2f}) with steady growth trajectory. "
                f"Catalyst '{event}' reveals operational focus on margin expansion and workflow consolidation."
            )
            pitch_angle = "Operational Efficiency & Unit-Economic Margin Optimization"
            talking_points = [
                f"Reference '{headline}' and explore how leadership plans to streamline cross-functional overhead.",
                f"Highlight our solution's ability to drive 2.5x ROI within 90 days against their current ${p['annual_revenue_m']:.1f}M revenue base.",
                f"Offer a targeted pilot phase to validate integration with minimal upfront capital expenditure."
            ]
            objection = "Budget scrutiny from corporate treasury; lead with measurable cost-reduction guarantees and flexible milestone billing."
        else:
            summary = (
                f"{comp} demonstrates elevated financial headwinds (Z-Score: {z_score:.2f}, D/E: {p['debt_to_equity']:.2f}) "
                f"prompting defensive cost containment. Catalyst '{event}' highlights operational restructuring."
            )
            pitch_angle = "Direct Opex Reduction & Risk Remediation"
            talking_points = [
                f"Acknowledge recent industry headwinds noted in '{headline}' and focus on rapid overhead containment.",
                f"Demonstrate how our automated tooling cuts legacy licensing costs by up to 35% without workforce expansion.",
                f"Position a performance-contingent proof-of-concept focused purely on liquidity preservation."
            ]
            objection = "Complete capital freeze on new vendor contracts; present as a budget-neutral consolidation that replaces multiple redundant legacy tools."

        signals = [
            f"Event Catalyst: {event} ({p['narrative_summary'][:90]}...)",
            f"Financial Trajectory: YoY Revenue Velocity at {rev_vel:.1f}%, FCF at ${p['free_cash_flow_m']:.1f}M",
            f"Credit Stance: Altman Z-Score {z_score:.2f} ({p['credit_risk_tier']})"
        ]

        return StrategicProspectingDossier(
            executive_summary=summary,
            market_signals=signals,
            outreach_talking_points=talking_points,
            recommended_pitch_angle=pitch_angle,
            potential_objection_handler=objection
        )

    def process_dataframe(self, df_leads: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates across scored leads and attaches structured prospecting intelligence.
        """
        enriched_rows = []
        total = len(df_leads)
        print(f"[INFO] Synthesizing strategic prospecting dossiers for {total} institutional leads...")

        for idx, (_, row) in enumerate(df_leads.iterrows(), start=1):
            dossier = self.synthesize_lead(row.to_dict())
            
            # Format lists into clean multi-line strings for CSV and Power BI table visuals
            signals_str = " | ".join(dossier.market_signals)
            talking_points_formatted = "\n• " + "\n• ".join(dossier.outreach_talking_points)
            
            lead_dict = row.to_dict()
            lead_dict["prospecting_summary"] = dossier.executive_summary
            lead_dict["market_signals"] = signals_str
            lead_dict["talking_points"] = talking_points_formatted
            lead_dict["pitch_angle"] = dossier.recommended_pitch_angle
            lead_dict["objection_handler"] = dossier.potential_objection_handler
            
            enriched_rows.append(lead_dict)

        return pd.DataFrame(enriched_rows)
