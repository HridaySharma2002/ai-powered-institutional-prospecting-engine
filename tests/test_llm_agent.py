import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.llm_agent import LLMProspectingAgent, StrategicProspectingDossier

class TestLLMAgent(unittest.TestCase):

    def setUp(self):
        self.agent = LLMProspectingAgent()
        self.sample_lead = {
            "company_name": "Apex Cloud Dynamics",
            "ticker": "APXC",
            "sector": "Technology",
            "sub_sector": "Hybrid Infrastructure",
            "headquarters": "Austin, TX",
            "employee_count": 1200,
            "lead_probability_score": 88.5,
            "priority_tier": "Tier 1: Strategic Alpha",
            "annual_revenue_m": 450.0,
            "revenue_velocity_yoy": 0.42,
            "revenue_cagr_3yr": 0.38,
            "ebitda_m": 95.0,
            "net_margin": 0.15,
            "free_cash_flow_m": 60.0,
            "debt_to_equity": 0.35,
            "current_ratio": 2.5,
            "quick_ratio": 1.9,
            "altman_z_score": 3.8,
            "credit_risk_tier": "Prime / Safe Zone",
            "event_type": "Executive Leadership Hire",
            "headline": "Apex Cloud Dynamics Appoints CRO to Expand Enterprise Accounts",
            "narrative_summary": "Mandate to scale Fortune 500 sales pipeline.",
            "signal_sentiment_score": 0.85
        }

    def test_synthesis(self):
        dossier = self.agent.synthesize_lead(self.sample_lead)
        self.assertIsInstance(dossier, StrategicProspectingDossier)
        self.assertTrue(len(dossier.executive_summary) > 20)
        self.assertTrue(len(dossier.outreach_talking_points) >= 2)
        self.assertTrue(len(dossier.recommended_pitch_angle) > 5)

if __name__ == "__main__":
    unittest.main()
