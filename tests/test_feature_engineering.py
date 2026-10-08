import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import compute_financial_features, MODEL_FEATURE_COLS

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df_companies = pd.DataFrame([{
            "company_id": "CORP_TEST_1",
            "company_name": "Test Alpha Corp",
            "ticker": "TAC",
            "sector": "Technology",
            "sub_sector": "Cloud",
            "headquarters": "New York, NY",
            "founding_year": 2015,
            "employee_count": 500,
            "target_lead_archetype": "high_growth",
            "historical_conversion": 1
        }])

        self.df_financials = pd.DataFrame([{
            "company_id": "CORP_TEST_1",
            "market_cap_m": 1000.0,
            "annual_revenue_m": 200.0,
            "prev_year_revenue_m": 150.0,
            "revenue_3yr_ago_m": 100.0,
            "ebitda_m": 50.0,
            "net_income_m": 30.0,
            "total_assets_m": 400.0,
            "total_liabilities_m": 150.0,
            "total_debt_m": 80.0,
            "total_equity_m": 250.0,
            "current_assets_m": 120.0,
            "current_liabilities_m": 60.0,
            "cash_and_equivalents_m": 45.0,
            "operating_cash_flow_m": 40.0,
            "capital_expenditures_m": 10.0,
            "rd_expense_m": 25.0
        }])

        self.df_news = pd.DataFrame([{
            "company_id": "CORP_TEST_1",
            "news_date": "2026-09-01",
            "news_source": "Bloomberg",
            "event_type": "Capital Raise",
            "headline": "Test Alpha Corp Raises $50M",
            "narrative_summary": "Earmarked for cloud expansion.",
            "signal_sentiment_score": 0.85
        }])

    def test_feature_computation(self):
        df_feat = compute_financial_features(self.df_companies, self.df_financials, self.df_news)
        
        # Verify all model feature columns exist
        for col in MODEL_FEATURE_COLS:
            self.assertIn(col, df_feat.columns, f"Missing feature column: {col}")

        # Check growth velocity
        # YoY: (200 - 150) / 150 = 0.3333
        self.assertAlmostEqual(df_feat["revenue_velocity_yoy"].iloc[0], 0.3333, places=2)

        # Check debt to equity
        # 80 / 250 = 0.32
        self.assertAlmostEqual(df_feat["debt_to_equity"].iloc[0], 0.32, places=2)

        # Check Altman Z-Score calculation is finite and positive
        self.assertGreater(df_feat["altman_z_score"].iloc[0], 0.0)

        # Free cash flow = 40 - 10 = 30
        self.assertEqual(df_feat["free_cash_flow_m"].iloc[0], 30.0)

if __name__ == "__main__":
    unittest.main()
