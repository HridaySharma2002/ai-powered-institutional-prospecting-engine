import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predictive_model import InstitutionalLeadScorer
from src.feature_engineering import MODEL_FEATURE_COLS

class TestPredictiveModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 30
        data = {col: np.random.uniform(1.0, 100.0, size=n) for col in MODEL_FEATURE_COLS}
        data["company_name"] = [f"Company_{i}" for i in range(n)]
        data["company_id"] = [f"CORP_{i:03d}" for i in range(n)]
        data["historical_conversion"] = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
        self.df_mock = pd.DataFrame(data)

    def test_train_and_predict(self):
        scorer = InstitutionalLeadScorer(model_type="gradient_boosting")
        metrics = scorer.train(self.df_mock, target_col="historical_conversion")
        
        self.assertIn("cv_roc_auc", metrics)
        self.assertIn("feature_importance", metrics)

        predictions = scorer.predict_lead_scores(self.df_mock)
        
        self.assertIn("lead_probability_score", predictions.columns)
        self.assertIn("priority_tier", predictions.columns)
        self.assertIn("qualification_status", predictions.columns)

        # Ensure scores are strictly bounded 0.0 to 100.0
        min_score = predictions["lead_probability_score"].min()
        max_score = predictions["lead_probability_score"].max()
        self.assertGreaterEqual(min_score, 0.0)
        self.assertLessEqual(max_score, 100.0)

if __name__ == "__main__":
    unittest.main()
