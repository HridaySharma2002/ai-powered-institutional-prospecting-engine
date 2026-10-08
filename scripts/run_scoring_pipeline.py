"""
Independent Execution Script for Module A: Predictive Lead Scoring
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RAW_COMPANIES_FILE, FINANCIAL_METRICS_FILE, MARKET_NEWS_FILE, PROCESSED_FEATURES_FILE
from src.feature_engineering import compute_financial_features
from src.predictive_model import InstitutionalLeadScorer

def main():
    print("Executing Module A: Predictive Lead Scoring Pipeline...")
    df_companies = pd.read_csv(RAW_COMPANIES_FILE)
    df_financials = pd.read_csv(FINANCIAL_METRICS_FILE)
    df_news = pd.read_csv(MARKET_NEWS_FILE)

    df_features = compute_financial_features(df_companies, df_financials, df_news)
    df_features.to_csv(PROCESSED_FEATURES_FILE, index=False)

    scorer = InstitutionalLeadScorer(model_type="gradient_boosting")
    metrics = scorer.train(df_features)
    print(f"Model Training Results: ROC-AUC={metrics['cv_roc_auc']:.4f}, Brier Loss={metrics['brier_score']:.4f}")

    df_scored = scorer.predict_lead_scores(df_features)
    output_path = Path("data/scored_leads_module_a.csv")
    df_scored.to_csv(output_path, index=False)
    print(f"Saved scored leads to {output_path}")

if __name__ == "__main__":
    main()
