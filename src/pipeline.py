"""
End-to-End Orchestration Pipeline
Stitches together Feature Engineering -> Predictive ML Scoring -> LLM Prospecting Agent -> Power BI Export.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_COMPANIES_FILE,
    FINANCIAL_METRICS_FILE,
    MARKET_NEWS_FILE,
    PROCESSED_FEATURES_FILE,
    POWERBI_EXPORT_CSV,
    POWERBI_EXPORT_JSON
)
from src.feature_engineering import compute_financial_features
from src.predictive_model import InstitutionalLeadScorer
from src.llm_agent import LLMProspectingAgent

def run_prospecting_pipeline(model_type: str = "gradient_boosting", export_powerbi: bool = True):
    print("=" * 80)
    print("AI-POWERED INSTITUTIONAL PROSPECTING ENGINE - EXECUTION PIPELINE")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[Step 1/5] Ingesting corporate records, financial filings, and market news...")
    if not RAW_COMPANIES_FILE.exists():
        raise FileNotFoundError(f"Missing {RAW_COMPANIES_FILE}. Please run scripts/generate_synthetic_data.py first.")

    df_companies = pd.read_csv(RAW_COMPANIES_FILE)
    df_financials = pd.read_csv(FINANCIAL_METRICS_FILE)
    df_news = pd.read_csv(MARKET_NEWS_FILE)
    print(f" -> Successfully loaded {len(df_companies)} corporate leads.")

    # 2. Feature Engineering
    print("\n[Step 2/5] Computing corporate creditworthiness, solvency, and growth velocity...")
    df_features = compute_financial_features(df_companies, df_financials, df_news)
    df_features.to_csv(PROCESSED_FEATURES_FILE, index=False)
    print(f" -> Generated {len(df_features.columns)} engineered features.")
    print(f" -> Saved processed feature matrix to: {PROCESSED_FEATURES_FILE}")

    # 3. Module A: Machine Learning Predictive Lead Scoring
    print(f"\n[Step 3/5] Training Module A Predictive Lead Scorer ({model_type})...")
    scorer = InstitutionalLeadScorer(model_type=model_type)
    train_metrics = scorer.train(df_features, target_col="historical_conversion")
    
    print(" -> Model Training Completed. Cross-Validation Metrics:")
    print(f"    • 5-Fold Stratified ROC-AUC: {train_metrics['cv_roc_auc']:.4f}")
    print(f"    • Precision:                {train_metrics['cv_precision']:.4f}")
    print(f"    • Recall:                   {train_metrics['cv_recall']:.4f}")
    print(f"    • Brier Calibration Loss:   {train_metrics['brier_score']:.4f}")
    print("\n -> Top 5 Feature Importance Drivers:")
    for feat, imp in list(train_metrics["feature_importance"].items())[:5]:
        print(f"    - {feat:25s}: {imp * 100:.2f}%")

    # Predict lead probability scores
    df_scored = scorer.predict_lead_scores(df_features)
    tier_counts = df_scored["priority_tier"].value_counts().to_dict()
    print("\n -> Institutional Priority Tier Distribution:")
    for tier, count in tier_counts.items():
        print(f"    • {tier:25s}: {count} leads ({count/len(df_scored)*100:.1f}%)")

    # 4. Module B: LangChain LLM Prospecting Agent
    print("\n[Step 4/5] Deploying Module B LangChain Prospecting Agent...")
    agent = LLMProspectingAgent()
    df_enriched = agent.process_dataframe(df_scored)

    # 5. Module C: Power BI Data Formatting and Export
    if export_powerbi:
        print("\n[Step 5/5] Structuring schema and exporting for Power BI dashboard ingestion...")
        
        # Columns designated for Power BI modeling
        powerbi_cols = [
            "company_id",
            "company_name",
            "ticker",
            "sector",
            "sub_sector",
            "headquarters",
            "employee_count",
            "annual_revenue_m",
            "revenue_velocity_yoy",
            "revenue_cagr_3yr",
            "ebitda_m",
            "net_margin",
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "altman_z_score",
            "credit_risk_tier",
            "market_cap_m",
            "market_cap_tier",
            "event_type",
            "headline",
            "signal_sentiment_score",
            "lead_probability_score",
            "priority_tier",
            "qualification_status",
            "prospecting_summary",
            "market_signals",
            "talking_points",
            "pitch_angle",
            "objection_handler"
        ]

        df_export = df_enriched[[c for c in powerbi_cols if c in df_enriched.columns]].copy()
        
        # Power BI specific friendly rounding
        df_export["revenue_velocity_yoy"] = (df_export["revenue_velocity_yoy"] * 100).round(2)
        df_export["revenue_cagr_3yr"] = (df_export["revenue_cagr_3yr"] * 100).round(2)
        df_export["net_margin"] = (df_export["net_margin"] * 100).round(2)

        # Export CSV and JSON
        df_export.to_csv(POWERBI_EXPORT_CSV, index=False)
        df_export.to_json(POWERBI_EXPORT_JSON, orient="records", indent=2)

        print(f" -> Exported Power BI CSV dataset:  {POWERBI_EXPORT_CSV} ({len(df_export)} records)")
        print(f" -> Exported Power BI JSON dataset: {POWERBI_EXPORT_JSON}")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE: TOP 5 HIGH-CONVICTION INSTITUTIONAL LEADS")
    print("=" * 80)
    top_leads = df_enriched.sort_values(by="lead_probability_score", ascending=False).head(5)
    for _, l in top_leads.iterrows():
        print(f"\n[{l['priority_tier']}] {l['company_name']} ({l['ticker']}) - Score: {l['lead_probability_score']}/100")
        print(f"  Sector: {l['sector']} | Revenue: ${l['annual_revenue_m']:.1f}M | YoY Growth: {l['revenue_velocity_yoy']*100:.1f}%")
        print(f"  Summary: {l['prospecting_summary']}")
        print(f"  Pitch Angle: {l['pitch_angle']}")

    return df_enriched

if __name__ == "__main__":
    run_prospecting_pipeline()
