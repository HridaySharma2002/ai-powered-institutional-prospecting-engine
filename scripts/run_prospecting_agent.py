"""
Independent Execution Script for Module B: LLM Prospecting Agent
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_FEATURES_FILE
from src.predictive_model import InstitutionalLeadScorer
from src.llm_agent import LLMProspectingAgent

def main():
    print("Executing Module B: LLM Prospecting Agent...")
    if not PROCESSED_FEATURES_FILE.exists():
        print("Features not found. Please run scripts/run_scoring_pipeline.py first.")
        return

    df_features = pd.read_csv(PROCESSED_FEATURES_FILE)
    scorer = InstitutionalLeadScorer()
    scorer.load()
    df_scored = scorer.predict_lead_scores(df_features)

    # Filter to top leads for fast strategic briefing
    top_leads = df_scored.sort_values(by="lead_probability_score", ascending=False).head(10)
    agent = LLMProspectingAgent()
    df_enriched = agent.process_dataframe(top_leads)

    output_path = Path("data/llm_prospecting_dossiers_sample.csv")
    df_enriched.to_csv(output_path, index=False)
    print(f"Saved LLM dossiers sample to {output_path}")

if __name__ == "__main__":
    main()
