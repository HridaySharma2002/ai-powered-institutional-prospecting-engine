"""
AI-Powered Institutional Prospecting Engine - Master CLI Runner
Executes the end-to-end institutional workflow:
1. Data Ingestion / Generation
2. Feature Engineering (Creditworthiness & Growth Velocity)
3. Predictive Lead Scoring (Scikit-learn Gradient Boosting)
4. LLM Prospecting Agent (LangChain)
5. Power BI Schema Export
"""

import argparse
import sys
from pathlib import Path

from scripts.generate_synthetic_data import generate_datasets
from src.pipeline import run_prospecting_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run AI-Powered Institutional Prospecting Engine"
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate synthetic institutional leads and financial records"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=["gradient_boosting", "random_forest"],
        help="ML algorithm to train for lead scoring"
    )

    args = parser.parse_args()

    # Generate data if flag specified or files missing
    data_file = Path("data/raw_companies.csv")
    if args.regenerate_data or not data_file.exists():
        print("[INIT] Generating realistic institutional dataset...")
        generate_datasets()

    # Run the full pipeline
    run_prospecting_pipeline(model_type=args.model, export_powerbi=True)

if __name__ == "__main__":
    main()
