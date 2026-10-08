"""
Financial Feature Engineering Pipeline
Computes corporate creditworthiness, growth velocity, liquidity, and solvency metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple

def compute_financial_features(
    df_companies: pd.DataFrame,
    df_financials: pd.DataFrame,
    df_news: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges corporate profile, raw financial statements, and news signals,
    then executes domain-specific feature engineering.
    
    Returns:
        pd.DataFrame: Enriched feature matrix ready for ML scoring and LLM synthesis.
    """
    # 1. Merge datasets on company_id
    df = df_companies.merge(df_financials, on="company_id", how="inner")
    df = df.merge(df_news, on="company_id", how="left")
    
    # 2. Growth Velocity Features
    # Revenue Velocity (YoY Growth Rate)
    df["revenue_velocity_yoy"] = (
        (df["annual_revenue_m"] - df["prev_year_revenue_m"]) /
        df["prev_year_revenue_m"].replace(0, np.nan)
    ).fillna(0.0)

    # 3-Year Revenue CAGR
    df["revenue_cagr_3yr"] = np.where(
        df["revenue_3yr_ago_m"] > 0,
        (df["annual_revenue_m"] / df["revenue_3yr_ago_m"]) ** (1.0 / 3.0) - 1.0,
        df["revenue_velocity_yoy"]
    )

    # 3. Solvency & Creditworthiness Metrics
    # Debt-to-Equity Ratio (clipped for stability)
    safe_equity = df["total_equity_m"].apply(lambda x: max(x, 1.0))
    df["debt_to_equity"] = (df["total_debt_m"] / safe_equity).round(4)
    
    # Debt-to-Assets Ratio
    df["debt_to_assets"] = (
        df["total_debt_m"] / df["total_assets_m"].replace(0, np.nan)
    ).fillna(0.0).round(4)

    # 4. Liquidity & Operational Efficiency
    # Current Ratio (Working Capital Ratio)
    safe_current_liab = df["current_liabilities_m"].apply(lambda x: max(x, 0.5))
    df["current_ratio"] = (df["current_assets_m"] / safe_current_liab).round(3)

    # Quick Ratio (Cash & Cash Equivalents over Current Liabilities)
    df["quick_ratio"] = (df["cash_and_equivalents_m"] / safe_current_liab).round(3)

    # Net Margin & EBITDA Margin
    safe_revenue = df["annual_revenue_m"].apply(lambda x: max(x, 1.0))
    df["net_margin"] = (df["net_income_m"] / safe_revenue).round(4)
    df["ebitda_margin"] = (df["ebitda_m"] / safe_revenue).round(4)

    # Free Cash Flow ($M)
    df["free_cash_flow_m"] = (
        df["operating_cash_flow_m"] - df["capital_expenditures_m"]
    ).round(2)
    
    # FCF Conversion Yield (FCF / Revenue)
    df["fcf_margin"] = (df["free_cash_flow_m"] / safe_revenue).round(4)

    # 5. Altman Z-Score Proxy (EM-Score for Institutional Credit Analysis)
    # Z = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings Proxy (Net Income) / Total Assets
    # X3 = EBIT (EBITDA Proxy) / Total Assets
    # X4 = Book Value of Equity / Total Liabilities
    working_capital = df["current_assets_m"] - df["current_liabilities_m"]
    safe_assets = df["total_assets_m"].apply(lambda x: max(x, 1.0))
    safe_liabilities = df["total_liabilities_m"].apply(lambda x: max(x, 1.0))
    
    x1 = (working_capital / safe_assets).clip(-1.0, 1.0)
    x2 = (df["net_income_m"] / safe_assets).clip(-1.0, 1.0)
    x3 = (df["ebitda_m"] / safe_assets).clip(-1.0, 1.0)
    x4 = (df["total_equity_m"] / safe_liabilities).clip(0.0, 10.0)

    df["altman_z_score"] = (6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4).round(2)

    # Credit Health Category based on Z-Score
    # Safe Zone > 2.6, Grey Zone 1.1 - 2.6, Distress Zone < 1.1
    df["credit_risk_tier"] = pd.cut(
        df["altman_z_score"],
        bins=[-np.inf, 1.1, 2.6, np.inf],
        labels=["High Risk / Distress", "Moderate Risk", "Prime / Safe Zone"]
    )

    # 6. Strategic Growth & Innovation Metrics
    # R&D Intensity
    df["rd_intensity"] = (df["rd_expense_m"] / safe_revenue).round(4)
    
    # Market Cap Scale (Log Transformed)
    df["log_market_cap"] = np.log1p(df["market_cap_m"]).round(3)
    
    # Market Cap Tier classification
    df["market_cap_tier"] = pd.cut(
        df["market_cap_m"],
        bins=[-np.inf, 500.0, 2000.0, np.inf],
        labels=["Small-Cap / Growth", "Mid-Market", "Large Enterprise"]
    )

    # 7. Unstructured Sentiment Integration
    df["signal_sentiment_score"] = df["signal_sentiment_score"].fillna(0.0)

    # Composite Growth-Health Multiplier
    # Rewards accelerating revenue with low leverage and positive operating cash flow
    growth_multiplier = np.clip(1.0 + df["revenue_velocity_yoy"], 0.5, 2.5)
    leverage_penalty = np.where(df["debt_to_equity"] > 2.5, 0.70, 1.0)
    df["composite_growth_index"] = (
        (df["altman_z_score"].clip(0.1, 10.0) * growth_multiplier * leverage_penalty)
    ).round(2)

    return df

# Explicit feature column list utilized by Scikit-learn
MODEL_FEATURE_COLS = [
    "annual_revenue_m",
    "revenue_velocity_yoy",
    "revenue_cagr_3yr",
    "debt_to_equity",
    "debt_to_assets",
    "current_ratio",
    "quick_ratio",
    "net_margin",
    "ebitda_margin",
    "free_cash_flow_m",
    "altman_z_score",
    "rd_intensity",
    "log_market_cap",
    "signal_sentiment_score"
]
