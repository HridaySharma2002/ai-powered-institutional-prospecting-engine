"""
Synthetic Institutional Lead Data Generator
Generates realistic financial metrics, market signals, and unstructured news
for corporate B2B sales prospecting.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_DIR,
    RAW_COMPANIES_FILE,
    FINANCIAL_METRICS_FILE,
    MARKET_NEWS_FILE
)

np.random.seed(42)

CORPORATE_PROFILES = [
    {"name": "NovaScale AI Systems", "ticker": "NVSC", "sector": "Technology", "sub_sector": "Enterprise AI & Cloud", "hq": "San Francisco, CA"},
    {"name": "Apex Cloud Dynamics", "ticker": "APXC", "sector": "Technology", "sub_sector": "Hybrid Infrastructure", "hq": "Austin, TX"},
    {"name": "OmniLogix Freight Tech", "ticker": "OLFT", "sector": "Logistics", "sub_sector": "Autonomous Supply Chain", "hq": "Chicago, IL"},
    {"name": "Vanguard BioPharma", "ticker": "VGBP", "sector": "Healthcare", "sub_sector": "Precision Therapeutics", "hq": "Boston, MA"},
    {"name": "Titanium Heavy Robotics", "ticker": "THRB", "sector": "Industrial", "sub_sector": "Smart Manufacturing", "hq": "Detroit, MI"},
    {"name": "Solaris CleanGrid", "ticker": "SLCL", "sector": "Clean Energy", "sub_sector": "Grid Scale Storage", "hq": "Denver, CO"},
    {"name": "FinPulse Global Payments", "ticker": "FPGL", "sector": "Financial Services", "sub_sector": "Cross-Border Settlement", "hq": "New York, NY"},
    {"name": "QuantumCore Security", "ticker": "QCSE", "sector": "Technology", "sub_sector": "Zero-Trust Cybersecurity", "hq": "Seattle, WA"},
    {"name": "AeroDynamics Avionics", "ticker": "ADAV", "sector": "Industrial", "sub_sector": "Commercial Aerospace Systems", "hq": "Dallas, TX"},
    {"name": "Veritas Health Informatics", "ticker": "VRHI", "sector": "Healthcare", "sub_sector": "Clinical AI Diagnostics", "hq": "Philadelphia, PA"},
    {"name": "Nexus Retail Solutions", "ticker": "NXRS", "sector": "Retail & E-Commerce", "sub_sector": "Omnichannel Point-of-Sale", "hq": "Atlanta, GA"},
    {"name": "Hyperion Datacenters", "ticker": "HYDC", "sector": "Technology", "sub_sector": "Edge Computing Facilities", "hq": "Phoenix, AZ"},
    {"name": "IronClad Supply Logistics", "ticker": "ICSL", "sector": "Logistics", "sub_sector": "Cold-Chain Fleet Automation", "hq": "Minneapolis, MN"},
    {"name": "Synapse BioEngineering", "ticker": "SNPB", "sector": "Healthcare", "sub_sector": "Synthetic Biology Platforms", "hq": "San Diego, CA"},
    {"name": "BlueRidge Commercial Banking", "ticker": "BRCB", "sector": "Financial Services", "sub_sector": "Commercial Lending", "hq": "Charlotte, NC"},
    {"name": "Zenith Agritech Systems", "ticker": "ZNAG", "sector": "Clean Energy", "sub_sector": "Precision Irrigation & Soil AI", "hq": "Des Moines, IA"},
    {"name": "Cobalt Industrial Tools", "ticker": "CBIT", "sector": "Industrial", "sub_sector": "Predictive Tooling Sensors", "hq": "Cleveland, OH"},
    {"name": "CipherGuard Identity", "ticker": "CPGD", "sector": "Technology", "sub_sector": "Decentralized IAM", "hq": "San Jose, CA"},
    {"name": "Pacifica Marine Freight", "ticker": "PCMF", "sector": "Logistics", "sub_sector": "Maritime Routing Systems", "hq": "Long Beach, CA"},
    {"name": "AuraCare Telehealth", "ticker": "ARCR", "sector": "Healthcare", "sub_sector": "Virtual Care Infrastructure", "hq": "Nashville, TN"},
    {"name": "Strata Wealth Tech", "ticker": "STWT", "sector": "Financial Services", "sub_sector": "Automated Asset Allocation", "hq": "Miami, FL"},
    {"name": "PulseWave Telecommunications", "ticker": "PWTEL", "sector": "Technology", "sub_sector": "5G Private Campus Networks", "hq": "Reston, VA"},
    {"name": "GeoThermal Frontier", "ticker": "GTFR", "sector": "Clean Energy", "sub_sector": "Deep Well Baseload Power", "hq": "Reno, NV"},
    {"name": "Kinetics Heavy Machining", "ticker": "KHM", "sector": "Industrial", "sub_sector": "High-Tolerance Turbine Parts", "hq": "Pittsburgh, PA"},
    {"name": "Sovereign Trade Finance", "ticker": "STFN", "sector": "Financial Services", "sub_sector": "Letter of Credit Automation", "hq": "New York, NY"},
    {"name": "MedMatrix Surgical", "ticker": "MMXS", "sector": "Healthcare", "sub_sector": "Minimally Invasive Robotics", "hq": "San Jose, CA"},
    {"name": "FreightForge Warehousing", "ticker": "FFWH", "sector": "Logistics", "sub_sector": "Automated Micro-Fulfillment", "hq": "Indianapolis, IN"},
    {"name": "EcoPack Materials Group", "ticker": "EPMG", "sector": "Clean Energy", "sub_sector": "Biodegradable Packaging", "hq": "Portland, OR"},
    {"name": "OmniChannel Direct", "ticker": "OCDR", "sector": "Retail & E-Commerce", "sub_sector": "AI Demand Forecasting", "hq": "Los Angeles, CA"},
    {"name": "Vector Aerospace Avionics", "ticker": "VRAV", "sector": "Industrial", "sub_sector": "Defense Navigation Sensors", "hq": "Huntsville, AL"},
    {"name": "Starlight Satellite Networks", "ticker": "SLSN", "sector": "Technology", "sub_sector": "LEO Satellite Broadband", "hq": "Boulder, CO"},
    {"name": "Centurion Underwriting", "ticker": "CNUN", "sector": "Financial Services", "sub_sector": "Parametric Risk Insurance", "hq": "Hartford, CT"},
    {"name": "Integra Diagnostic Labs", "ticker": "INDL", "sector": "Healthcare", "sub_sector": "High-Throughput Sequencing", "hq": "Raleigh, NC"},
    {"name": "Velocity Parcel Logistics", "ticker": "VPLG", "sector": "Logistics", "sub_sector": "Last-Mile Drone Delivery", "hq": "Memphis, TN"},
    {"name": "TerraFirma Solid Battery", "ticker": "TFSB", "sector": "Clean Energy", "sub_sector": "Lithium-Sulfur Cells", "hq": "Salt Lake City, UT"},
    {"name": "Foundry Precision Castings", "ticker": "FDRP", "sector": "Industrial", "sub_sector": "Additive Metallurgy", "hq": "Milwaukee, WI"},
    {"name": "CloudNine SaaS ERP", "ticker": "CNERP", "sector": "Technology", "sub_sector": "Mid-Market Financial Cloud", "hq": "San Mateo, CA"},
    {"name": "Beacon Credit Union Tech", "ticker": "BCUT", "sector": "Financial Services", "sub_sector": "Open Banking Core APIs", "hq": "Columbus, OH"},
    {"name": "CarePoint Specialty Clinics", "ticker": "CPSC", "sector": "Healthcare", "sub_sector": "Outpatient Network Operations", "hq": "Houston, TX"},
    {"name": "Summit Cold Storage", "ticker": "SMCS", "sector": "Logistics", "sub_sector": "Temperature-Controlled Hubs", "hq": "Kansas City, MO"},
    {"name": "PureWater Desalination", "ticker": "PWDS", "sector": "Clean Energy", "sub_sector": "Reverse Osmosis Systems", "hq": "San Antonio, TX"},
    {"name": "ProCraft Industrial Coatings", "ticker": "PCIC", "sector": "Industrial", "sub_sector": "Corrosion-Resistant Polymers", "hq": "Cincinnati, OH"},
    {"name": "Synthetix Machine Vision", "ticker": "SYMV", "sector": "Technology", "sub_sector": "Autonomous Quality Inspection", "hq": "Santa Clara, CA"},
    {"name": "Meridian Merchant Services", "ticker": "MDMS", "sector": "Financial Services", "sub_sector": "POS Hardware & Merchant Rails", "hq": "Scottsdale, AZ"},
    {"name": "GeneThera Biologics", "ticker": "GTBL", "sector": "Healthcare", "sub_sector": "Adeno-Associated Virus Vectors", "hq": "Cambridge, MA"},
    {"name": "SwiftMile Intermodal", "ticker": "SMIM", "sector": "Logistics", "sub_sector": "Rail-to-Truck Transloading", "hq": "Omaha, NE"},
    {"name": "GreenVolt Hydrogen Systems", "ticker": "GVHS", "sector": "Clean Energy", "sub_sector": "PEM Electrolyzer Stacks", "hq": "Houston, TX"},
    {"name": "Borealis Specialty Alloys", "ticker": "BSAL", "sector": "Industrial", "sub_sector": "Nickel-Titanium Shape Memory", "hq": "Allentown, PA"},
    {"name": "DataVault Enterprise Cloud", "ticker": "DVEC", "sector": "Technology", "sub_sector": "Sovereign Cloud Data Warehousing", "hq": "Arlington, VA"},
    {"name": "Apex Capital Lending Group", "ticker": "ACLG", "sector": "Financial Services", "sub_sector": "Asset-Backed Equipment Finance", "hq": "Chicago, IL"}
]

STRATEGIC_EVENTS = [
    {
        "event_type": "Executive Leadership Hire",
        "headline_template": "{name} Appoints Former Enterprise Sales VP as Chief Revenue Officer to Accelerate Global Expansion",
        "details_template": "Newly appointed CRO brings a mandate to double enterprise institutional contracts and overhaul digital prospecting pipelines across EMEA and North America.",
        "sentiment": 0.85
    },
    {
        "event_type": "Capital Raise & Growth Equity",
        "headline_template": "{name} Secures $140M in Growth Financing Led by Institutional Tech Investors",
        "details_template": "Proceeds will be directly earmarked for scaling sales ops, modernizing data infrastructure, and integrating next-generation AI workflows across enterprise accounts.",
        "sentiment": 0.90
    },
    {
        "event_type": "Strategic M&A Acquisition",
        "headline_template": "{name} Completes Strategic Acquisition of Automated Analytics Platform to Broaden B2B Footprint",
        "details_template": "Transaction integrates complementary B2B analytics engines, creating urgent demand for vendor consolidation, cross-sell enablement, and institutional tooling.",
        "sentiment": 0.78
    },
    {
        "event_type": "Enterprise Cloud Transformation",
        "headline_template": "{name} Unveils Multi-Year $80M Digital Modernization Initiative Across Core Operations",
        "details_template": "Executive board approves major capital spend focused on enterprise software, automated risk evaluation, and modern customer intelligence platforms.",
        "sentiment": 0.82
    },
    {
        "event_type": "Cost Restructuring & Margin Squeeze",
        "headline_template": "{name} Initiates Strategic Cost Review Amid Rising Debt Service and Supply Chain Bottlenecks",
        "details_template": "Company reports shrinking operating margins and elevated leverage, prioritizing efficiency tooling, unit-economic optimization, and vendor rationalization.",
        "sentiment": -0.35
    },
    {
        "event_type": "Regulatory Compliance Mandate",
        "headline_template": "{name} Faces Heightened Scrutiny Under New Federal Data Transparency and Reporting Standards",
        "details_template": "Compliance deadlines are accelerating urgency to adopt institutional audit-grade software solutions with verifiable data governance and risk tracking.",
        "sentiment": 0.40
    }
]

def generate_datasets():
    companies_data = []
    financial_data = []
    news_data = []

    for i, profile in enumerate(CORPORATE_PROFILES):
        comp_id = f"CORP_{i+1:03d}"
        
        # Determine archetype:
        # High-growth solvent (45%), Stable mid-market (35%), Stressed/leveraged (20%)
        p_type = np.random.choice(["high_growth", "stable_mature", "stressed"], p=[0.45, 0.35, 0.20])
        
        founding_year = int(np.random.randint(1998, 2021))
        
        if p_type == "high_growth":
            revenue = round(np.random.uniform(120.0, 950.0), 2)
            rev_growth_yoy = round(np.random.uniform(0.25, 0.75), 4) # 25% - 75% growth
            rev_cagr_3yr = round(np.random.uniform(0.28, 0.65), 4)
            prev_rev = round(revenue / (1.0 + rev_growth_yoy), 2)
            rev_3yr_ago = round(revenue / ((1.0 + rev_cagr_3yr) ** 3), 2)
            
            ebitda_margin = np.random.uniform(0.18, 0.38)
            net_margin = np.random.uniform(0.10, 0.24)
            
            debt_to_equity = round(np.random.uniform(0.15, 0.85), 3) # Low to moderate debt
            quick_ratio = round(np.random.uniform(1.8, 3.8), 2)
            current_ratio = round(quick_ratio + np.random.uniform(0.4, 0.9), 2)
            
            market_cap = round(revenue * np.random.uniform(4.5, 12.0), 2)
            total_assets = round(revenue * np.random.uniform(1.5, 3.0), 2)
            total_equity = round(total_assets / (1.0 + debt_to_equity), 2)
            total_debt = round(total_equity * debt_to_equity, 2)
            total_liabilities = round(total_assets - total_equity, 2)
            
            current_liabilities = round(total_liabilities * 0.45, 2)
            current_assets = round(current_liabilities * current_ratio, 2)
            cash = round(current_assets * np.random.uniform(0.40, 0.70), 2)
            
            ebitda = round(revenue * ebitda_margin, 2)
            net_income = round(revenue * net_margin, 2)
            operating_cf = round(ebitda * np.random.uniform(0.75, 1.05), 2)
            capex = round(revenue * np.random.uniform(0.04, 0.09), 2)
            rd_expense = round(revenue * np.random.uniform(0.12, 0.22), 2)
            employees = int(revenue * np.random.uniform(3.5, 6.0))
            
            # High probability conversion ground truth
            converted = 1 if np.random.rand() < 0.88 else 0
            event_choice = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
            
        elif p_type == "stable_mature":
            revenue = round(np.random.uniform(300.0, 2500.0), 2)
            rev_growth_yoy = round(np.random.uniform(0.05, 0.20), 4) # 5% - 20% growth
            rev_cagr_3yr = round(np.random.uniform(0.06, 0.18), 4)
            prev_rev = round(revenue / (1.0 + rev_growth_yoy), 2)
            rev_3yr_ago = round(revenue / ((1.0 + rev_cagr_3yr) ** 3), 2)
            
            ebitda_margin = np.random.uniform(0.14, 0.25)
            net_margin = np.random.uniform(0.06, 0.14)
            
            debt_to_equity = round(np.random.uniform(0.70, 1.60), 3) # Moderate debt
            quick_ratio = round(np.random.uniform(1.1, 1.8), 2)
            current_ratio = round(quick_ratio + np.random.uniform(0.3, 0.6), 2)
            
            market_cap = round(revenue * np.random.uniform(1.8, 4.0), 2)
            total_assets = round(revenue * np.random.uniform(1.8, 3.5), 2)
            total_equity = round(total_assets / (1.0 + debt_to_equity), 2)
            total_debt = round(total_equity * debt_to_equity, 2)
            total_liabilities = round(total_assets - total_equity, 2)
            
            current_liabilities = round(total_liabilities * 0.50, 2)
            current_assets = round(current_liabilities * current_ratio, 2)
            cash = round(current_assets * np.random.uniform(0.25, 0.45), 2)
            
            ebitda = round(revenue * ebitda_margin, 2)
            net_income = round(revenue * net_margin, 2)
            operating_cf = round(ebitda * np.random.uniform(0.70, 0.95), 2)
            capex = round(revenue * np.random.uniform(0.05, 0.12), 2)
            rd_expense = round(revenue * np.random.uniform(0.05, 0.11), 2)
            employees = int(revenue * np.random.uniform(4.0, 7.5))
            
            converted = 1 if np.random.rand() < 0.50 else 0
            event_choice = np.random.choice([0, 2, 3, 5], p=[0.25, 0.25, 0.30, 0.20])
            
        else: # Stressed / highly leveraged
            revenue = round(np.random.uniform(80.0, 600.0), 2)
            rev_growth_yoy = round(np.random.uniform(-0.15, 0.04), 4) # Flat to declining
            rev_cagr_3yr = round(np.random.uniform(-0.08, 0.03), 4)
            prev_rev = round(revenue / (1.0 + rev_growth_yoy), 2)
            rev_3yr_ago = round(revenue / ((1.0 + rev_cagr_3yr) ** 3), 2)
            
            ebitda_margin = np.random.uniform(-0.05, 0.08)
            net_margin = np.random.uniform(-0.12, 0.02)
            
            debt_to_equity = round(np.random.uniform(2.4, 6.5), 3) # Highly levered
            quick_ratio = round(np.random.uniform(0.45, 0.95), 2)
            current_ratio = round(quick_ratio + np.random.uniform(0.1, 0.3), 2)
            
            market_cap = round(revenue * np.random.uniform(0.5, 1.4), 2)
            total_assets = round(revenue * np.random.uniform(1.2, 2.2), 2)
            total_equity = round(max(5.0, total_assets / (1.0 + debt_to_equity)), 2)
            total_debt = round(total_equity * debt_to_equity, 2)
            total_liabilities = round(total_assets - total_equity, 2)
            
            current_liabilities = round(total_liabilities * 0.65, 2)
            current_assets = round(current_liabilities * current_ratio, 2)
            cash = round(current_assets * np.random.uniform(0.10, 0.25), 2)
            
            ebitda = round(revenue * ebitda_margin, 2)
            net_income = round(revenue * net_margin, 2)
            operating_cf = round(ebitda * 0.5, 2)
            capex = round(revenue * np.random.uniform(0.02, 0.05), 2)
            rd_expense = round(revenue * np.random.uniform(0.02, 0.06), 2)
            employees = int(revenue * np.random.uniform(3.0, 5.0))
            
            converted = 1 if np.random.rand() < 0.10 else 0
            event_choice = np.random.choice([4, 5], p=[0.70, 0.30])

        event = STRATEGIC_EVENTS[event_choice]

        # Company profile row
        companies_data.append({
            "company_id": comp_id,
            "company_name": profile["name"],
            "ticker": profile["ticker"],
            "sector": profile["sector"],
            "sub_sector": profile["sub_sector"],
            "headquarters": profile["hq"],
            "founding_year": founding_year,
            "employee_count": employees,
            "target_lead_archetype": p_type,
            "historical_conversion": converted
        })

        # Financial metrics row
        financial_data.append({
            "company_id": comp_id,
            "market_cap_m": market_cap,
            "annual_revenue_m": revenue,
            "prev_year_revenue_m": prev_rev,
            "revenue_3yr_ago_m": rev_3yr_ago,
            "ebitda_m": ebitda,
            "net_income_m": net_income,
            "total_assets_m": total_assets,
            "total_liabilities_m": total_liabilities,
            "total_debt_m": total_debt,
            "total_equity_m": total_equity,
            "current_assets_m": current_assets,
            "current_liabilities_m": current_liabilities,
            "cash_and_equivalents_m": cash,
            "operating_cash_flow_m": operating_cf,
            "capital_expenditures_m": capex,
            "rd_expense_m": rd_expense
        })

        # News & Market signals row
        news_data.append({
            "company_id": comp_id,
            "news_date": "2026-09-15",
            "news_source": np.random.choice(["Bloomberg Institutional", "Reuters Tech", "Wall Street Journal", "SEC 8-K Filing"]),
            "event_type": event["event_type"],
            "headline": event["headline_template"].format(name=profile["name"]),
            "narrative_summary": event["details_template"],
            "signal_sentiment_score": event["sentiment"]
        })

    # Save to CSV
    df_companies = pd.DataFrame(companies_data)
    df_financials = pd.DataFrame(financial_data)
    df_news = pd.DataFrame(news_data)

    df_companies.to_csv(RAW_COMPANIES_FILE, index=False)
    df_financials.to_csv(FINANCIAL_METRICS_FILE, index=False)
    df_news.to_csv(MARKET_NEWS_FILE, index=False)

    print(f"[OK] Generated {len(df_companies)} corporate profiles -> {RAW_COMPANIES_FILE}")
    print(f"[OK] Generated financial metrics -> {FINANCIAL_METRICS_FILE}")
    print(f"[OK] Generated market news signals -> {MARKET_NEWS_FILE}")

if __name__ == "__main__":
    generate_datasets()
