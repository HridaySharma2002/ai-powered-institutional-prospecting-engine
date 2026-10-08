import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
POWERBI_DIR = PROJECT_ROOT / "powerbi"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
POWERBI_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ML Pipeline Configuration
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
LEAD_SCORE_THRESHOLD = float(os.getenv("LEAD_SCORE_THRESHOLD", "75.0"))

# Lead Score Tiers
TIER_1_CUTOFF = 80.0
TIER_2_CUTOFF = 65.0
TIER_3_CUTOFF = 45.0

# Export Filenames
RAW_COMPANIES_FILE = DATA_DIR / "raw_companies.csv"
FINANCIAL_METRICS_FILE = DATA_DIR / "financial_metrics.csv"
MARKET_NEWS_FILE = DATA_DIR / "market_news.csv"
PROCESSED_FEATURES_FILE = DATA_DIR / "lead_features.csv"
POWERBI_EXPORT_CSV = DATA_DIR / "institutional_prospects_powerbi.csv"
POWERBI_EXPORT_JSON = DATA_DIR / "institutional_prospects_powerbi.json"
MODEL_CHECKPOINT_FILE = MODELS_DIR / "lead_scoring_model.joblib"
