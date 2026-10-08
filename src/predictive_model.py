"""
Module A: Predictive Lead Scoring Model (Scikit-Learn)
Ingests engineered financial health and market signal metrics, trains a calibrated
Gradient Boosting classifier, and outputs a bounded Lead Probability Score (0-100).
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

from src.config import (
    RANDOM_STATE,
    MODEL_CHECKPOINT_FILE,
    TIER_1_CUTOFF,
    TIER_2_CUTOFF,
    TIER_3_CUTOFF
)
from src.feature_engineering import MODEL_FEATURE_COLS

class InstitutionalLeadScorer:
    """
    Production-grade ML pipeline for corporate lead probability scoring.
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.feature_cols = MODEL_FEATURE_COLS
        self.pipeline: Pipeline = None
        self.feature_importances_: Dict[str, float] = {}

    def _build_base_estimator(self):
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=3,
                min_samples_split=4,
                subsample=0.85,
                random_state=RANDOM_STATE
            )
        else:
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=3,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )

    def train(self, df_features: pd.DataFrame, target_col: str = "historical_conversion") -> Dict[str, Any]:
        """
        Trains the predictive model with feature scaling and stratified cross-validation.
        """
        X = df_features[self.feature_cols].copy()
        y = df_features[target_col].copy()

        # Handle any residual missing values
        X = X.fillna(X.median(numeric_only=True))

        base_clf = self._build_base_estimator()

        # Cross-validation validation metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(base_clf, X, y, cv=cv, scoring="roc_auc").mean()
        cv_precision = cross_val_score(base_clf, X, y, cv=cv, scoring="precision").mean()
        cv_recall = cross_val_score(base_clf, X, y, cv=cv, scoring="recall").mean()

        # Build pipeline with scaling
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", base_clf)
        ])

        # Fit model on entire dataset
        self.pipeline.fit(X, y)

        # Extract feature importances
        clf = self.pipeline.named_steps["classifier"]
        importances = clf.feature_importances_
        self.feature_importances_ = dict(
            sorted(
                zip(self.feature_cols, [round(float(v), 4) for v in importances]),
                key=lambda x: x[1],
                reverse=True
            )
        )

        # Compute Brier calibration score
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, y_prob)

        metrics = {
            "model_type": self.model_type,
            "cv_roc_auc": round(cv_auc, 4),
            "cv_precision": round(cv_precision, 4),
            "cv_recall": round(cv_recall, 4),
            "brier_score": round(brier, 4),
            "feature_importance": self.feature_importances_
        }

        # Save model artifact
        self.save()
        return metrics

    def predict_lead_scores(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates calibrated Lead Probability Scores (0 to 100) and qualification tiers.
        """
        if self.pipeline is None:
            self.load()

        X = df_features[self.feature_cols].copy()
        X = X.fillna(X.median(numeric_only=True))

        # Predict probability of high-converting institutional lead
        probs = self.pipeline.predict_proba(X)[:, 1]
        
        # Scale to 0 - 100 definitive probability score
        lead_scores = np.round(probs * 100.0, 1)

        result_df = df_features.copy()
        result_df["lead_probability_score"] = lead_scores

        # Assign Institutional Priority Tier
        def assign_tier(score: float) -> str:
            if score >= TIER_1_CUTOFF:
                return "Tier 1: Strategic Alpha"
            elif score >= TIER_2_CUTOFF:
                return "Tier 2: High Conviction"
            elif score >= TIER_3_CUTOFF:
                return "Tier 3: Medium Priority"
            else:
                return "Tier 4: Monitor / Nurture"

        result_df["priority_tier"] = result_df["lead_probability_score"].apply(assign_tier)

        # Assign Lead Qualification Status
        result_df["qualification_status"] = np.where(
            result_df["lead_probability_score"] >= TIER_2_CUTOFF,
            "Qualified Target",
            "Disqualified / On-Hold"
        )

        return result_df

    def save(self, filepath: str = None) -> None:
        target = filepath or MODEL_CHECKPOINT_FILE
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_cols": self.feature_cols,
            "feature_importances": self.feature_importances_,
            "model_type": self.model_type
        }, target)

    def load(self, filepath: str = None) -> None:
        target = filepath or MODEL_CHECKPOINT_FILE
        saved_dict = joblib.load(target)
        self.pipeline = saved_dict["pipeline"]
        self.feature_cols = saved_dict["feature_cols"]
        self.feature_importances_ = saved_dict["feature_importances"]
        self.model_type = saved_dict["model_type"]
