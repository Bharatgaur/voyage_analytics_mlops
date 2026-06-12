"""
utils.py
========
Shared utility functions used across the Travel ML Project.
Includes model persistence, metric computation, and logging helpers.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# MODEL PERSISTENCE
# ==============================================================================

def save_model(model, name: str) -> str:
    """Save a sklearn / xgboost model to disk using joblib."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str):
    """Load a saved model from disk."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


def save_artifact(obj, name: str) -> str:
    """Save any Python object (encoders, scalers, etc.) using joblib."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(obj, path)
    logger.info("Artifact saved → %s", path)
    return path


def load_artifact(name: str):
    """Load a saved artifact (encoder, scaler, etc.)."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


# ==============================================================================
# REGRESSION METRICS
# ==============================================================================

def regression_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Compute RMSE, MAE, and R² for a regression model.

    Parameters
    ----------
    y_true     : array-like of true target values
    y_pred     : array-like of predicted values
    model_name : optional label for logging

    Returns
    -------
    dict with keys: rmse, mae, r2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

    label = f"[{model_name}] " if model_name else ""
    logger.info("%sRMSE=%.4f | MAE=%.4f | R²=%.4f", label, rmse, mae, r2)
    return metrics


# ==============================================================================
# CLASSIFICATION METRICS
# ==============================================================================

def classification_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Compute Accuracy, Precision, Recall, and F1 for a classification model.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
    }

    label = f"[{model_name}] " if model_name else ""
    logger.info(
        "%sAccuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f",
        label, acc, prec, rec, f1
    )
    logger.info("\n%s", classification_report(y_true, y_pred, zero_division=0))
    return metrics


# ==============================================================================
# GENERAL HELPERS
# ==============================================================================

def print_section(title: str) -> None:
    """Print a formatted section header to stdout."""
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}")


def feature_importance_df(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a tree-based model (RF, XGB).
    Returns a sorted DataFrame.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute.")

    fi = pd.DataFrame({
        "feature"   : feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi
