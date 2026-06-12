"""
model_training.py
=================
Trains all three ML models for the Travel ML Project:
  A. Regression   — Flight Price Prediction (Linear Regression, Random Forest, XGBoost)
  B. Classification — Gender Prediction     (Logistic Regression, Random Forest)
  C. Recommendation — Hotel Recommendation  (Content-Based Filtering + Collaborative Filtering)

All experiments are logged with MLflow.
Best models are persisted to the /models directory.

Usage
-----
    python src/model_training.py
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

from sklearn.linear_model   import LinearRegression, LogisticRegression
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost                 import XGBRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing   import MinMaxScaler

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_datasets,
    prepare_flight_regression_data,
    prepare_classification_data,
    prepare_recommendation_data,
)
from src.utils import (
    regression_metrics,
    classification_metrics,
    save_model,
    save_artifact,
    print_section,
    feature_importance_df,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── MLflow setup ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI = f"file://{os.path.join(ROOT_DIR, 'mlruns')}"
mlflow.set_tracking_uri(MLFLOW_URI)


# ==============================================================================
# A. REGRESSION — Flight Price Prediction
# ==============================================================================

def train_regression_models(flights: pd.DataFrame) -> dict:
    """
    Train Linear Regression, Random Forest Regressor, and XGBoost Regressor.
    Perform hyperparameter tuning for RF and XGB using GridSearchCV.
    Log all runs to MLflow.

    Returns
    -------
    results : dict  {model_name: metrics_dict}
    """
    print_section("A. REGRESSION — Flight Price Prediction")

    X_train, X_test, y_train, y_test, encoders, scaler = prepare_flight_regression_data(flights)
    feature_names = list(X_train.columns)

    mlflow.set_experiment("flight_price_regression")
    results = {}
    best_r2 = -np.inf
    best_model = None
    best_name  = ""

    # ── A1. Linear Regression ────────────────────────────────────────────────
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        m = regression_metrics(y_test, y_pred, "LinearRegression")

        mlflow.log_params({"model": "LinearRegression"})
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(lr, "model")
        results["LinearRegression"] = m
        logger.info("Linear Regression → R²: %.4f", m["r2"])

        if m["r2"] > best_r2:
            best_r2, best_model, best_name = m["r2"], lr, "LinearRegression"

    # ── A2. Random Forest Regressor ──────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForestRegressor"):
        # Param grid kept small (2x3x2 = 12 combos) so GridSearchCV stays fast
        # on a laptop; n_estimators capped at 200 since gains beyond that were
        # marginal in earlier exploratory runs (see notebooks/01_eda...).
        rf_params = {
            "n_estimators" : [100, 200],
            "max_depth"    : [10, 20, None],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        # cv=3 (not 5) — trades a little variance for much faster search given
        # the dataset size (~270k flight rows).
        gs_rf = GridSearchCV(rf, rf_params, cv=3, scoring="r2", n_jobs=-1, verbose=0)
        gs_rf.fit(X_train, y_train)
        best_rf   = gs_rf.best_estimator_
        y_pred_rf = best_rf.predict(X_test)
        m = regression_metrics(y_test, y_pred_rf, "RandomForestRegressor")

        mlflow.log_params(gs_rf.best_params_)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(best_rf, "model")
        results["RandomForestRegressor"] = m
        logger.info("Random Forest → Best params: %s | R²: %.4f", gs_rf.best_params_, m["r2"])

        if m["r2"] > best_r2:
            best_r2, best_model, best_name = m["r2"], best_rf, "RandomForestRegressor"

        # Feature importance
        fi = feature_importance_df(best_rf, feature_names)
        logger.info("Top features:\n%s", fi.head(5).to_string(index=False))

    # ── A3. XGBoost Regressor ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="XGBoostRegressor"):
        # learning_rate + n_estimators searched together since they trade off
        # against each other; subsample < 1.0 included to reduce overfitting
        # risk on the dense flight-route features.
        xgb_params = {
            "n_estimators"      : [100, 200],
            "max_depth"         : [4, 6],
            "learning_rate"     : [0.05, 0.1],
            "subsample"         : [0.8, 1.0],
        }
        xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        gs_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring="r2", n_jobs=-1, verbose=0)
        gs_xgb.fit(X_train, y_train)
        best_xgb   = gs_xgb.best_estimator_
        y_pred_xgb = best_xgb.predict(X_test)
        m = regression_metrics(y_test, y_pred_xgb, "XGBoostRegressor")

        mlflow.log_params(gs_xgb.best_params_)
        mlflow.log_metrics(m)
        mlflow.xgboost.log_model(best_xgb, "model")
        results["XGBoostRegressor"] = m
        logger.info("XGBoost → Best params: %s | R²: %.4f", gs_xgb.best_params_, m["r2"])

        if m["r2"] > best_r2:
            best_r2, best_model, best_name = m["r2"], best_xgb, "XGBoostRegressor"

    # ── Save best model and preprocessing artifacts ───────────────────────────
    # Only the single best-R2 model (and its encoders/scaler) is persisted to
    # /models, since the Flask API and Streamlit app only ever serve one
    # "production" regressor at a time. The other runs remain queryable in
    # MLflow for comparison/audit purposes.
    logger.info("\nBest regression model: %s (R²=%.4f)", best_name, best_r2)
    save_model(best_model, "flight_price_model")
    save_artifact(encoders, "flight_encoders")
    save_artifact(scaler,   "flight_scaler")

    # Summary table
    summary = pd.DataFrame(results).T
    print("\nRegression Model Comparison:")
    print(summary.to_string())

    return results


# ==============================================================================
# B. CLASSIFICATION — Gender Prediction
# ==============================================================================

def train_classification_models(users: pd.DataFrame, flights: pd.DataFrame, hotels: pd.DataFrame) -> dict:
    """
    Train Logistic Regression and Random Forest Classifier for gender prediction.
    Log experiments to MLflow.

    Returns
    -------
    results : dict  {model_name: metrics_dict}
    """
    print_section("B. CLASSIFICATION — Gender Prediction")

    X_train, X_test, y_train, y_test, encoders, scaler = prepare_classification_data(users, flights, hotels)

    mlflow.set_experiment("gender_classification")
    results = {}
    best_f1    = -np.inf
    best_model = None
    best_name  = ""

    # f1_weighted is used as the selection metric (instead of accuracy) because
    # the gender classes in this dataset are not perfectly balanced — accuracy
    # alone could be misleadingly high for a model that just predicts the
    # majority class.

    # ── B1. Logistic Regression ───────────────────────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression"):
        lr_params = {"C": [0.01, 0.1, 1, 10], "max_iter": [500]}
        lr = LogisticRegression(random_state=42, solver="lbfgs")
        gs_lr = GridSearchCV(lr, lr_params, cv=5, scoring="f1_weighted", n_jobs=-1)
        gs_lr.fit(X_train, y_train)
        best_lr  = gs_lr.best_estimator_
        y_pred   = best_lr.predict(X_test)
        m = classification_metrics(y_test, y_pred, "LogisticRegression")

        mlflow.log_params(gs_lr.best_params_)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(best_lr, "model")
        results["LogisticRegression"] = m

        if m["f1"] > best_f1:
            best_f1, best_model, best_name = m["f1"], best_lr, "LogisticRegression"

    # ── B2. Random Forest Classifier ──────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForestClassifier"):
        rf_params = {
            "n_estimators": [100, 200],
            "max_depth"   : [5, 10, None],
        }
        rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
        gs_rfc = GridSearchCV(rfc, rf_params, cv=5, scoring="f1_weighted", n_jobs=-1)
        gs_rfc.fit(X_train, y_train)
        best_rfc = gs_rfc.best_estimator_
        y_pred   = best_rfc.predict(X_test)
        m = classification_metrics(y_test, y_pred, "RandomForestClassifier")

        mlflow.log_params(gs_rfc.best_params_)
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(best_rfc, "model")
        results["RandomForestClassifier"] = m

        if m["f1"] > best_f1:
            best_f1, best_model, best_name = m["f1"], best_rfc, "RandomForestClassifier"

    # ── Save best model ───────────────────────────────────────────────────────
    logger.info("\nBest classification model: %s (F1=%.4f)", best_name, best_f1)
    save_model(best_model, "gender_classifier")
    save_artifact(encoders, "gender_encoders")
    save_artifact(scaler,   "gender_scaler")

    summary = pd.DataFrame(results).T
    print("\nClassification Model Comparison:")
    print(summary.to_string())

    return results


# ==============================================================================
# C. RECOMMENDATION SYSTEM — Hotel Recommendation
# ==============================================================================

class HotelRecommender:
    """
    Hybrid hotel recommendation system.

    Approach
    --------
    1. Content-Based Filtering: Uses hotel attributes (place, price range, avg days)
       to find hotels similar to what the user has booked before.
    2. Collaborative Filtering (User-Based): Builds a user-item matrix
       (users × hotels) based on booking frequency and uses cosine similarity
       to find like-minded users and recommend their hotels.

    The final recommendation blends both approaches.
    """

    def __init__(self):
        self.user_item_matrix     = None
        self.user_similarity      = None
        self.content_matrix       = None
        self.hotel_features       = None
        self.hotel_list           = None
        self.user_list            = None
        self.hotel_content_index  = None

    # ── Train ────────────────────────────────────────────────────────────────
    def fit(self, rec_df: pd.DataFrame) -> "HotelRecommender":
        """
        Build both user-item and content matrices from the enriched hotel DataFrame.

        Parameters
        ----------
        rec_df : output of prepare_recommendation_data()
        """
        logger.info("Building recommendation matrices …")

        # ── Collaborative Filtering: User-Item matrix ──────────────────────
        # Value = number of times user booked that hotel
        self.user_item_matrix = (
            rec_df.groupby(["userCode", "hotel_name"])["travelCode"]
            .count()
            .unstack(fill_value=0)
        )

        self.user_list  = self.user_item_matrix.index.tolist()
        self.hotel_list = self.user_item_matrix.columns.tolist()

        # Normalise rows (per user) — without this, users who simply made
        # more total bookings would dominate cosine similarity purely due to
        # larger vector magnitude, not because their *taste* is more similar.
        norm = self.user_item_matrix.values.astype(float)
        row_sums = norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div by zero
        norm /= row_sums

        self.user_similarity = cosine_similarity(norm)

        # ── Content-Based Filtering: Hotel feature matrix ──────────────────
        hotel_agg = (
            rec_df.groupby("hotel_name")
            .agg(
                avg_price  = ("price", "mean"),
                avg_days   = ("days",  "mean"),
                booking_ct = ("travelCode", "count"),
            )
            .reset_index()
        )

        # Normalise for cosine similarity
        scaler = MinMaxScaler()
        hotel_agg[["avg_price_n", "avg_days_n", "booking_ct_n"]] = scaler.fit_transform(
            hotel_agg[["avg_price", "avg_days", "booking_ct"]]
        )

        self.hotel_features = hotel_agg
        feature_mat = hotel_agg[["avg_price_n", "avg_days_n", "booking_ct_n"]].values
        self.content_matrix = cosine_similarity(feature_mat)
        self.hotel_content_index = {name: i for i, name in enumerate(hotel_agg["hotel_name"])}

        logger.info("  User-item matrix: %s", self.user_item_matrix.shape)
        logger.info("  Content matrix  : %s", self.content_matrix.shape)
        return self

    # ── Collaborative Recommendation ──────────────────────────────────────────
    def recommend_collaborative(self, user_code: int, top_n: int = 5) -> list[dict]:
        """
        Find the most similar users and recommend hotels they booked
        that the target user has NOT booked.
        """
        if user_code not in self.user_list:
            logger.warning("User %d not in training data — returning popular hotels.", user_code)
            return self._popular_hotels(top_n)

        user_idx  = self.user_list.index(user_code)
        sim_scores = self.user_similarity[user_idx]

        # Get top-10 similar users (exclude self)
        similar_idx = np.argsort(sim_scores)[::-1][1:11]
        similar_scores = sim_scores[similar_idx]

        # Weighted sum of their hotel bookings
        hotel_scores = np.zeros(len(self.hotel_list))
        for idx, score in zip(similar_idx, similar_scores):
            hotel_scores += score * self.user_item_matrix.iloc[idx].values

        # Zero out hotels the user already booked
        user_bookings = self.user_item_matrix.loc[user_code].values
        hotel_scores[user_bookings > 0] = 0

        # Top N
        top_idx = np.argsort(hotel_scores)[::-1][:top_n]
        recommendations = []
        for i in top_idx:
            hname = self.hotel_list[i]
            info  = self.hotel_features[self.hotel_features["hotel_name"] == hname]
            recommendations.append({
                "hotel_name"  : hname,
                "score"       : round(float(hotel_scores[i]), 4),
                "avg_price"   : round(float(info["avg_price"].values[0]), 2) if len(info) else 0,
                "avg_days"    : round(float(info["avg_days"].values[0]),  2) if len(info) else 0,
                "method"      : "collaborative_filtering",
            })
        return recommendations

    # ── Content-Based Recommendation ─────────────────────────────────────────
    def recommend_content_based(self, hotel_name: str, top_n: int = 5) -> list[dict]:
        """
        Recommend hotels similar in price and stay duration to a given hotel.
        """
        if hotel_name not in self.hotel_content_index:
            logger.warning("Hotel '%s' not found — returning popular hotels.", hotel_name)
            return self._popular_hotels(top_n)

        idx = self.hotel_content_index[hotel_name]
        sim_scores = self.content_matrix[idx]
        top_idx = np.argsort(sim_scores)[::-1][1:top_n + 1]

        recommendations = []
        for i in top_idx:
            row = self.hotel_features.iloc[i]
            recommendations.append({
                "hotel_name": row["hotel_name"],
                "similarity": round(float(sim_scores[i]), 4),
                "avg_price" : round(float(row["avg_price"]), 2),
                "avg_days"  : round(float(row["avg_days"]),  2),
                "method"    : "content_based_filtering",
            })
        return recommendations

    # ── Hybrid ───────────────────────────────────────────────────────────────
    def recommend_hybrid(self, user_code: int, top_n: int = 5) -> list[dict]:
        """
        Blend collaborative and content-based recommendations.
        """
        collab  = self.recommend_collaborative(user_code, top_n)
        if not collab:
            return self._popular_hotels(top_n)

        # Use the top collaborative result as seed for content-based.
        # Rationale: collaborative filtering captures "users like you booked
        # this", while content-based then finds hotels *similar to that one*
        # (by price/duration) — broadening the recommendation pool beyond
        # what similar users have literally booked, while staying anchored
        # to the most personally relevant signal.
        seed_hotel = collab[0]["hotel_name"]
        content    = self.recommend_content_based(seed_hotel, top_n)

        # Merge and deduplicate
        seen   = set()
        merged = []
        for rec in collab + content:
            if rec["hotel_name"] not in seen:
                seen.add(rec["hotel_name"])
                merged.append(rec)

        return merged[:top_n]

    # ── Popularity fallback ───────────────────────────────────────────────────
    def _popular_hotels(self, top_n: int) -> list[dict]:
        """Return the most frequently booked hotels as a fallback."""
        top = (
            self.hotel_features
            .sort_values("booking_ct", ascending=False)
            .head(top_n)
        )
        return [
            {
                "hotel_name": row["hotel_name"],
                "score"     : round(float(row["booking_ct"]), 0),
                "avg_price" : round(float(row["avg_price"]), 2),
                "avg_days"  : round(float(row["avg_days"]),  2),
                "method"    : "popularity_fallback",
            }
            for _, row in top.iterrows()
        ]


def train_recommendation_model(hotels: pd.DataFrame, users: pd.DataFrame) -> HotelRecommender:
    """
    Build and save the hotel recommendation model.
    """
    print_section("C. RECOMMENDATION — Hotel Recommendation System")

    rec_df = prepare_recommendation_data(hotels, users)
    recommender = HotelRecommender().fit(rec_df)

    # Quick sanity check
    sample_user = rec_df["userCode"].iloc[0]
    recs = recommender.recommend_hybrid(sample_user, top_n=3)
    logger.info("Sample recommendations for user %d: %s", sample_user, [r["hotel_name"] for r in recs])

    save_artifact(recommender, "hotel_recommender")
    save_artifact(rec_df,      "recommendation_data")

    mlflow.set_experiment("hotel_recommendation")
    with mlflow.start_run(run_name="HotelRecommender_Hybrid"):
        mlflow.log_param("method", "hybrid_collaborative_content")
        mlflow.log_param("num_users", len(recommender.user_list))
        mlflow.log_param("num_hotels", len(recommender.hotel_list))
        mlflow.log_metric("user_item_matrix_density",
            float((recommender.user_item_matrix > 0).sum().sum()) /
            float(recommender.user_item_matrix.size)
        )

    print(f"\nRecommender trained. Users={len(recommender.user_list)}, Hotels={len(recommender.hotel_list)}")
    return recommender


# ==============================================================================
# MAIN — Run all training
# ==============================================================================

if __name__ == "__main__":
    print_section("TRAVEL ML PROJECT — Full Training Pipeline")

    # Load raw data
    flights, hotels, users = load_datasets()

    # A. Regression
    reg_results  = train_regression_models(flights)

    # B. Classification
    clf_results  = train_classification_models(users, flights, hotels)

    # C. Recommendation
    recommender  = train_recommendation_model(hotels, users)

    print_section("ALL MODELS TRAINED SUCCESSFULLY")
    print("Regression Results  :", reg_results)
    print("Classification Results:", clf_results)
