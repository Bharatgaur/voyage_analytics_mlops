"""
data_processing.py
==================
Complete data processing pipeline for the Travel ML Project.
Handles loading, cleaning, feature engineering, and encoding
for flights, hotels, and users datasets.

Author: Travel ML Project
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RANDOM_STATE = 42


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_datasets(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three raw CSV datasets.

    Returns
    -------
    flights, hotels, users : pd.DataFrame
    """
    logger.info("Loading datasets from: %s", data_dir)

    flights = pd.read_csv(os.path.join(data_dir, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(data_dir, "hotels.csv"))
    users   = pd.read_csv(os.path.join(data_dir, "users.csv"))

    logger.info("Flights : %s", flights.shape)
    logger.info("Hotels  : %s", hotels.shape)
    logger.info("Users   : %s", users.shape)

    return flights, hotels, users


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================

def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the flights dataset.
    - Parse dates
    - Drop duplicates
    - Remove invalid price / distance rows
    """
    logger.info("Cleaning flights dataset …")
    df = df.copy()

    # Parse date column
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("  Dropped %d duplicate rows", before - len(df))

    # Remove rows with non-positive price or distance
    df = df[(df["price"] > 0) & (df["distance"] > 0) & (df["time"] > 0)]

    # Strip whitespace in string columns
    for col in ["from", "to", "flightType", "agency"]:
        df[col] = df[col].str.strip()

    logger.info("  Cleaned flights shape: %s", df.shape)
    return df.reset_index(drop=True)


def clean_hotels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the hotels dataset.
    - Parse dates
    - Remove negative or zero prices
    """
    logger.info("Cleaning hotels dataset …")
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    df = df[(df["price"] > 0) & (df["days"] > 0) & (df["total"] > 0)]

    for col in ["name", "place"]:
        df[col] = df[col].str.strip()

    # Recalculate total for sanity (should be days * price)
    df["total_check"] = (df["days"] * df["price"]).round(2)
    df.drop(columns=["total_check"], inplace=True)

    logger.info("  Cleaned hotels shape: %s", df.shape)
    return df.reset_index(drop=True)


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the users dataset.
    - Standardize gender values
    - Remove users with gender='none' for classification task
    """
    logger.info("Cleaning users dataset …")
    df = df.copy()

    df["gender"] = df["gender"].str.strip().str.lower()
    df["name"]   = df["name"].str.strip()
    df["company"] = df["company"].str.strip()

    logger.info("  Gender distribution:\n%s", df["gender"].value_counts())
    logger.info("  Cleaned users shape: %s", df.shape)
    return df.reset_index(drop=True)


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================

def engineer_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for the flight price regression model.

    New features
    ------------
    - price_per_km    : price efficiency metric
    - speed_kmph      : average speed of the flight
    - month, dayofweek: temporal features from the date column
    - is_weekend      : boolean flag
    - route           : origin + destination combined
    """
    logger.info("Engineering flight features …")
    df = df.copy()

    # Price per km  (avoid div by zero, already cleaned but be safe)
    df["price_per_km"]  = df["price"] / df["distance"].replace(0, np.nan)

    # Speed: distance / time (time is in hours)
    df["speed_kmph"]    = df["distance"] / df["time"].replace(0, np.nan)

    # Temporal features
    df["month"]         = df["date"].dt.month.astype("Int64")
    df["dayofweek"]     = df["date"].dt.dayofweek.astype("Int64")   # 0=Mon, 6=Sun
    df["is_weekend"]    = (df["dayofweek"] >= 5).astype(int)

    # Route string (useful for groupby stats later)
    df["route"]         = df["from"].str.strip() + " → " + df["to"].str.strip()

    # Distance bucket
    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[0, 500, 1000, 2000, np.inf],
        labels=["short", "medium", "long", "ultra"]
    )

    logger.info("  Feature-engineered flights shape: %s", df.shape)
    return df


def engineer_user_features(users: pd.DataFrame, flights: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the users dataset with aggregated travel stats.
    Used for gender classification.
    """
    logger.info("Engineering user features …")

    # ── Flight aggregates per user ──
    flight_agg = flights.groupby("userCode").agg(
        total_flights        = ("travelCode", "count"),
        avg_flight_price     = ("price", "mean"),
        total_flight_spend   = ("price", "sum"),
        avg_distance         = ("distance", "mean"),
        first_class_count    = ("flightType", lambda x: (x == "firstClass").sum()),
        economic_count       = ("flightType", lambda x: (x == "economic").sum()),
        premium_count        = ("flightType", lambda x: (x == "premium").sum()),
    ).reset_index()

    flight_agg["first_class_ratio"] = (
        flight_agg["first_class_count"] / flight_agg["total_flights"]
    )

    # ── Hotel aggregates per user ──
    hotel_agg = hotels.groupby("userCode").agg(
        total_hotel_bookings = ("travelCode", "count"),
        avg_hotel_price      = ("price", "mean"),
        total_hotel_spend    = ("total", "sum"),
        avg_days_stayed      = ("days", "mean"),
    ).reset_index()

    # ── Merge all ──
    df = users.rename(columns={"code": "userCode"})
    df = df.merge(flight_agg, on="userCode", how="left")
    df = df.merge(hotel_agg, on="userCode", how="left")

    # Fill NaNs for users who have no flight/hotel history
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info("  Feature-engineered users shape: %s", df.shape)
    return df


# ==============================================================================
# 4. ENCODING
# ==============================================================================

def encode_flight_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features for flight price regression.

    Returns
    -------
    df_encoded : pd.DataFrame
    encoders   : dict of fitted LabelEncoders (for inference)
    """
    logger.info("Encoding flight categorical features …")
    df = df.copy()
    encoders = {}

    cat_cols = ["from", "to", "flightType", "agency", "distance_bucket"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("  Encoded '%s' → %d classes", col, len(le.classes_))

    return df, encoders


def encode_user_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical features for gender classification.
    Filters out 'none' gender rows (not useful for binary/multi classification).
    """
    logger.info("Encoding user categorical features …")
    df = df.copy()
    encoders = {}

    # Keep only male/female for classification
    df_clean = df[df["gender"].isin(["male", "female"])].copy()

    le_gender = LabelEncoder()
    df_clean["gender_enc"] = le_gender.fit_transform(df_clean["gender"])
    encoders["gender"] = le_gender

    le_company = LabelEncoder()
    df_clean["company_enc"] = le_company.fit_transform(df_clean["company"])
    encoders["company"] = le_company

    logger.info("  Gender classes: %s", le_gender.classes_)
    logger.info("  Encoded users shape: %s", df_clean.shape)
    return df_clean, encoders


# ==============================================================================
# 5. TRAIN / TEST SPLIT HELPERS
# ==============================================================================

def prepare_flight_regression_data(
    flights: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """
    Full pipeline: clean → engineer → encode → split.
    Returns X_train, X_test, y_train, y_test, encoders, scaler
    """
    df = clean_flights(flights)
    df = engineer_flight_features(df)
    df, encoders = encode_flight_data(df)

    feature_cols = [
        "from_enc", "to_enc", "flightType_enc", "agency_enc",
        "distance_bucket_enc", "time", "distance",
        "month", "dayofweek", "is_weekend",
        "speed_kmph", "price_per_km"
    ]

    # Drop rows with NaN in any feature (from date parsing failures etc.)
    df_model = df[feature_cols + ["price"]].dropna()

    X = df_model[feature_cols]
    y = df_model["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_scaled  = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )

    logger.info("Regression data — Train: %s | Test: %s", X_train_scaled.shape, X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), encoders, scaler


def prepare_classification_data(
    users: pd.DataFrame,
    flights: pd.DataFrame,
    hotels: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """
    Full pipeline for gender classification.
    Returns X_train, X_test, y_train, y_test, encoders, scaler
    """
    users_clean  = clean_users(users)
    flights_clean = clean_flights(flights)
    hotels_clean  = clean_hotels(hotels)

    df = engineer_user_features(users_clean, flights_clean, hotels_clean)
    df, encoders = encode_user_data(df)

    feature_cols = [
        "age", "company_enc",
        "total_flights", "avg_flight_price", "total_flight_spend",
        "avg_distance", "first_class_count", "economic_count", "premium_count",
        "first_class_ratio",
        "total_hotel_bookings", "avg_hotel_price", "total_hotel_spend",
        "avg_days_stayed"
    ]

    df_model = df[feature_cols + ["gender_enc"]].dropna()
    X = df_model[feature_cols]
    y = df_model["gender_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  columns=feature_cols)

    logger.info("Classification data — Train: %s | Test: %s", X_train_scaled.shape, X_test_scaled.shape)
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), encoders, scaler


def prepare_recommendation_data(
    hotels: pd.DataFrame,
    users: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge hotels with users to build user-item interaction matrix.
    """
    hotels_clean = clean_hotels(hotels)
    users_clean  = clean_users(users)

    df = hotels_clean.merge(
        users_clean.rename(columns={"code": "userCode"})[["userCode", "name", "gender", "age"]],
        on="userCode", how="left"
    )
    df.rename(columns={"name_x": "hotel_name", "name_y": "user_name"}, inplace=True)
    return df


# ==============================================================================
# MAIN — Quick sanity check
# ==============================================================================
if __name__ == "__main__":
    flights, hotels, users = load_datasets()
    X_tr, X_te, y_tr, y_te, enc, sc = prepare_flight_regression_data(flights)
    print("\nRegression data ready:", X_tr.shape)

    X_tr2, X_te2, y_tr2, y_te2, enc2, sc2 = prepare_classification_data(users, flights, hotels)
    print("Classification data ready:", X_tr2.shape)

    rec_df = prepare_recommendation_data(hotels, users)
    print("Recommendation data ready:", rec_df.shape)
