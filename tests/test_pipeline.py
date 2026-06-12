"""
tests/test_pipeline.py
======================
Unit tests for the Travel ML Project.
Tests data processing, model training, and API endpoints.

Run:
    pytest tests/ -v --tb=short
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    clean_flights,
    clean_hotels,
    clean_users,
    engineer_flight_features,
    encode_flight_data,
    prepare_flight_regression_data,
)
from src.utils import regression_metrics, classification_metrics


# ==============================================================================
# FIXTURES — Sample data frames
# ==============================================================================

@pytest.fixture
def sample_flights():
    """Small synthetic flights DataFrame for unit testing."""
    return pd.DataFrame({
        "travelCode": [1, 2, 3, 4],
        "userCode"  : [0, 0, 1, 1],
        "from"      : ["Recife (PE)", "Brasilia (DF)", "Recife (PE)", "Salvador (BH)"],
        "to"        : ["Florianopolis (SC)", "Recife (PE)", "Salvador (BH)", "Brasilia (DF)"],
        "flightType": ["firstClass", "economic", "premium", "firstClass"],
        "price"     : [1434.38, 500.00, 800.00, 1200.00],
        "time"      : [1.76, 2.0, 1.5, 2.5],
        "distance"  : [676.53, 900.0, 400.0, 1100.0],
        "agency"    : ["FlyingDrops", "CloudFy", "Rainbow", "FlyingDrops"],
        "date"      : ["09/26/2019", "10/03/2019", "11/15/2019", "12/01/2019"],
    })


@pytest.fixture
def sample_hotels():
    return pd.DataFrame({
        "travelCode": [1, 2, 3],
        "userCode"  : [0, 0, 1],
        "name"      : ["Hotel A", "Hotel K", "Hotel Z"],
        "place"     : ["Florianopolis (SC)", "Salvador (BH)", "Brasilia (DF)"],
        "days"      : [4, 2, 3],
        "price"     : [313.02, 263.41, 200.00],
        "total"     : [1252.08, 526.82, 600.00],
        "date"      : ["09/26/2019", "10/10/2019", "11/20/2019"],
    })


@pytest.fixture
def sample_users():
    return pd.DataFrame({
        "code"   : [0, 1, 2],
        "company": ["4You", "4You", "Hotmart"],
        "name"   : ["Roy Braun", "Joseph Holsten", "Wilma Mcinnis"],
        "gender" : ["male", "male", "female"],
        "age"    : [21, 37, 48],
    })


# ==============================================================================
# DATA PROCESSING TESTS
# ==============================================================================

class TestCleanFlights:
    def test_shape_preserved(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        assert len(cleaned) == len(sample_flights)

    def test_date_parsed(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])

    def test_no_negative_prices(self, sample_flights):
        # Insert a bad row
        bad = sample_flights.copy()
        bad.loc[0, "price"] = -100
        cleaned = clean_flights(bad)
        assert (cleaned["price"] > 0).all()

    def test_strips_whitespace(self, sample_flights):
        df = sample_flights.copy()
        df.loc[0, "from"] = "  Recife (PE)  "
        cleaned = clean_flights(df)
        assert cleaned.loc[0, "from"] == "Recife (PE)"


class TestCleanUsers:
    def test_gender_lowercase(self, sample_users):
        df = sample_users.copy()
        df.loc[0, "gender"] = "Male"
        cleaned = clean_users(df)
        assert cleaned["gender"].str.islower().all()

    def test_shape_preserved(self, sample_users):
        cleaned = clean_users(sample_users)
        assert len(cleaned) == len(sample_users)


class TestEngineerFlightFeatures:
    def test_new_columns_added(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        engineered = engineer_flight_features(cleaned)
        expected_cols = ["price_per_km", "speed_kmph", "month", "dayofweek", "is_weekend", "route"]
        for col in expected_cols:
            assert col in engineered.columns, f"Missing column: {col}"

    def test_speed_positive(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        engineered = engineer_flight_features(cleaned)
        assert (engineered["speed_kmph"] > 0).all()

    def test_is_weekend_binary(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        engineered = engineer_flight_features(cleaned)
        assert set(engineered["is_weekend"].unique()).issubset({0, 1})


class TestEncode:
    def test_encoded_columns_created(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        engineered = engineer_flight_features(cleaned)
        encoded, encoders = encode_flight_data(engineered)
        assert "from_enc" in encoded.columns
        assert "flightType_enc" in encoded.columns
        assert "agency" in encoders

    def test_encoder_inverse_works(self, sample_flights):
        cleaned = clean_flights(sample_flights)
        engineered = engineer_flight_features(cleaned)
        _, encoders = encode_flight_data(engineered)
        original = list(sample_flights["flightType"].str.strip().unique())
        for val in original:
            enc = encoders["flightType"].transform([val])
            dec = encoders["flightType"].inverse_transform(enc)
            assert dec[0] == val


# ==============================================================================
# METRICS TESTS
# ==============================================================================

class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([100, 200, 300, 400])
        m = regression_metrics(y, y)
        assert m["rmse"] == 0.0
        assert m["mae"]  == 0.0
        assert m["r2"]   == 1.0

    def test_keys_present(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.1])
        m = regression_metrics(y_true, y_pred)
        assert "rmse" in m
        assert "mae"  in m
        assert "r2"   in m

    def test_rmse_non_negative(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([90,  210, 290])
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] >= 0


class TestClassificationMetrics:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        m = classification_metrics(y, y)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_keys_present(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])
        m = classification_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in m


# ==============================================================================
# END-TO-END DATA PIPELINE TEST
# ==============================================================================

class TestFullPipeline:
    def test_regression_data_shapes(self, sample_flights):
        """X_train must have the correct number of feature columns."""
        # Use a larger synthetic dataset to avoid split edge cases
        big_flights = pd.concat([sample_flights] * 20, ignore_index=True)
        X_tr, X_te, y_tr, y_te, enc, sc = prepare_flight_regression_data(big_flights)
        assert X_tr.shape[1] == 12     # 12 feature columns
        assert len(X_tr) + len(X_te) == len(y_tr) + len(y_te)

    def test_no_nan_in_features(self, sample_flights):
        """Processed features should have no NaN values."""
        big_flights = pd.concat([sample_flights] * 20, ignore_index=True)
        X_tr, X_te, *_ = prepare_flight_regression_data(big_flights)
        assert X_tr.isnull().sum().sum() == 0
        assert X_te.isnull().sum().sum() == 0
