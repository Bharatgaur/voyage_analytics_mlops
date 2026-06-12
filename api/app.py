"""
api/app.py
==========
Flask REST API serving all three ML models:
  POST /predict-flight-price   → Regression
  POST /predict-gender         → Classification
  POST /recommend-hotels       → Recommendation

Run locally
-----------
    cd voyage_analytics_mlops
    python api/app.py

Production
----------
    gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model, load_artifact

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests (for Streamlit frontend)

# ── Load models at startup ─────────────────────────────────────────────────────
# Models are loaded once at process start (not per-request) since deserializing
# .pkl files is relatively expensive — doing it on every API call would add
# noticeable latency. Each block is wrapped individually so that, e.g., a
# missing recommender model doesn't prevent the price-prediction endpoint from
# working — /health then reports exactly which models are unavailable.
logger.info("Loading ML models …")
try:
    flight_model    = load_model("flight_price_model")
    flight_encoders = load_artifact("flight_encoders")
    flight_scaler   = load_artifact("flight_scaler")
    logger.info("Regression model loaded")
except Exception as e:
    logger.error("Regression model load failed: %s", e)
    flight_model = flight_encoders = flight_scaler = None

try:
    gender_model    = load_model("gender_classifier")
    gender_encoders = load_artifact("gender_encoders")
    gender_scaler   = load_artifact("gender_scaler")
    logger.info("Classification model loaded")
except Exception as e:
    logger.error("Classification model load failed: %s", e)
    gender_model = gender_encoders = gender_scaler = None

try:
    recommender = load_artifact("hotel_recommender")
    logger.info("Recommendation model loaded")
except Exception as e:
    logger.error("Recommendation model load failed: %s", e)
    recommender = None


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status"             : "healthy",
        "regression_model"   : flight_model is not None,
        "classification_model": gender_model is not None,
        "recommendation_model": recommender is not None,
    }), 200


# ==============================================================================
# ENDPOINT 1 — Flight Price Prediction
# ==============================================================================

@app.route("/predict-flight-price", methods=["POST"])
def predict_flight_price():
    """
    Predict the price of a flight.

    Request JSON
    ------------
    {
        "from"        : "Recife (PE)",
        "to"          : "Florianopolis (SC)",
        "flightType"  : "firstClass",   // firstClass | economic | premium
        "agency"      : "FlyingDrops",  // FlyingDrops | CloudFy | Rainbow
        "time"        : 1.76,           // flight duration in hours
        "distance"    : 676.53,         // distance in km
        "month"       : 9,              // 1-12
        "dayofweek"   : 3,              // 0=Mon … 6=Sun
        "is_weekend"  : 0               // 0 or 1
    }

    Response JSON
    -------------
    { "predicted_price": 1423.56, "currency": "USD" }
    """
    if flight_model is None:
        return jsonify({"error": "Regression model not loaded. Run model_training.py first."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        # Required fields
        required = ["from", "to", "flightType", "agency", "time", "distance"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Encode categoricals
        def safe_encode(encoder, value, col_name):
            try:
                return int(encoder.transform([value])[0])
            except ValueError:
                # Unseen label → use most frequent class (index 0).
                # We deliberately don't fail the request here: a new
                # airport/agency not seen during training shouldn't break
                # the API for the user — the model just falls back to the
                # most common category and we log it for monitoring.
                logger.warning("Unseen label '%s' for '%s' — defaulting to 0", value, col_name)
                return 0

        enc = flight_encoders
        from_enc       = safe_encode(enc["from"],            data["from"],       "from")
        to_enc         = safe_encode(enc["to"],              data["to"],         "to")
        flight_enc     = safe_encode(enc["flightType"],      data["flightType"], "flightType")
        agency_enc     = safe_encode(enc["agency"],          data["agency"],     "agency")

        time     = float(data["time"])
        distance = float(data["distance"])

        # Derived features
        speed_kmph   = distance / time if time > 0 else 0
        # price_per_km is a placeholder: it was a feature used during training
        # (derived from the actual price), but at inference time the price is
        # exactly what we're trying to predict, so a neutral constant (1.0) is
        # substituted. The model still relies primarily on distance, time,
        # class, and agency for its prediction.
        price_per_km = 1.0
        month        = int(data.get("month",      1))
        dayofweek    = int(data.get("dayofweek",  0))
        is_weekend   = int(data.get("is_weekend", 0))

        # Distance bucket
        if   distance <= 500 : dbucket = "short"
        elif distance <= 1000: dbucket = "medium"
        elif distance <= 2000: dbucket = "long"
        else                 : dbucket = "ultra"
        db_enc = safe_encode(enc["distance_bucket"], dbucket, "distance_bucket")

        feature_names = [
            "from_enc", "to_enc", "flightType_enc", "agency_enc",
            "distance_bucket_enc", "time", "distance",
            "month", "dayofweek", "is_weekend",
            "speed_kmph", "price_per_km"
        ]

        X = pd.DataFrame([[
            from_enc, to_enc, flight_enc, agency_enc,
            db_enc, time, distance,
            month, dayofweek, is_weekend,
            speed_kmph, price_per_km
        ]], columns=feature_names)

        X_scaled = pd.DataFrame(
            flight_scaler.transform(X), columns=feature_names
        )

        predicted_price = float(flight_model.predict(X_scaled)[0])
        predicted_price = round(max(predicted_price, 0), 2)

        logger.info("Flight price prediction → $%.2f", predicted_price)
        return jsonify({"predicted_price": predicted_price, "currency": "USD"}), 200

    except Exception as e:
        logger.exception("Error in /predict-flight-price")
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# ENDPOINT 2 — Gender Prediction
# ==============================================================================

@app.route("/predict-gender", methods=["POST"])
def predict_gender():
    """
    Predict user gender based on travel behaviour.

    Request JSON
    ------------
    {
        "age"                   : 30,
        "company"               : "4You",
        "total_flights"         : 10,
        "avg_flight_price"      : 950.0,
        "total_flight_spend"    : 9500.0,
        "avg_distance"          : 700.0,
        "first_class_count"     : 3,
        "economic_count"        : 5,
        "premium_count"         : 2,
        "first_class_ratio"     : 0.3,
        "total_hotel_bookings"  : 8,
        "avg_hotel_price"       : 250.0,
        "total_hotel_spend"     : 5000.0,
        "avg_days_stayed"       : 3.5
    }

    Response JSON
    -------------
    { "predicted_gender": "male", "confidence": 0.83 }
    """
    if gender_model is None:
        return jsonify({"error": "Classification model not loaded. Run model_training.py first."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        def safe_encode(encoder, value, col_name):
            try:
                return int(encoder.transform([value])[0])
            except ValueError:
                return 0

        company_enc = safe_encode(gender_encoders["company"], str(data.get("company", "")), "company")

        feature_names = [
            "age", "company_enc",
            "total_flights", "avg_flight_price", "total_flight_spend",
            "avg_distance", "first_class_count", "economic_count", "premium_count",
            "first_class_ratio",
            "total_hotel_bookings", "avg_hotel_price", "total_hotel_spend",
            "avg_days_stayed"
        ]

        X = pd.DataFrame([[
            float(data.get("age", 30)),
            company_enc,
            float(data.get("total_flights", 0)),
            float(data.get("avg_flight_price", 0)),
            float(data.get("total_flight_spend", 0)),
            float(data.get("avg_distance", 0)),
            float(data.get("first_class_count", 0)),
            float(data.get("economic_count", 0)),
            float(data.get("premium_count", 0)),
            float(data.get("first_class_ratio", 0)),
            float(data.get("total_hotel_bookings", 0)),
            float(data.get("avg_hotel_price", 0)),
            float(data.get("total_hotel_spend", 0)),
            float(data.get("avg_days_stayed", 0)),
        ]], columns=feature_names)

        X_scaled = pd.DataFrame(gender_scaler.transform(X), columns=feature_names)

        pred_enc    = gender_model.predict(X_scaled)[0]
        predicted   = gender_encoders["gender"].inverse_transform([pred_enc])[0]
        confidence  = float(np.max(gender_model.predict_proba(X_scaled)[0]))

        logger.info("Gender prediction → %s (confidence=%.2f)", predicted, confidence)
        return jsonify({"predicted_gender": predicted, "confidence": round(confidence, 4)}), 200

    except Exception as e:
        logger.exception("Error in /predict-gender")
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# ENDPOINT 3 — Hotel Recommendation
# ==============================================================================

@app.route("/recommend-hotels", methods=["POST"])
def recommend_hotels():
    """
    Recommend hotels for a user.

    Request JSON
    ------------
    {
        "user_code"    : 42,
        "method"       : "hybrid",      // hybrid | collaborative | content
        "hotel_name"   : "Hotel A",     // required only for method=content
        "top_n"        : 5
    }

    Response JSON
    -------------
    {
        "user_code": 42,
        "recommendations": [
            {"hotel_name": "Hotel K", "avg_price": 263.41, "avg_days": 2.5, "score": 0.87, "method": "..."},
            ...
        ]
    }
    """
    if recommender is None:
        return jsonify({"error": "Recommendation model not loaded. Run model_training.py first."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        user_code  = int(data.get("user_code", 0))
        method     = str(data.get("method", "hybrid")).lower()
        hotel_name = str(data.get("hotel_name", ""))
        top_n      = int(data.get("top_n", 5))
        # Clamp top_n to [1, 10]: prevents a malicious/typo'd large value from
        # forcing the recommender to sort/return an excessive list, and a
        # value of 0 (or negative) from returning an empty/invalid response.
        top_n      = max(1, min(top_n, 10))

        if method == "collaborative":
            recs = recommender.recommend_collaborative(user_code, top_n)
        elif method == "content":
            if not hotel_name:
                return jsonify({"error": "hotel_name required for content-based method"}), 400
            recs = recommender.recommend_content_based(hotel_name, top_n)
        else:  # hybrid (default)
            recs = recommender.recommend_hybrid(user_code, top_n)

        logger.info("Hotel recommendations for user %d (method=%s): %d results", user_code, method, len(recs))
        return jsonify({"user_code": user_code, "recommendations": recs}), 200

    except Exception as e:
        logger.exception("Error in /recommend-hotels")
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info("Starting Flask API on port %d …", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
