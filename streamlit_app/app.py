"""
streamlit_app/app.py
====================
Interactive Streamlit Web Application for the Travel ML Project.

Features:
  - Flight Price Prediction (calls Flask API)
  - Hotel Recommendation (calls Flask API / direct model)
  - Interactive visualizations from the datasets
  - EDA Dashboard

Run:
    streamlit run streamlit_app/app.py
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ── Configuration ────────────────────────────────────────────────────────────
API_URL  = os.environ.get("API_URL", "http://localhost:5000")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Travel ML Dashboard",
    page_icon  = "✈️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    h1 { color: #667eea; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING (cached)
# ==============================================================================

@st.cache_data
def load_data():
    """Load all three datasets with caching."""
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels  = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    users   = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y", errors="coerce")
    hotels["date"]  = pd.to_datetime(hotels["date"],  format="%m/%d/%Y", errors="coerce")
    return flights, hotels, users


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

st.sidebar.title("Travel ML Project")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview & EDA", "Flight Price Predictor", "Hotel Recommender", "Analytics Dashboard"]
)

flights, hotels, users = load_data()


# ==============================================================================
# PAGE 1 — Overview & EDA
# ==============================================================================

if page == "Overview & EDA":
    st.title("Travel ML Project Dashboard")
    st.markdown("##### End-to-End Machine Learning for Travel Insights")
    st.markdown("---")

    # ── Summary Metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flights",  f"{len(flights):,}")
    with col2:
        st.metric("Total Hotels",   f"{len(hotels):,}")
    with col3:
        st.metric("Total Users",    f"{len(users):,}")
    with col4:
        st.metric("Avg Flight Price", f"${flights['price'].mean():.0f}")

    st.markdown("---")

    # ── Row 1: Price Distribution + Flight Types ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Flight Price Distribution")
        fig = px.histogram(
            flights, x="price", nbins=60,
            color="flightType",
            title="Price Distribution by Flight Type",
            labels={"price": "Price (USD)", "count": "Number of Flights"},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Flight Type Breakdown")
        type_counts = flights["flightType"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        fig2 = px.pie(
            type_counts, values="count", names="type",
            title="Proportion of Flight Types",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Price by Agency + Origin Routes ────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Average Price by Agency")
        agency_avg = flights.groupby("agency")["price"].mean().reset_index()
        fig3 = px.bar(
            agency_avg, x="agency", y="price",
            title="Average Flight Price by Agency",
            color="price",
            color_continuous_scale="Viridis",
            labels={"price": "Avg Price (USD)"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.subheader("Price vs Distance")
        sample = flights.sample(min(3000, len(flights)), random_state=42)
        fig4 = px.scatter(
            sample, x="distance", y="price",
            color="flightType",
            title="Price vs Distance (sample of 3,000)",
            opacity=0.6,
            labels={"distance": "Distance (km)", "price": "Price (USD)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: User Demographics ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("👥 User Demographics")
    col_e, col_f, col_g = st.columns(3)

    with col_e:
        gender_counts = users["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        fig5 = px.pie(gender_counts, values="count", names="gender",
                      title="Gender Distribution", hole=0.4,
                      color_discrete_sequence=["#667eea", "#f7931e", "#aaa"])
        st.plotly_chart(fig5, use_container_width=True)

    with col_f:
        fig6 = px.histogram(users, x="age", nbins=30,
                            title="Age Distribution",
                            color_discrete_sequence=["#764ba2"])
        st.plotly_chart(fig6, use_container_width=True)

    with col_g:
        company_counts = users["company"].value_counts().reset_index()
        company_counts.columns = ["company", "count"]
        fig7 = px.bar(company_counts, x="company", y="count",
                      title="Users per Company",
                      color="count", color_continuous_scale="Blues")
        st.plotly_chart(fig7, use_container_width=True)


# ==============================================================================
# PAGE 2 — Flight Price Predictor
# ==============================================================================

elif page == "Flight Price Predictor":
    st.title("Flight Price Predictor")
    st.markdown("Enter flight details to predict the price in real-time.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        from_city   = st.selectbox("Origin City",
            sorted(flights["from"].unique()), index=0)
        flight_type = st.selectbox("Flight Class",
            ["firstClass", "economic", "premium"])
        time_hrs    = st.slider("Flight Duration (hours)", 0.5, 12.0, 2.0, 0.25)
        month       = st.slider("Month", 1, 12, 6)

    with col2:
        to_city     = st.selectbox("Destination City",
            sorted(flights["to"].unique()), index=1)
        agency      = st.selectbox("Agency",
            flights["agency"].unique())
        distance    = st.slider("📏 Distance (km)", 100, 5000, 700, 50)
        is_weekend  = st.checkbox("Is Weekend?")

    st.markdown("---")

    if st.button("Predict Flight Price"):
        payload = {
            "from"       : from_city,
            "to"         : to_city,
            "flightType" : flight_type,
            "agency"     : agency,
            "time"       : time_hrs,
            "distance"   : distance,
            "month"      : month,
            "dayofweek"  : 5 if is_weekend else 2,
            "is_weekend" : int(is_weekend),
        }

        with st.spinner("Calling prediction API …"):
            try:
                response = requests.post(
                    f"{API_URL}/predict-flight-price",
                    json=payload, timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    price  = result["predicted_price"]

                    st.success(f"### Predicted Flight Price: **${price:,.2f}**")

                    # Comparison with class averages
                    avg_prices = flights.groupby("flightType")["price"].mean()
                    class_avg  = avg_prices.get(flight_type, 957)

                    diff = ((price - class_avg) / class_avg) * 100
                    if diff > 0:
                        st.info(f"This is **{diff:.1f}% above** average for {flight_type} class (avg: ${class_avg:.0f})")
                    else:
                        st.info(f"This is **{abs(diff):.1f}% below** average for {flight_type} class (avg: ${class_avg:.0f})")

                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode  = "gauge+number+delta",
                        value = price,
                        delta = {"reference": class_avg},
                        title = {"text": "Predicted Price vs Class Average (USD)"},
                        gauge = {
                            "axis"  : {"range": [300, 1800]},
                            "bar"   : {"color": "#667eea"},
                            "steps" : [
                                {"range": [300,  700], "color": "#d4edda"},
                                {"range": [700, 1200], "color": "#fff3cd"},
                                {"range": [1200, 1800], "color": "#f8d7da"},
                            ],
                            "threshold": {
                                "line" : {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": class_avg
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                else:
                    st.error(f"API Error: {response.json().get('error', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                # Fallback: direct model prediction
                st.warning("API not reachable. Attempting direct model prediction …")
                try:
                    from src.utils import load_model, load_artifact
                    model    = load_model("flight_price_model")
                    encoders = load_artifact("flight_encoders")
                    scaler   = load_artifact("flight_scaler")

                    def safe_enc(enc, val):
                        try: return int(enc.transform([val])[0])
                        except: return 0

                    speed = distance / time_hrs if time_hrs > 0 else 0
                    if distance <= 500: db = "short"
                    elif distance <= 1000: db = "medium"
                    elif distance <= 2000: db = "long"
                    else: db = "ultra"

                    X = pd.DataFrame([[
                        safe_enc(encoders["from"], from_city),
                        safe_enc(encoders["to"], to_city),
                        safe_enc(encoders["flightType"], flight_type),
                        safe_enc(encoders["agency"], agency),
                        safe_enc(encoders["distance_bucket"], db),
                        time_hrs, distance, month,
                        5 if is_weekend else 2, int(is_weekend),
                        speed, 1.0
                    ]], columns=[
                        "from_enc","to_enc","flightType_enc","agency_enc",
                        "distance_bucket_enc","time","distance",
                        "month","dayofweek","is_weekend","speed_kmph","price_per_km"
                    ])
                    X_sc = pd.DataFrame(scaler.transform(X), columns=X.columns)
                    pred = float(model.predict(X_sc)[0])
                    st.success(f"### Predicted Price (direct): **${max(pred,0):,.2f}**")
                except Exception as e:
                    st.error(f"Direct prediction also failed: {e}. Please run model_training.py first.")


# ==============================================================================
# PAGE 3 — Hotel Recommender
# ==============================================================================

elif page == "Hotel Recommender":
    st.title("Hotel Recommendation System")
    st.markdown("Get personalized hotel recommendations based on your travel history.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        user_code  = st.number_input(" User Code", min_value=0,
                                      max_value=int(users["code"].max()),
                                      value=0, step=1)
        method     = st.selectbox("Recommendation Method",
                                   ["hybrid", "collaborative", "content"])
        top_n      = st.slider("Number of Recommendations", 1, 10, 5)

    with col2:
        hotel_name = st.selectbox("Seed Hotel (for content-based)",
                                   sorted(hotels["name"].unique()))
        st.markdown("&nbsp;")
        st.info("**Hybrid** blends Collaborative Filtering + Content-Based Filtering for best results.")

    st.markdown("---")

    if st.button("Get Recommendations"):
        payload = {
            "user_code" : int(user_code),
            "method"    : method,
            "hotel_name": hotel_name if method == "content" else "",
            "top_n"     : top_n
        }

        with st.spinner("Getting recommendations …"):
            try:
                response = requests.post(
                    f"{API_URL}/recommend-hotels",
                    json=payload, timeout=10
                )
                if response.status_code == 200:
                    recs = response.json()["recommendations"]
                    if recs:
                        st.success(f"Top {len(recs)} recommended hotels for User {user_code}:")
                        rec_df = pd.DataFrame(recs)
                        rec_df.index = range(1, len(rec_df) + 1)
                        rec_df.columns = [c.replace("_", " ").title() for c in rec_df.columns]
                        st.dataframe(rec_df, use_container_width=True)

                        # Bar chart of avg prices
                        fig = px.bar(
                            rec_df, x="Hotel Name", y="Avg Price",
                            title="Recommended Hotels — Average Price per Night",
                            color="Avg Price",
                            color_continuous_scale="Purpor",
                            labels={"Avg Price": "Avg Price/Night (USD)"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No recommendations found for this user.")
                else:
                    st.error(f"API Error: {response.json().get('error', 'Unknown')}")

            except requests.exceptions.ConnectionError:
                st.warning("API not reachable. Loading recommender directly …")
                try:
                    from src.utils import load_artifact
                    rec = load_artifact("hotel_recommender")
                    if method == "collaborative":
                        recs = rec.recommend_collaborative(int(user_code), top_n)
                    elif method == "content":
                        recs = rec.recommend_content_based(hotel_name, top_n)
                    else:
                        recs = rec.recommend_hybrid(int(user_code), top_n)

                    rec_df = pd.DataFrame(recs)
                    rec_df.index = range(1, len(rec_df) + 1)
                    st.dataframe(rec_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Direct load failed: {e}. Please run model_training.py first.")

    # ── Popular Hotels Section ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌟 Most Popular Hotels (by booking frequency)")
    hotel_pop = hotels["name"].value_counts().reset_index()
    hotel_pop.columns = ["Hotel", "Bookings"]
    fig_pop = px.bar(
        hotel_pop, x="Hotel", y="Bookings",
        title="Hotel Popularity",
        color="Bookings", color_continuous_scale="Teal"
    )
    st.plotly_chart(fig_pop, use_container_width=True)


# ==============================================================================
# PAGE 4 — Analytics Dashboard
# ==============================================================================

elif page == "Analytics Dashboard":
    st.title("Travel Analytics Dashboard")
    st.markdown("Deep-dive insights from the travel dataset.")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    st.sidebar.subheader("Filters")
    selected_types = st.sidebar.multiselect(
        "Flight Types", flights["flightType"].unique(),
        default=list(flights["flightType"].unique())
    )
    selected_agencies = st.sidebar.multiselect(
        "Agencies", flights["agency"].unique(),
        default=list(flights["agency"].unique())
    )

    filtered = flights[
        (flights["flightType"].isin(selected_types)) &
        (flights["agency"].isin(selected_agencies))
    ]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Filtered Flights", f"{len(filtered):,}")
    kpi2.metric("Avg Price",        f"${filtered['price'].mean():,.0f}")
    kpi3.metric("Avg Distance",     f"{filtered['distance'].mean():,.0f} km")
    kpi4.metric("Avg Duration",     f"{filtered['time'].mean():.1f} hrs")

    st.markdown("---")

    # ── Monthly trends ────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        monthly = filtered.copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_avg = monthly.groupby("month")["price"].mean().reset_index()
        fig = px.line(monthly_avg, x="month", y="price",
                      title="Monthly Average Flight Price Trend",
                      markers=True,
                      labels={"price": "Avg Price (USD)", "month": "Month"})
        fig.update_traces(line_color="#667eea")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        route_df = filtered.copy()
        route_df["route"] = route_df["from"].str[:10] + " → " + route_df["to"].str[:10]
        top_routes = route_df["route"].value_counts().head(10).reset_index()
        top_routes.columns = ["route", "count"]
        fig2 = px.bar(top_routes, x="count", y="route", orientation="h",
                      title="Top 10 Most Popular Routes",
                      color="count", color_continuous_scale="Viridis")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Price heatmap: origin vs flight type ─────────────────────────────────
    st.subheader("🌡️ Price Heatmap — Origin City × Flight Type")
    pivot = filtered.pivot_table(values="price", index="from", columns="flightType", aggfunc="mean")
    fig3 = px.imshow(pivot, text_auto=".0f", color_continuous_scale="RdYlGn",
                     title="Average Price: Origin × Flight Class",
                     labels={"color": "Avg Price (USD)"})
    st.plotly_chart(fig3, use_container_width=True)

    # ── Hotel insights ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Hotel Analytics")
    col_c, col_d = st.columns(2)

    with col_c:
        hotel_avg = hotels.groupby("place")["price"].mean().reset_index()
        fig4 = px.bar(hotel_avg.sort_values("price", ascending=False),
                      x="place", y="price",
                      title="Average Hotel Price by City",
                      color="price", color_continuous_scale="Reds",
                      labels={"price": "Avg Price/Night (USD)"})
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        fig5 = px.scatter(hotels, x="days", y="total",
                          color="name",
                          title="Hotel Stay Duration vs Total Cost",
                          labels={"days": "Days Stayed", "total": "Total Cost (USD)"},
                          opacity=0.6,
                          color_discrete_sequence=px.colors.qualitative.Plotly)
        st.plotly_chart(fig5, use_container_width=True)

    # ── Raw data viewer ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("View Raw Data"):
        tab1, tab2, tab3 = st.tabs(["Flights", "Hotels", "Users"])
        with tab1:
            st.dataframe(filtered.head(500), use_container_width=True)
        with tab2:
            st.dataframe(hotels.head(200), use_container_width=True)
        with tab3:
            st.dataframe(users.head(100), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa;'>Travel ML Project — End-to-End MLOps Pipeline | Built with Streamlit, Flask, MLflow</p>",
    unsafe_allow_html=True
)
