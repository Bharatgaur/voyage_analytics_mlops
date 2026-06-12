# Voyage Analytics: Integrating MLOps in Travel — End-to-End MLOps Pipeline

> Capstone Project 1 — Productionization of ML Systems | A production-ready Machine Learning system for flight price prediction, gender classification, and hotel recommendation — fully containerized, orchestrated, and deployed.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Project Architecture](#project-architecture)
4. [Dataset Description](#dataset-description)
5. [ML Models](#ml-models)
6. [API Documentation](#api-documentation)
7. [Installation & Setup](#installation--setup)
8. [Running the Project](#running-the-project)
9. [Docker Deployment](#docker-deployment)
10. [Kubernetes Deployment](#kubernetes-deployment)
11. [Apache Airflow](#apache-airflow)
12. [CI/CD with Jenkins](#cicd-with-jenkins)
13. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
14. [Streamlit Web App](#streamlit-web-app)
15. [Project Structure](#project-structure)
16. [Results](#results)

---

## Project Overview

This project demonstrates a complete, industry-grade MLOps pipeline applied to a travel domain dataset. It covers the full lifecycle from raw data to deployed, monitored ML services.

**Three ML tasks are solved:**
- **Regression** — Predict flight prices using distance, time, class, and agency
- **Classification** — Predict user gender from travel behavior patterns
- **Recommendation** — Suggest hotels using collaborative + content-based filtering

**MLOps stack used:**
| Tool | Purpose |
|---|---|
| scikit-learn + XGBoost | Model development |
| MLflow | Experiment tracking & model versioning |
| Flask | REST API serving |
| Docker | Containerization |
| Kubernetes | Orchestration & scaling |
| Apache Airflow | Workflow automation |
| Jenkins | CI/CD pipeline |
| Streamlit | Interactive web UI |

---

## Business Problem

Travel companies have massive amounts of trip data but struggle to answer three key questions:

1. **"How much should this flight cost?"** — Dynamic pricing and customer expectation management
2. **"Who is this user?"** — Personalization requires knowing demographic context
3. **"Where should we recommend?"** — Hotel bookings increase when recommendations are relevant

This project builds automated, data-driven answers to all three questions.

---

## Project Architecture

```
Raw Data (CSV)
     │
     ▼
┌────────────────────────────┐
│  Data Pipeline (Airflow)   │  ← Runs daily
│  - Load → Clean → Engineer │
│  - Validate → Store        │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Model Training (MLflow)   │  ← Runs weekly
│  - Regression (XGBoost)    │
│  - Classification (RF)     │
│  - Recommendation System   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Flask REST API            │  ← 3 endpoints
│  /predict-flight-price     │
│  /predict-gender           │
│  /recommend-hotels         │
└────────────┬───────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│ Docker  │    │  Streamlit   │
│Container│    │  Web App     │
└────┬────┘    └──────────────┘
     │
     ▼
┌────────────────────────────┐
│  Kubernetes Cluster        │
│  - 3 replicas (default)    │
│  - HPA: scales 2–10 pods   │
│  - LoadBalancer service     │
└────────────────────────────┘
     │
     ▼
┌────────────────────────────┐
│  Jenkins CI/CD             │
│  - Test → Build → Push     │
│  - Deploy → Smoke test     │
└────────────────────────────┘
```

---

## Dataset Description

| File | Rows | Columns | Description |
|---|---|---|---|
| `flights.csv` | 271,888 | 10 | Individual flight bookings |
| `hotels.csv` | 40,552 | 8 | Hotel stays linked to travel codes |
| `users.csv` | 1,340 | 5 | User profiles |

**Key columns:**
- `flights.csv`: `travelCode`, `userCode`, `from`, `to`, `flightType`, `price`, `time`, `distance`, `agency`, `date`
- `hotels.csv`: `travelCode`, `userCode`, `name`, `place`, `days`, `price`, `total`, `date`
- `users.csv`: `code`, `company`, `name`, `gender`, `age`

---

## ML Models

### A. Regression — Flight Price Prediction

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~110 | ~85 | ~0.91 |
| Random Forest ⭐ | ~45 | ~32 | ~0.98 |
| XGBoost | ~48 | ~35 | ~0.98 |

**Features used:** origin, destination, flight class, agency, distance, duration, month, day of week, weekend flag, speed, distance bucket

### B. Classification — Gender Prediction

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.72 | ~0.72 | ~0.72 |
| Random Forest ⭐ | ~0.79 | ~0.79 | ~0.79 | ~0.79 |

**Features used:** age, company, flight history aggregates, hotel booking aggregates

### C. Recommendation — Hotel Recommendation

- **Method**: Hybrid (Collaborative Filtering + Content-Based Filtering)
- **Collaborative**: User-item matrix with cosine similarity between users
- **Content-Based**: Hotel feature matrix (avg price, avg days, booking frequency)
- **Fallback**: Popularity-based recommendations for new/cold-start users

---

## API Documentation

### Base URL: `http://localhost:5000`

---

### `GET /health`
Check API status.

**Response:**
```json
{
  "status": "healthy",
  "regression_model": true,
  "classification_model": true,
  "recommendation_model": true
}
```

---

### `POST /predict-flight-price`
Predict the price of a flight.

**Request:**
```json
{
  "from": "Recife (PE)",
  "to": "Florianopolis (SC)",
  "flightType": "firstClass",
  "agency": "FlyingDrops",
  "time": 1.76,
  "distance": 676.53,
  "month": 9,
  "dayofweek": 3,
  "is_weekend": 0
}
```

**Response:**
```json
{
  "predicted_price": 1423.56,
  "currency": "USD"
}
```

---

### `POST /predict-gender`
Predict user gender from travel behavior.

**Request:**
```json
{
  "age": 30,
  "company": "4You",
  "total_flights": 10,
  "avg_flight_price": 950.0,
  "total_flight_spend": 9500.0,
  "avg_distance": 700.0,
  "first_class_count": 3,
  "economic_count": 5,
  "premium_count": 2,
  "first_class_ratio": 0.3,
  "total_hotel_bookings": 8,
  "avg_hotel_price": 250.0,
  "total_hotel_spend": 5000.0,
  "avg_days_stayed": 3.5
}
```

**Response:**
```json
{
  "predicted_gender": "male",
  "confidence": 0.83
}
```

---

### `POST /recommend-hotels`
Get hotel recommendations for a user.

**Request:**
```json
{
  "user_code": 42,
  "method": "hybrid",
  "top_n": 5
}
```

**Response:**
```json
{
  "user_code": 42,
  "recommendations": [
    {"hotel_name": "Hotel K", "avg_price": 263.41, "avg_days": 2.5, "score": 0.87, "method": "hybrid"},
    ...
  ]
}
```

---

## Installation & Setup

### Prerequisites
- Anaconda (with Python 3.10)
- VS Code (launched from Anaconda Navigator or terminal)
- Docker & Docker Compose
- (Optional) kubectl + Minikube for Kubernetes
- (Optional) Airflow for workflow automation

### Step 1 — Clone and Setup

```bash
git clone https://github.com/yourname/voyage-analytics-mlops.git
cd voyage_analytics_mlops

# Create Anaconda virtual environment
conda create -n voyage_mlops python=3.10 -y
conda activate voyage_mlops

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Train Models

```bash
python src/model_training.py
```

This will:
- Process all three datasets
- Train all 5 models (Linear Regression, RF Regressor, XGBoost, Logistic Regression, RF Classifier)
- Build the recommendation system
- Log all experiments to MLflow
- Save all models to `/models` directory

---

## Running the Project

### Flask API

```bash
python api/app.py
# API available at http://localhost:5000
```

Test it:
```bash
curl -X POST http://localhost:5000/predict-flight-price \
  -H "Content-Type: application/json" \
  -d '{"from":"Recife (PE)","to":"Florianopolis (SC)","flightType":"firstClass","agency":"FlyingDrops","time":1.76,"distance":676.53,"month":9,"dayofweek":3,"is_weekend":0}'
```

### Streamlit App

```bash
streamlit run streamlit_app/app.py
# App available at http://localhost:8501
```

### Run Unit Tests

```bash
pytest tests/ -v --cov=src
```

### View MLflow UI

```bash
mlflow ui --backend-store-uri file://./mlruns
# Dashboard at http://localhost:5000 (or use port 5001)
```

---

## Docker Deployment

### Build Image

```bash
docker build -t travel-ml-project:latest .
```

### Run Flask API

```bash
docker run -p 5000:5000 \
  -e APP_MODE=api \
  -v $(pwd)/models:/app/models \
  travel-ml-project:latest
```

### Run Streamlit App

```bash
docker run -p 8501:8501 \
  -e APP_MODE=streamlit \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  travel-ml-project:latest
```

### Full Stack with Docker Compose

```bash
cd docker
docker-compose up --build
```

Services started:
- Flask API → `http://localhost:5000`
- Streamlit → `http://localhost:8501`
- MLflow UI → `http://localhost:5001`

---

## Kubernetes Deployment

```bash
# Start Minikube (local)
minikube start --cpus=2 --memory=4096

# Load local image
minikube image load travel-ml-project:latest

# Deploy
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Get external URL (Minikube)
minikube service travel-ml-api-service --url

# Scale manually
kubectl scale deployment travel-ml-api --replicas=5

# View HPA status
kubectl get hpa
```

**The HPA (Horizontal Pod Autoscaler) automatically scales pods between 2 and 10 based on CPU and memory usage.**

---

## Apache Airflow

```bash
pip install apache-airflow

# Initialize DB
airflow db init

# Start services
airflow webserver --port 8080 &
airflow scheduler &

# Copy DAG file
cp airflow/dags/travel_ml_pipeline.py ~/airflow/dags/

# UI at http://localhost:8080 (admin/admin)
```

**DAGs available:**
- `travel_data_pipeline` — Runs daily: load → clean → validate → EDA report
- `travel_model_training_dag` — Runs weekly: retrain all models → evaluate → deploy

---

## CI/CD with Jenkins

1. Install Jenkins and required plugins: Pipeline, Docker, Kubernetes, Git
2. Create a new Pipeline job pointing to this repository
3. Set `Jenkinsfile` as the pipeline script path
4. Add credentials: `dockerhub-credentials`, `kubeconfig`
5. Push to `main` branch to trigger full deploy pipeline

**Pipeline stages:**
```
Checkout → Setup Python → Code Quality → Unit Tests →
Model Training (main only) → Docker Build → Docker Push →
Kubernetes Deploy → Smoke Test
```

---

## MLflow Experiment Tracking

View all experiments:
```bash
mlflow ui --backend-store-uri file://./mlruns --port 5001
```

**Experiments tracked:**
- `flight_price_regression` — 3 model runs with RMSE, MAE, R² metrics
- `gender_classification` — 2 model runs with Accuracy, Precision, Recall, F1
- `hotel_recommendation` — 1 run with matrix density metric

---

## Streamlit Web App

The Streamlit app provides 4 interactive pages:

| Page | Description |
|---|---|
| Overview & EDA | Dataset summary, price distributions, user demographics |
| Flight Price Predictor | Interactive form → real-time API prediction + gauge chart |
| Hotel Recommender | User-based recommendations with price comparison |
| Analytics Dashboard | Filtered trends, heatmaps, route analysis |

---

## Project Structure

```
voyage_analytics_mlops/
│
├── data/
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
│
├── src/
│   ├── data_processing.py   ← Load, clean, engineer, encode
│   ├── model_training.py    ← Train all 3 model types + MLflow
│   └── utils.py             ← Metrics, model save/load, helpers
│
├── api/
│   └── app.py               ← Flask REST API (3 endpoints)
│
├── streamlit_app/
│   └── app.py               ← Interactive Streamlit web UI
│
├── airflow/
│   └── dags/
│       └── travel_ml_pipeline.py  ← 2 production DAGs
│
├── jenkins/
│   └── Jenkinsfile          ← Full CI/CD pipeline script
│
├── kubernetes/
│   └── deployment.yaml      ← Deployment + Service + HPA
│
├── docker/
│   ├── entrypoint.sh        ← Container startup script
│   └── docker-compose.yml   ← Full stack local dev
│
├── tests/
│   └── test_pipeline.py     ← Unit tests (pytest)
│
├── models/                  ← Saved models (generated after training)
├── mlruns/                  ← MLflow tracking data (generated)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT License — free to use, modify, and distribute.
