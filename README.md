# Traffy Bangkok Ticket Late Prediction Pipeline

> End-to-end ML pipeline for predicting whether Bangkok Traffy Fondue tickets will be resolved late (>7 days) using Apache Airflow, FastAPI, and Streamlit.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.7.0-green.svg)](https://airflow.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)

---

## Table of Contents

-   [Overview](#overview)
-   [Architecture](#architecture)
-   [Features](#features)
-   [Project Structure](#project-structure)
-   [Setup & Installation](#setup--installation)
-   [Usage](#usage)
-   [API Documentation](#api-documentation)
-   [Pipeline Details](#pipeline-details)
-   [Contributing](#contributing)
-   [License](#license)

---

## Overview

This project implements a complete machine learning pipeline for predicting the resolution time of Bangkok's Traffy Fondue citizen complaint tickets. The system:

-   **Collects** external data (hospitals, weather) to enrich predictions
-   **Engineers** features combining temporal, spatial, and contextual information
-   **Trains** a Random Forest classifier to predict late vs on-time resolution
-   **Serves** predictions via REST API and interactive web UI
-   **Orchestrates** the entire workflow with Apache Airflow

**Business Value:** Help Bangkok city officials prioritize high-risk tickets and allocate resources more effectively.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                 │
├─────────────────────────────────────────────────────────────────┤
│  • Bangkok Traffy Dataset (787k tickets)                        │
│  • Hospital Network (web scraping)                              │
│  • Hourly Weather Data (API)                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               AIRFLOW ORCHESTRATION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │ Data Prep    │ → │  External    │ → │   Feature    │ →     │
│  │              │   │  Scraping    │   │ Engineering  │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│                                                ↓                │
│                      ┌──────────────────────────┐              │
│                      │   ML Training            │              │
│                      │  (Random Forest)         │              │
│                      └──────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  SERVING LAYER                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐         ┌────────────────────┐         │
│  │   FastAPI          │         │   Streamlit UI     │         │
│  │   REST API         │ ←────── │   Interactive      │         │
│  │   (Port 8000)      │         │   Dashboard        │         │
│  └────────────────────┘         │   (Port 8501)      │         │
│           ↓                     └────────────────────┘         │
│  ┌────────────────────┐                                        │
│  │  Trained Model     │                                        │
│  │  (.joblib)         │                                        │
│  └────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### ML Pipeline

-   Stratified sampling (300k from 787k records)
-   External data enrichment (hospitals + weather)
-   18 engineered features (temporal, spatial, contextual)
-   Random Forest classifier with hyperparameter tuning
-   Performance metrics: Accuracy, F1-score, Confusion Matrix

### Automation

-   Apache Airflow DAG for end-to-end orchestration
-   Automatic data refresh and model retraining
-   Error handling and retry logic
-   Scheduled execution (configurable)

### API Service (FastAPI)

-   `/predict` - Single ticket prediction
-   `/predict/batch` - Batch predictions
-   `/health` - Service health check
-   `/model/info` - Model metadata
-   Interactive API docs at `/docs`

### Web UI (Streamlit)

-   Interactive map picker for coordinates
-   Real-time predictions with probability scores
-   Batch prediction viewer
-   Feature input forms with validation
-   Responsive design

### Containerization

-   Docker Compose for one-command deployment
-   Separate containers for Airflow, API, and UI
-   PostgreSQL for Airflow metadata
-   Volume mounting for data persistence

---

## Project Structure

```
traffy-prediction-pipeline/
├── airflow/                    # Airflow home directory
│   ├── dags/
│   │   ├── traffy_model_pipeline_dag.py   # Main DAG
│   │   └── pipeline_tasks/
│   │       ├── prep.py                     # Data preparation
│   │       ├── external_scraping.py        # Scrape hospitals & weather
│   │       ├── feature_engineering.py      # Feature creation
│   │       └── ml_training.py              # Model training
│   ├── logs/                   # Airflow logs
│   └── airflow.cfg             # Airflow configuration
│
├── api/                        # FastAPI prediction service
│   ├── main.py                 # API endpoints
│   └── __init__.py
│
├── data/                       # Data directory (gitignored)
│   ├── bangkok_traffy.csv      # Raw data (download separately)
│   ├── bangkok_traffy_clean.csv
│   ├── external/               # External data sources
│   ├── features/               # Engineered features
│   └── predictions/            # Model outputs
│
├── models/                     # Trained models (gitignored)
│   └── traffy_rf_model.joblib
│
├── notebooks/                  # Jupyter notebooks (EDA)
│   ├── 01_traffy_prep.ipynb
│   ├── 02_external_scraping.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_ml_training.ipynb
│
├── scripts/
│   └── download_data.py        # Data download script
│
├── app.py                      # Streamlit UI
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile.api              # FastAPI container
├── Dockerfile.streamlit        # Streamlit container
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

-   **Docker & Docker Compose** (recommended) OR
-   **Python 3.10+** and **pip**

---


#### 1. Clone the repository

```bash
git clone https://github.com/frostnzx/traffy-prediction-pipeline.git
cd traffy-prediction-pipeline
```

#### 2. Download the dataset

```bash
# Install gdown if not installed
pip install gdown

# Run download script (update FILE_ID first)
python scripts/download_data.py
```

>  **Important:** Update `GOOGLE_DRIVE_FILE_ID` in `scripts/download_data.py` with your Google Drive file ID, or manually place `bangkok_traffy.csv` in the `data/` folder.

#### 3. Start all services 

```bash
docker-compose up -d
```

**What happens automatically:**

1. PostgreSQL starts (Airflow metadata DB)
2. Airflow webserver & scheduler start
3. **Training pipeline auto-triggers** (runs once on first startup)
4. Data preparation → External scraping → Feature engineering → Model training
5. API & Streamlit wait for model to be ready
6. Once model exists (~10-15 minutes), services start automatically

#### 4. Monitor training progress (optional)

```bash
# Watch Airflow scheduler logs
docker-compose logs -f airflow-scheduler

# Or check Airflow UI
open http://localhost:8080  # Login: admin/admin
```

#### 5. Access the services (after training completes)

-   **Airflow UI:** http://localhost:8080 (username: `admin`, password: `admin`)
-   **FastAPI Docs:** http://localhost:8000/docs
-   **Streamlit UI:** http://localhost:8501

**Note:** The DAG runs automatically once on first startup. For retraining, manually trigger it in the Airflow UI





---

## Usage

### Using the REST API

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ปัญหาถนน",
    "organization": "กทม.",
    "district": "บางรัก",
    "lat": 13.75,
    "lon": 100.50,
    "star": 3,
    "count_reopen": 0,
    "hour": 14,
    "dayofweek": 2,
    "month": 6,
    "year": 2023,
    "num_hospitals_in_district": 10,
    "rain_mm": 0.5,
    "is_rainy_hour": 1,
    "rain_last_3h": 1.5,
    "temperature": 32.0,
    "high_temperature": 0,
    "wind_speed": 3.5
  }'
```

**Response:**

```json
{
    "prediction": "ON-TIME",
    "probability_late": 0.32,
    "is_late": 0
}
```


---

### Using the Streamlit UI

1. Open http://localhost:8501
2. Use the sidebar to input ticket features
3. Click on the map to select coordinates (optional)
4. Click **Predict**
5. View prediction result with probability score

---

## API Documentation

### Endpoints

| Method | Endpoint         | Description       |
| ------ | ---------------- | ----------------- |
| GET    | `/`              | API information   |
| GET    | `/health`        | Health check      |
| GET    | `/model/info`    | Model metadata    |
| POST   | `/predict`       | Single prediction |
| POST   | `/predict/batch` | Batch predictions |

### Interactive Docs

Visit http://localhost:8000/docs for interactive API documentation with request/response examples.

---

## Pipeline Details

### DAG: `traffy_model_pipeline`

**Tasks:**

1. **prep_data** - Data cleaning, sampling, feature extraction
2. **external_scraping** - Fetch hospital & weather data
3. **feature_engineering** - Merge and create features
4. **ml_training** - Train Random Forest model

**Schedule:** Manual (on-demand) by default  
**Execution Time:** ~5-10 minutes (depends on data size)

---

## Configuration

### Airflow Configuration

Edit `airflow/airflow.cfg` for:

-   Executor type (LocalExecutor/CeleryExecutor)
-   Database connection
-   Parallelism settings

### Model Hyperparameters

Edit `airflow/dags/pipeline_tasks/ml_training.py`:

```python
model = RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=None,        # Unlimited depth
    random_state=42
)
```

### Data Sample Size

Edit `airflow/dags/pipeline_tasks/prep.py`:

```python
TARGET_ROWS = 300_000  # Adjust sample size
```
