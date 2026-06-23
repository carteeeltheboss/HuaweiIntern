# MLOps Pipeline — Hands-On Practical Lab

> **From a notebook experiment to a production-grade ML system in 10 steps.**
> This repository is a complete, hands-on TP (Travaux Pratiques) that walks you through
> the entire MLOps lifecycle using a single, concrete example. No decisions to make — every
> variable is fixed so you can focus on understanding the *how* and *why*.

---

## The Example

| Parameter | Value |
|-----------|-------|
| **Dataset** | Wine Quality (Red) — UCI Machine Learning Repository |
| **Samples** | 1,599 red wine samples, 11 physicochemical features |
| **Task** | Binary classification — "Is this wine good?" (quality >= 7 is Good, else Not Good) |
| **Model** | RandomForestClassifier (scikit-learn) — n_estimators=100, max_depth=10 |
| **API** | FastAPI served with Uvicorn on port 8000 |
| **Tracking** | MLflow experiment server on port 5000 |
| **Data versioning** | DVC with a local remote |
| **CI/CD** | GitHub Actions (lint + test on every push) |
| **Orchestration** | Apache Airflow DAG (daily retraining at 08:00) |
| **Monitoring** | Prometheus (port 9090) + Grafana (port 3000) |

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone <your-repo-url> && cd HuaweiIntern

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the pipeline step by step
python steps/step_01_data/download_data.py
python steps/step_02_explore/explore_data.py
python steps/step_03_features/feature_engineering.py
python steps/step_04_train/train_model.py
python steps/step_05_evaluate/evaluate_model.py

# 5. Launch the API
cd steps/step_06_serve && uvicorn app:app --host 0.0.0.0 --port 8000

# 6. Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}'
```

---

## Visual Pipeline

Open `pipeline/index.html` in your browser to see an **interactive visualization** of the entire
MLOps pipeline. Click on any stage to see:

- What happens at that step and why it matters
- The key concepts introduced
- The tools used and their roles
- The exact commands and code to run
- What output to expect
- A verification checklist before moving on

The visualization covers all 10 stages and includes the retraining feedback loop from monitoring
back to data collection.

---

## The 10 Stages

Each stage corresponds to a directory under `steps/`. Every script is self-contained, heavily
commented, and runnable on its own. Read the docstring at the top of each file before running it.

### Stage 1 — Data Collection

**Script:** `steps/step_01_data/download_data.py`

Downloads the Wine Quality dataset from UCI and saves it to `data/raw/winequality-red.csv`.
Includes a synthetic fallback if the network is unavailable.

```bash
python steps/step_01_data/download_data.py
```

**What you learn:** Automated data ingestion, directory structure conventions, fallback mechanisms.

---

### Stage 2 — Data Exploration and Validation

**Script:** `steps/step_02_explore/explore_data.py`

Explores the dataset: shape, types, descriptive statistics, missing values, target distribution.
Generates a correlation heatmap and a target distribution chart.

```bash
python steps/step_02_explore/explore_data.py
```

**What you learn:** Why you must understand your data before training, how to spot class imbalance,
how to detect feature correlations that could affect model performance.

**Outputs:**
- `outputs/exploration/correlation_matrix.png`
- `outputs/exploration/target_distribution.png`

---

### Stage 3 — Feature Engineering

**Script:** `steps/step_03_features/feature_engineering.py`

Creates the binary target variable (`is_good`), performs a stratified 80/20 train/test split,
and applies `StandardScaler` (fit on training data only, applied to both sets).

```bash
python steps/step_03_features/feature_engineering.py
```

**What you learn:** Why you never fit the scaler on test data (data leakage), why stratified
splitting matters for imbalanced classes, how to persist preprocessing artifacts.

**Outputs:**
- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/scaler.joblib`

---

### Stage 4 — Model Training with Experiment Tracking

**Script:** `steps/step_04_train/train_model.py`

Trains three RandomForestClassifier configurations (n_estimators = 50, 100, 200) and logs
parameters, metrics, and model artifacts to MLflow. Selects the best model by F1 score.

```bash
# First, start the MLflow tracking server in a separate terminal
mlflow ui --port 5000

# Then run training
python steps/step_04_train/train_model.py
```

**What you learn:** Why you must track experiments systematically, how MLflow stores runs, how to
compare hyperparameter configurations, why F1 is a better metric than accuracy for imbalanced data.

**Outputs:**
- `models/wine_quality_rf.joblib` (best model)
- MLflow runs visible at `http://localhost:5000`

---

### Stage 5 — Model Evaluation

**Script:** `steps/step_05_evaluate/evaluate_model.py`

Evaluates the best model on the held-out test set. Generates a confusion matrix, a classification
report, and a feature importance chart. Includes a quality gate check.

```bash
python steps/step_05_evaluate/evaluate_model.py
```

**What you learn:** How to evaluate a model beyond accuracy, what a confusion matrix tells you
about false positives vs false negatives, how feature importance informs data collection priorities.

**Outputs:**
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/feature_importances.png`
- `outputs/evaluation/classification_report.json`

---

### Stage 6 — Model Serving

**Script:** `steps/step_06_serve/app.py`

A FastAPI application that serves predictions over HTTP. Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Predict a single wine sample |
| `POST` | `/predict/batch` | Predict multiple samples at once |
| `GET` | `/health` | Liveness check (is the model loaded?) |
| `GET` | `/info` | Model metadata (type, features, classes) |
| `GET` | `/metrics` | Prometheus metrics (auto-instrumented) |
| `GET` | `/docs` | Interactive Swagger documentation |

```bash
# Run locally
cd steps/step_06_serve
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Or build and run with Docker
docker build -t wine-api .
docker run -p 8000:8000 wine-api
```

**What you learn:** How to wrap a model in a REST API, Pydantic input validation, fail-fast model
loading, Prometheus instrumentation for production observability.

---

### Stage 7 — Testing

**Script:** `steps/step_07_test/test_api.py`

A pytest suite that validates the API: happy path, missing fields, wrong types, batch requests,
health and info endpoints. Uses `pytest.parametrize` for comprehensive coverage.

```bash
# Run from project root
pytest steps/step_07_test/ -v
```

**What you learn:** Why tests are non-negotiable before deployment, how to test FastAPI with
`TestClient`, how parametrized tests keep your suite DRY.

---

### Stage 8 — CI/CD with GitHub Actions

**Config:** `.github/workflows/ml_pipeline.yml`

Runs on every push to `main` and on pull requests. Two jobs:

1. **Lint** — checks code style with `ruff`
2. **Test** — installs dependencies, trains the model, runs the test suite

```bash
# Test locally before pushing
ruff check .
pytest steps/step_07_test/ -v
```

**What you learn:** How to automate quality gates, why linting catches bugs before they ship,
how CI pipelines enforce team standards.

---

### Stage 9 — Orchestration with Airflow

**Script:** `steps/step_09_orchestrate/wine_pipeline_dag.py`

An Airflow DAG that chains the pipeline steps:

```
download_data >> engineer_features >> train_model >> evaluate_model
```

Scheduled to run daily at 08:00 with 2 retries per task.

```bash
# Copy the DAG to your Airflow dags folder
cp steps/step_09_orchestrate/wine_pipeline_dag.py ~/airflow/dags/

# Start Airflow (if not already running)
airflow standalone

# Trigger manually from the UI at http://localhost:8080
```

**What you learn:** Why manual script execution does not scale, how DAG dependencies prevent
data inconsistencies, how scheduled retraining keeps a model current.

---

### Stage 10 — Monitoring with Prometheus and Grafana

**Config:** `steps/step_10_monitor/docker-compose.yml`

Launches the full observability stack: the Wine API, Prometheus, and Grafana.

```bash
cd steps/step_10_monitor
docker compose up -d
```

| Service | URL | Purpose |
|---------|-----|---------|
| Wine API | `http://localhost:8000` | The model serving endpoint |
| Prometheus | `http://localhost:9090` | Metric collection and querying |
| Grafana | `http://localhost:3000` | Dashboards (default login: admin/admin) |

The pre-configured Grafana dashboard includes four panels:

1. **Request Rate** — requests per second by endpoint and status
2. **Response Time** — p50 and p95 latency
3. **Error Rate** — percentage of 4xx/5xx responses
4. **Prediction Distribution** — ratio of Good vs Not Good predictions

**What you learn:** Why logging alone is not enough for production, how Prometheus scrapes metrics,
how to build operational dashboards, the RED method (Rate, Errors, Duration).

---

## Project Structure

```
HuaweiIntern/
|-- README.md                              # This file
|-- requirements.txt                       # All Python dependencies
|
|-- pipeline/                              # Interactive visual pipeline
|   |-- index.html                         # Open in browser
|   |-- style.css
|   +-- script.js
|
|-- steps/
|   |-- step_01_data/
|   |   +-- download_data.py              # Data collection
|   |-- step_02_explore/
|   |   +-- explore_data.py               # EDA and validation
|   |-- step_03_features/
|   |   +-- feature_engineering.py         # Feature processing
|   |-- step_04_train/
|   |   +-- train_model.py                # Training + MLflow logging
|   |-- step_05_evaluate/
|   |   +-- evaluate_model.py             # Evaluation + quality gate
|   |-- step_06_serve/
|   |   |-- app.py                        # FastAPI application
|   |   |-- Dockerfile                    # Container definition
|   |   +-- requirements_serve.txt        # Serving-only dependencies
|   |-- step_07_test/
|   |   |-- test_api.py                   # pytest suite
|   |   +-- conftest.py                   # Test fixtures
|   |-- step_09_orchestrate/
|   |   +-- wine_pipeline_dag.py          # Airflow DAG
|   +-- step_10_monitor/
|       |-- docker-compose.yml            # API + Prometheus + Grafana
|       |-- prometheus.yml                # Scrape configuration
|       +-- grafana/
|           |-- dashboards/
|           |   +-- wine_api_dashboard.json
|           +-- provisioning/
|               |-- dashboards.yml
|               +-- datasources.yml
|
|-- .github/workflows/
|   +-- ml_pipeline.yml                   # CI/CD pipeline
|
|-- data/                                 # Created by the scripts
|   |-- raw/                              # Raw downloaded data
|   +-- processed/                        # Processed train/test splits
|
|-- models/                               # Created by the scripts
|   |-- wine_quality_rf.joblib            # Trained model
|   +-- scaler.joblib                     # Fitted scaler
|
+-- outputs/                              # Created by the scripts
    |-- exploration/                      # EDA plots
    +-- evaluation/                       # Evaluation artifacts
```

---

## Prerequisites

| Tool | Required For | Installation |
|------|-------------|--------------|
| Python 3.10+ | Everything | [python.org](https://www.python.org/downloads/) |
| pip | Package management | Included with Python |
| Docker | Steps 6, 10 | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | Step 10 | Included with Docker Desktop |
| MLflow | Step 4 | `pip install mlflow` (included in requirements.txt) |
| Airflow | Step 9 | `pip install apache-airflow` |
| Git | Version control, CI/CD | [git-scm.com](https://git-scm.com/) |

---

## How to Use This Repository

1. **Open the visual pipeline** — start with `pipeline/index.html` in your browser to see
   the big picture before writing any code.

2. **Run each step in order** — the scripts are numbered for a reason. Each one builds on
   the output of the previous step. Read the docstring at the top of every file.

3. **Read the "WHY THIS MATTERS" comments** — every script explains not just *what* the code
   does but *why* it matters in a production ML system.

4. **Try the "TRY THIS" suggestions** — each script includes modification ideas to deepen
   your understanding.

5. **Check the verification items** — each stage in the visual pipeline has a checklist.
   If you cannot verify it, the step is not complete.

6. **Commit after each stage** — practice the version control discipline that MLOps demands.

---

## The MLOps Lifecycle

This is the cycle your pipeline implements. After monitoring detects degradation (drift, accuracy
drop, data distribution shift), the loop restarts from data collection:

```
Data Collection
      |
      v
Data Validation
      |
      v
Feature Engineering
      |
      v
Model Training  <--+
      |             |
      v             |
Model Evaluation    |
      |             |
      v             |
Model Registry      |
      |             |
      v             |
Model Serving       |
      |             |
      v             |
Testing / CI/CD     |
      |             |
      v             |
Orchestration       |
      |             |
      v             |
Monitoring ---------+
(drift detected -> retrain)
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/...` | Step 1 not run | Run `python steps/step_01_data/download_data.py` first |
| `FileNotFoundError: models/...` | Steps 3-4 not run | Run steps 1 through 4 in order |
| `422 Unprocessable Entity` | Bad request body | Check that all 11 features are present as floats |
| `500 Internal Server Error` | Model not loaded | Check server logs — model or scaler file may be missing |
| Container exits immediately | Port conflict or missing file | Run `docker logs <container>` to see the error |
| MLflow UI is empty | Wrong tracking URI | Ensure `mlflow ui` is running in the project root |
| Prometheus target DOWN | Wrong hostname in compose network | Use the Docker service name, not localhost |
| Grafana shows "No data" | No traffic yet or wrong time range | Send a few requests, then check the time range picker |
| GitHub Action fails | Missing requirements or test failure | Run `pytest` locally first to reproduce |

---

## Further Reading

| Topic | Resource |
|-------|----------|
| FastAPI | https://fastapi.tiangolo.com/ |
| Docker | https://docs.docker.com/get-started/ |
| MLflow | https://mlflow.org/docs/latest/index.html |
| DVC | https://dvc.org/doc/start |
| pytest + FastAPI | https://fastapi.tiangolo.com/tutorial/testing/ |
| GitHub Actions | https://docs.github.com/en/actions |
| Apache Airflow | https://airflow.apache.org/docs/ |
| Prometheus | https://prometheus.io/docs/introduction/overview/ |
| Grafana | https://grafana.com/docs/grafana/latest/ |
| scikit-learn | https://scikit-learn.org/stable/user_guide.html |
