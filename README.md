# Huawei Internship — MLOps & Cloud Infrastructure

> **End-to-end exploration of MLOps practices and cloud-native autoscaling infrastructure.**
> This repository documents hands-on work carried out during an internship at Huawei, covering experiment tracking, data drift monitoring, and OpenStack self-scaling systems.

---

## Repository Structure

```
.
├── mlflow-experiments/       # MLflow experiment tracking & MLOps pipeline
│   ├── train.py              # Model training script with MLflow logging
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # Detailed MLOps pipeline guide (10 stages)
│   ├── visualization/        # Interactive pipeline visualization (HTML/JS/CSS)
│   └── notes/                # Weekly study notes and build logs
│
├── data-drift/               # Data drift detection & monitoring
│   ├── data_drift_analysis.ipynb   # Hands-on notebook: drift detection on credit data
│   ├── data_drift_theory.tex       # LaTeX course: covariate, label & concept drift
│   └── data_drift_theory.pdf       # Compiled theory document
│
├── internship-report/        # Cumulative internship report (LaTeX)
│   ├── report.tex            # Main report source (incrementally updated)
│   ├── stage_hw.pdf          # Compiled report
│   └── *.png                 # Figures and screenshots
│
├── .github/workflows/        # CI/CD pipeline (GitHub Actions)
│   └── ml_pipeline.yml       # Lint + test on every push/PR
│
├── .dvc/                     # DVC configuration for data versioning
├── .dvcignore
└── .gitignore
```

---

## Modules Overview

### 1. MLflow Experiments (`mlflow-experiments/`)

Hands-on exploration of the full **MLOps lifecycle** using scikit-learn, MLflow, FastAPI, DVC, Airflow, and Prometheus/Grafana. Covers 10 stages from data collection to production monitoring.

**Key topics:**
- Experiment tracking with MLflow (parameter logging, metric comparison, model registry)
- Data versioning with DVC
- Model serving via FastAPI + Docker
- CI/CD with GitHub Actions
- Orchestration with Apache Airflow
- Monitoring with Prometheus & Grafana

→ See [`mlflow-experiments/README.md`](mlflow-experiments/README.md) for the full walkthrough.

---

### 2. Data Drift Detection (`data-drift/`)

A theoretical and practical study of **data drift in production ML systems** — the silent killer of model performance.

**What's included:**
- **Theory document** (`data_drift_theory.pdf`) — Formal definitions of covariate drift, label drift, and concept drift with mathematical formulations
- **Analysis notebook** (`data_drift_analysis.ipynb`) — Progressive, constructive Jupyter notebook demonstrating drift on a synthetic credit default prediction dataset

**Drift types covered:**

| Type | What changes | What stays the same |
|------|-------------|---------------------|
| Covariate | P(X) | P(Y\|X) |
| Label | P(Y) | P(X\|Y) |
| Concept | P(Y\|X) | P(X) |

---

### 3. Internship Report (`internship-report/`)

A **cumulative LaTeX report** documenting all work, decisions, bugs encountered, and solutions. This is incrementally updated and serves as the primary deliverable.

**Highlights documented:**
- OpenStack self-scaling infrastructure (DevStack + Heat + Ceilometer + Aodh)
- Nested virtualization chain (macOS → VirtualBox → Ubuntu → KVM → OpenStack VMs)
- DVC data versioning setup
- MLflow experiment tracking workflow
- Data drift theory and practical analysis

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML / Data Science** | scikit-learn, pandas, numpy, matplotlib, seaborn |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **API Serving** | FastAPI, Uvicorn, Docker |
| **CI/CD** | GitHub Actions, ruff, pytest |
| **Orchestration** | Apache Airflow |
| **Monitoring** | Prometheus, Grafana |
| **Cloud / Infra** | OpenStack (Nova, Heat, Ceilometer, Gnocchi, Aodh) |
| **Documentation** | LaTeX, Jupyter Notebooks |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/carteeeltheboss/HuaweiIntern.git
cd HuaweiIntern

# Set up a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies for MLflow experiments
pip install -r mlflow-experiments/requirements.txt

# Run the training script
python mlflow-experiments/train.py

# Launch MLflow UI to view results
mlflow ui --port 5000
```

---

## License

This repository is part of an internship project at **Huawei**. All rights reserved.
