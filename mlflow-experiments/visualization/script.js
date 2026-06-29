/* ==========================================================
   MLOps Pipeline — Wine Quality Classification
   Interactive stage navigation and content rendering
   ========================================================== */

(function () {
    "use strict";

    /* ----------------------------------------------------------
       STAGE DATA
       ---------------------------------------------------------- */
    const STAGES = [
        {
            id: 1,
            title: "Data Collection",
            shortLabel: "Data Collection",
            tagline: "Download and store the UCI Wine Quality dataset for red wine.",
            what: "The pipeline begins by acquiring the raw dataset from the UCI Machine Learning Repository. The Wine Quality (Red) dataset contains 1,599 samples with 11 physicochemical input features and one output variable (quality score 3-8). The file is semicolon-separated and must be stored locally under version control with DVC.",
            concepts: [
                "Data provenance and lineage tracking",
                "Reproducible data acquisition via scripted downloads",
                "Data versioning with DVC to track large file changes",
                "Raw data immutability principle",
                "Separation of data storage from code repositories"
            ],
            tools: [
                { name: "curl / wget", desc: "HTTP download" },
                { name: "DVC", desc: "Data version control" },
                { name: "Git", desc: "Code versioning" },
                { name: "Python", desc: "Scripted download" }
            ],
            code: [
                {
                    label: "Terminal: Download and version the dataset",
                    lang: "bash",
                    code: `# Create project structure
mkdir -p data/raw
cd data/raw

# Download the Wine Quality (Red) dataset
curl -o winequality-red.csv \\
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# Verify the download
wc -l winequality-red.csv
head -n 3 winequality-red.csv

# Initialize DVC and track the dataset
cd ../..
dvc init
dvc add data/raw/winequality-red.csv
git add data/raw/winequality-red.csv.dvc data/raw/.gitignore .dvc/
git commit -m "Add raw wine quality dataset with DVC tracking"`
                }
            ],
            output: `  1600 winequality-red.csv
"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";...
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5`,
            checklist: [
                "File data/raw/winequality-red.csv exists and has 1,600 lines (header + 1,599 rows)",
                "File is semicolon-separated, not comma-separated",
                "DVC file winequality-red.csv.dvc is tracked in Git",
                "Dataset contains 12 columns (11 features + 1 target 'quality')",
                "No authentication required for the download URL"
            ]
        },
        {
            id: 2,
            title: "Data Validation",
            shortLabel: "Validation",
            tagline: "Verify schema integrity, check for missing values, and profile feature distributions.",
            what: "Before any transformation, the raw data must be validated against expected schema constraints. This stage checks column names, data types, value ranges, missing entries, and basic statistical distributions. Catching data quality issues here prevents silent failures downstream in training or serving.",
            concepts: [
                "Schema enforcement and contract testing",
                "Missing value detection and handling strategies",
                "Statistical profiling: mean, std, min, max, quantiles",
                "Distribution drift detection baselines",
                "Data quality gates as pipeline checkpoints"
            ],
            tools: [
                { name: "pandas", desc: "Data manipulation" },
                { name: "Great Expectations", desc: "Validation framework" },
                { name: "Python", desc: "Validation script" }
            ],
            code: [
                {
                    label: "Python: validate_data.py",
                    lang: "python",
                    code: `import pandas as pd
import sys

# Load the dataset
df = pd.read_csv("data/raw/winequality-red.csv", sep=";")

# Schema validation
EXPECTED_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol", "quality"
]

assert list(df.columns) == EXPECTED_COLUMNS, "Schema mismatch detected"
assert len(df) == 1599, f"Expected 1599 rows, got {len(df)}"

# Missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)
assert missing.sum() == 0, "Missing values found"

# Data types
print("\\nData types:")
print(df.dtypes)

# Statistical summary
print("\\nStatistical summary:")
print(df.describe().round(3).to_string())

# Value range checks
assert df["pH"].between(0, 14).all(), "pH values out of range"
assert df["alcohol"].between(0, 20).all(), "Alcohol values out of range"
assert df["quality"].between(0, 10).all(), "Quality scores out of range"

print("\\nAll validation checks passed.")`
                }
            ],
            output: `Missing values per column:
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64

Data types:
fixed acidity           float64
volatile acidity        float64
...
quality                   int64

All validation checks passed.`,
            checklist: [
                "All 12 expected columns are present with correct names",
                "Zero missing values across all columns",
                "All numeric columns have appropriate data types (float64 or int64)",
                "pH values are within 0-14 range",
                "Quality scores are integers between 3 and 8",
                "Statistical summary shows reasonable distributions"
            ]
        },
        {
            id: 3,
            title: "Feature Engineering",
            shortLabel: "Features",
            tagline: "Create the binary target variable, scale features, and split into train/test sets.",
            what: "The raw quality score (3-8) is converted into a binary classification target: wines with quality >= 7 are labeled 'Good' (1), all others 'Not Good' (0). The 11 input features are standardized using StandardScaler to zero mean and unit variance. The dataset is then split 80/20 into training and test sets with stratification to preserve class balance.",
            concepts: [
                "Binary target encoding from ordinal scores",
                "Feature standardization (zero mean, unit variance)",
                "Stratified train/test splitting for imbalanced classes",
                "Class imbalance awareness (approx. 13.5% positive class)",
                "Feature matrix (X) and target vector (y) separation",
                "Reproducibility via fixed random_state"
            ],
            tools: [
                { name: "pandas", desc: "Data manipulation" },
                { name: "scikit-learn", desc: "StandardScaler, train_test_split" },
                { name: "joblib", desc: "Serializing the scaler" }
            ],
            code: [
                {
                    label: "Python: feature_engineering.py",
                    lang: "python",
                    code: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load validated data
df = pd.read_csv("data/raw/winequality-red.csv", sep=";")

# Create binary target: quality >= 7 is Good (1), else Not Good (0)
df["target"] = (df["quality"] >= 7).astype(int)
print(f"Class distribution:\\n{df['target'].value_counts()}")
print(f"Positive class ratio: {df['target'].mean():.3f}")

# Define feature columns
FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol"
]

X = df[FEATURE_COLS]
y = df["target"]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=FEATURE_COLS, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURE_COLS, index=X_test.index
)

# Save artifacts
os.makedirs("data/processed", exist_ok=True)
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.joblib")
print("\\nFeature engineering complete. Artifacts saved.")`
                }
            ],
            output: `Class distribution:
0    1382
1     217
Name: target, dtype: int64
Positive class ratio: 0.136

Train set: 1279 samples
Test set:  320 samples

Feature engineering complete. Artifacts saved.`,
            checklist: [
                "Binary target has two classes: 0 (Not Good) and 1 (Good)",
                "Positive class ratio is approximately 13.6%",
                "Train set has 1,279 samples, test set has 320 samples",
                "Scaler is fit on training data only, then applied to test data",
                "Files X_train.csv, X_test.csv, y_train.csv, y_test.csv exist in data/processed/",
                "Scaler artifact saved at models/scaler.joblib"
            ]
        },
        {
            id: 4,
            title: "Model Training",
            shortLabel: "Training",
            tagline: "Train a RandomForestClassifier and log all parameters and metrics to MLflow.",
            what: "A RandomForestClassifier is trained on the scaled training data with n_estimators=100 and max_depth=10. All hyperparameters, training metrics, and the serialized model artifact are logged to MLflow for full experiment reproducibility. The MLflow tracking server runs locally on port 5000.",
            concepts: [
                "Ensemble methods: Random Forest as a collection of decision trees",
                "Hyperparameter specification and logging",
                "Experiment tracking for reproducibility",
                "Model serialization and artifact storage",
                "MLflow runs, experiments, and the tracking URI concept",
                "Training on scaled features for consistent results"
            ],
            tools: [
                { name: "scikit-learn", desc: "RandomForestClassifier" },
                { name: "MLflow", desc: "Experiment tracking" },
                { name: "joblib", desc: "Model serialization" },
                { name: "Python", desc: "Training script" }
            ],
            code: [
                {
                    label: "Terminal: Start MLflow tracking server",
                    lang: "bash",
                    code: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db`
                },
                {
                    label: "Python: train_model.py",
                    lang: "python",
                    code: `import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Set MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wine-quality-classification")

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Define model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Train with MLflow tracking
with mlflow.start_run(run_name="rf-baseline"):
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)
    test_f1   = f1_score(y_test, y_pred_test)

    # Log to MLflow
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_features": X_train.shape[1],
        "n_train_samples": X_train.shape[0]
    })
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1
    })
    mlflow.sklearn.log_model(model, "random-forest-model")

    # Also save locally
    joblib.dump(model, "models/model.joblib")

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test F1 score:  {test_f1:.4f}")
    print(f"MLflow run ID:  {mlflow.active_run().info.run_id}")`
                }
            ],
            output: `Train accuracy: 0.9890
Test accuracy:  0.9281
Test F1 score:  0.6154
MLflow run ID:  a1b2c3d4e5f6789...`,
            checklist: [
                "MLflow tracking server is running on http://localhost:5000",
                "Experiment 'wine-quality-classification' is visible in the MLflow UI",
                "All hyperparameters are logged (n_estimators, max_depth, random_state)",
                "Train and test metrics are logged (accuracy, F1)",
                "Model artifact is logged to MLflow and saved locally at models/model.joblib",
                "Test accuracy is above 0.90 for the baseline model"
            ]
        },
        {
            id: 5,
            title: "Model Evaluation",
            shortLabel: "Evaluation",
            tagline: "Generate confusion matrix, classification report, feature importances, and apply quality gates.",
            what: "The trained model is evaluated comprehensively on the held-out test set. A confusion matrix reveals prediction error patterns, the classification report shows per-class precision, recall, and F1, and feature importances identify which physicochemical properties drive predictions. A quality gate ensures the model meets minimum performance thresholds before promotion.",
            concepts: [
                "Confusion matrix: true/false positives and negatives",
                "Precision, recall, and F1-score per class",
                "Feature importance ranking from tree-based models",
                "Quality gating: minimum threshold enforcement",
                "Overfitting detection by comparing train vs test metrics",
                "Class-weighted evaluation for imbalanced datasets"
            ],
            tools: [
                { name: "scikit-learn", desc: "Classification metrics" },
                { name: "MLflow", desc: "Artifact logging" },
                { name: "matplotlib", desc: "Visualization (optional)" },
                { name: "Python", desc: "Evaluation script" }
            ],
            code: [
                {
                    label: "Python: evaluate_model.py",
                    lang: "python",
                    code: `import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import mlflow
import json

# Load data and model
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
model  = joblib.load("models/model.joblib")

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

# Classification report
print("\\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Good", "Good"]
))

# Feature importances
FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates",
    "alcohol"
]
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("Feature Importances (descending):")
for i in sorted_idx:
    print(f"  {FEATURE_COLS[i]:>22s}: {importances[i]:.4f}")

# Quality gate
test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred)
MIN_ACCURACY = 0.90
MIN_F1 = 0.40

gate_passed = test_acc >= MIN_ACCURACY and test_f1 >= MIN_F1
print(f"\\nQuality Gate: {'PASSED' if gate_passed else 'FAILED'}")
print(f"  Accuracy: {test_acc:.4f} (min: {MIN_ACCURACY})")
print(f"  F1 Score: {test_f1:.4f} (min: {MIN_F1})")

# Save evaluation report
report = {
    "accuracy": round(test_acc, 4),
    "f1_score": round(test_f1, 4),
    "confusion_matrix": cm.tolist(),
    "gate_passed": gate_passed
}
with open("models/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)`
                }
            ],
            output: `Confusion Matrix:
  TN=274  FP=3
  FN=20   TP=23

Classification Report:
              precision    recall  f1-score   support
   Not Good       0.93      0.99      0.96       277
       Good       0.88      0.53      0.67        43
    accuracy                           0.93       320
   macro avg       0.91      0.76      0.81       320
weighted avg       0.93      0.93      0.92       320

Feature Importances (descending):
               alcohol: 0.1432
      volatile acidity: 0.1198
             sulphates: 0.1087
  ...

Quality Gate: PASSED
  Accuracy: 0.9281 (min: 0.90)
  F1 Score: 0.6154 (min: 0.40)`,
            checklist: [
                "Confusion matrix shows reasonable TN/TP counts given class imbalance",
                "Classification report generated for both classes",
                "Feature importances are logged (alcohol and volatile acidity typically rank highest)",
                "Quality gate evaluates both accuracy and F1 score",
                "Evaluation report saved to models/evaluation_report.json",
                "No signs of severe overfitting (train vs test gap is reasonable)"
            ]
        },
        {
            id: 6,
            title: "Model Registry",
            shortLabel: "Registry",
            tagline: "Version the trained model in MLflow and promote it through staging to production.",
            what: "Once the model passes quality gates, it is registered in the MLflow Model Registry with a versioned name. Models progress through lifecycle stages: None, Staging, Production, and Archived. This provides a centralized catalog of all trained models, enabling rollback, A/B comparisons, and auditable promotion workflows.",
            concepts: [
                "Model versioning and unique naming conventions",
                "Lifecycle stages: Staging, Production, Archived",
                "Model promotion as a controlled deployment gate",
                "Rollback capability through version history",
                "Separation of model training from model deployment",
                "Model lineage: linking runs to registered versions"
            ],
            tools: [
                { name: "MLflow", desc: "Model Registry" },
                { name: "Python", desc: "Registration script" }
            ],
            code: [
                {
                    label: "Python: register_model.py",
                    lang: "python",
                    code: `import mlflow
from mlflow.tracking import MlflowClient
import json

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "wine-quality-classifier"

# Load evaluation report to confirm gate passed
with open("models/evaluation_report.json") as f:
    report = json.load(f)

if not report["gate_passed"]:
    print("Quality gate not passed. Skipping registration.")
    exit(1)

# Get the latest run from the experiment
experiment = client.get_experiment_by_name("wine-quality-classification")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)
latest_run = runs[0]
run_id = latest_run.info.run_id

# Register the model
model_uri = f"runs:/{run_id}/random-forest-model"
result = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered model version: {result.version}")

# Transition to Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Staging"
)
print(f"Model v{result.version} transitioned to Staging")

# Promote to Production (after manual review in real workflow)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production"
)
print(f"Model v{result.version} promoted to Production")`
                }
            ],
            output: `Registered model version: 1
Model v1 transitioned to Staging
Model v1 promoted to Production`,
            checklist: [
                "Model 'wine-quality-classifier' appears in MLflow Model Registry",
                "Version number is assigned automatically",
                "Model transitions from None to Staging to Production",
                "The registered model links back to the original training run",
                "Quality gate check prevents unqualified models from registration",
                "Previous production model (if any) can be archived for rollback"
            ]
        },
        {
            id: 7,
            title: "Model Serving",
            shortLabel: "Serving",
            tagline: "Deploy the model as a FastAPI REST API with health check and prediction endpoints.",
            what: "The production model is wrapped in a FastAPI application exposing three endpoints: /predict for inference, /health for liveness checks, and /info for model metadata. The API loads the serialized model and scaler at startup, accepts JSON payloads with the 11 wine features, and returns the predicted class with confidence scores. The entire service is containerized with Docker.",
            concepts: [
                "RESTful API design for ML inference",
                "Request/response schema validation with Pydantic",
                "Model and scaler loading at application startup",
                "Health check endpoints for orchestration readiness probes",
                "Containerization for environment reproducibility",
                "Stateless serving for horizontal scalability"
            ],
            tools: [
                { name: "FastAPI", desc: "Python web framework" },
                { name: "Uvicorn", desc: "ASGI server" },
                { name: "Pydantic", desc: "Data validation" },
                { name: "Docker", desc: "Containerization" },
                { name: "joblib", desc: "Model loading" }
            ],
            code: [
                {
                    label: "Python: app/serve.py",
                    lang: "python",
                    code: `from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Wine Quality Classifier API")

# Load model and scaler at startup
model  = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid",
    "residual_sugar", "chlorides", "free_sulfur_dioxide",
    "total_sulfur_dioxide", "density", "pH", "sulphates",
    "alcohol"
]

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/info")
def info():
    return {
        "model": "RandomForestClassifier",
        "features": FEATURE_NAMES,
        "classes": ["Not Good", "Good"],
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(features: WineFeatures):
    data = pd.DataFrame([features.dict()])
    # Column names must match scaler training names
    data.columns = [c.replace("_", " ") for c in data.columns]
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    return {
        "prediction": int(pred),
        "label": "Good" if pred == 1 else "Not Good",
        "confidence": {
            "not_good": round(float(proba[0]), 4),
            "good": round(float(proba[1]), 4)
        }
    }`
                },
                {
                    label: "Terminal: Run the API server",
                    lang: "bash",
                    code: `uvicorn app.serve:app --host 0.0.0.0 --port 8000 --reload`
                },
                {
                    label: "Terminal: Test the prediction endpoint",
                    lang: "bash",
                    code: `curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'`
                }
            ],
            output: `{
  "prediction": 0,
  "label": "Not Good",
  "confidence": {
    "not_good": 0.9700,
    "good": 0.0300
  }
}`,
            checklist: [
                "FastAPI server starts on port 8000 without errors",
                "GET /health returns {\"status\": \"healthy\"}",
                "GET /info returns model metadata and feature list",
                "POST /predict accepts all 11 features and returns prediction with confidence",
                "API auto-generates Swagger docs at /docs",
                "Invalid input returns a 422 validation error with descriptive message"
            ]
        },
        {
            id: 8,
            title: "Testing and CI/CD",
            shortLabel: "CI/CD",
            tagline: "Write pytest unit tests and automate them with GitHub Actions on every push.",
            what: "Automated tests validate that the model loads correctly, the API endpoints return expected responses, data validation logic catches malformed inputs, and the pipeline scripts execute without errors. A GitHub Actions workflow runs these tests on every push and pull request, ensuring no regression reaches the main branch.",
            concepts: [
                "Unit testing for ML pipeline components",
                "Integration testing for API endpoints",
                "Continuous Integration: automated test execution",
                "Continuous Deployment: automated release pipelines",
                "Test fixtures and mock data for reproducibility",
                "Branch protection with required status checks"
            ],
            tools: [
                { name: "pytest", desc: "Testing framework" },
                { name: "GitHub Actions", desc: "CI/CD platform" },
                { name: "httpx", desc: "Async HTTP test client" }
            ],
            code: [
                {
                    label: "Python: tests/test_api.py",
                    lang: "python",
                    code: `import pytest
from fastapi.testclient import TestClient
from app.serve import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "RandomForestClassifier"
    assert len(data["features"]) == 11

def test_predict_valid_input():
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["label"] in ["Good", "Not Good"]

def test_predict_missing_field():
    payload = {"fixed_acidity": 7.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422`
                },
                {
                    label: "YAML: .github/workflows/ci.yml",
                    lang: "yaml",
                    code: `name: ML Pipeline CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest httpx

      - name: Run data validation
        run: python validate_data.py

      - name: Run tests
        run: pytest tests/ -v --tb=short

      - name: Check model quality gate
        run: python evaluate_model.py`
                }
            ],
            output: `tests/test_api.py::test_health_endpoint    PASSED
tests/test_api.py::test_info_endpoint      PASSED
tests/test_api.py::test_predict_valid_input PASSED
tests/test_api.py::test_predict_missing_field PASSED

======= 4 passed in 1.23s =======`,
            checklist: [
                "All 4 tests pass locally with pytest",
                "GitHub Actions workflow file exists at .github/workflows/ci.yml",
                "CI runs on push to main and on pull requests",
                "Pipeline includes data validation, unit tests, and quality gate steps",
                "Failed tests block merging to main branch",
                "Test coverage covers health, info, predict, and validation error paths"
            ]
        },
        {
            id: 9,
            title: "Orchestration",
            shortLabel: "Orchestration",
            tagline: "Define an Airflow DAG to chain download, feature engineering, training, and evaluation.",
            what: "Apache Airflow orchestrates the entire pipeline as a directed acyclic graph (DAG). Each stage becomes a task with defined dependencies, retry policies, and scheduling intervals. The DAG can be triggered manually or on a cron schedule (e.g., weekly retraining), ensuring that data flows through all stages in the correct order with proper error handling.",
            concepts: [
                "DAGs: directed acyclic graphs for workflow definition",
                "Task dependencies and execution order",
                "Scheduling with cron expressions",
                "Retry policies and failure callbacks",
                "Idempotent tasks for safe re-execution",
                "Airflow operators: PythonOperator, BashOperator"
            ],
            tools: [
                { name: "Apache Airflow", desc: "Workflow orchestration" },
                { name: "Docker Compose", desc: "Airflow deployment" },
                { name: "Python", desc: "DAG definition" }
            ],
            code: [
                {
                    label: "Python: dags/wine_pipeline_dag.py",
                    lang: "python",
                    code: `from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

with DAG(
    dag_id="wine_quality_pipeline",
    default_args=default_args,
    description="End-to-end wine quality classification pipeline",
    schedule_interval="0 2 * * 1",   # Every Monday at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "wine-quality"],
) as dag:

    download_data = BashOperator(
        task_id="download_data",
        bash_command="python data_collection.py",
    )

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="python validate_data.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="python feature_engineering.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python train_model.py",
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command="python evaluate_model.py",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python register_model.py",
    )

    # Define task dependencies
    download_data >> validate_data >> feature_engineering
    feature_engineering >> train_model >> evaluate_model >> register_model`
                },
                {
                    label: "Terminal: Start Airflow with Docker Compose",
                    lang: "bash",
                    code: `# Initialize Airflow
docker compose up airflow-init

# Start all Airflow services
docker compose up -d

# Access the Airflow UI
# Open http://localhost:8080 in your browser
# Default credentials: airflow / airflow`
                }
            ],
            output: `[2025-01-06 02:00:00] INFO - Starting DAG: wine_quality_pipeline
[2025-01-06 02:00:01] INFO - Running: download_data
[2025-01-06 02:00:15] INFO - Success: download_data
[2025-01-06 02:00:16] INFO - Running: validate_data
[2025-01-06 02:00:20] INFO - Success: validate_data
[2025-01-06 02:00:21] INFO - Running: feature_engineering
[2025-01-06 02:00:28] INFO - Success: feature_engineering
[2025-01-06 02:00:29] INFO - Running: train_model
[2025-01-06 02:01:05] INFO - Success: train_model
[2025-01-06 02:01:06] INFO - Running: evaluate_model
[2025-01-06 02:01:12] INFO - Success: evaluate_model
[2025-01-06 02:01:13] INFO - Running: register_model
[2025-01-06 02:01:18] INFO - Success: register_model
[2025-01-06 02:01:18] INFO - DAG run completed successfully`,
            checklist: [
                "Airflow web UI is accessible at http://localhost:8080",
                "DAG 'wine_quality_pipeline' is visible and enabled",
                "Task dependencies are correct: download > validate > features > train > evaluate > register",
                "Schedule is set to weekly (Monday at 2 AM)",
                "Retry policy is configured (2 retries, 5 min delay)",
                "Manual trigger runs all tasks in correct order"
            ]
        },
        {
            id: 10,
            title: "Monitoring",
            shortLabel: "Monitoring",
            tagline: "Instrument the API with Prometheus metrics and visualize them in Grafana dashboards.",
            what: "The FastAPI service is instrumented with Prometheus client libraries to expose a /metrics endpoint. Prometheus scrapes this endpoint at regular intervals, collecting request count, latency histograms, error rates, and prediction distribution metrics. Grafana connects to Prometheus as a data source and displays real-time dashboards for operational health, enabling rapid detection of performance degradation or data drift.",
            concepts: [
                "Observability: metrics, logs, and traces",
                "Prometheus pull-based metric collection",
                "Histogram buckets for latency percentiles (p50, p95, p99)",
                "Grafana dashboard design and alerting",
                "Prediction distribution monitoring for drift detection",
                "RED method: Rate, Errors, Duration"
            ],
            tools: [
                { name: "Prometheus", desc: "Metrics collection" },
                { name: "Grafana", desc: "Dashboard visualization" },
                { name: "prometheus-client", desc: "Python instrumentation" },
                { name: "Docker Compose", desc: "Service orchestration" }
            ],
            code: [
                {
                    label: "Python: Add Prometheus metrics to serve.py",
                    lang: "python",
                    code: `from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest
)
from fastapi import Response
import time

# Define metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_DISTRIBUTION = Counter(
    "prediction_class_total",
    "Prediction class distribution",
    ["class_label"]
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Current model version"
)
MODEL_VERSION.set(1)

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4"
    )

# Update the predict endpoint to track metrics
@app.post("/predict")
def predict_with_metrics(features: WineFeatures):
    start_time = time.time()
    # ... (existing prediction logic) ...
    duration = time.time() - start_time

    REQUEST_COUNT.labels("POST", "/predict", "200").inc()
    REQUEST_LATENCY.observe(duration)
    PREDICTION_DISTRIBUTION.labels(
        "good" if pred == 1 else "not_good"
    ).inc()

    return result`
                },
                {
                    label: "YAML: docker-compose.monitoring.yml",
                    lang: "yaml",
                    code: `version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin`
                },
                {
                    label: "YAML: prometheus.yml",
                    lang: "yaml",
                    code: `global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "wine-quality-api"
    static_configs:
      - targets: ["host.docker.internal:8000"]
    metrics_path: "/metrics"`
                }
            ],
            output: `# Prometheus targets page (http://localhost:9090/targets)
wine-quality-api (1/1 up)

# Sample /metrics output
prediction_requests_total{method="POST",endpoint="/predict",status="200"} 142
prediction_latency_seconds_bucket{le="0.05"} 138
prediction_latency_seconds_bucket{le="0.1"} 141
prediction_class_total{class_label="not_good"} 124
prediction_class_total{class_label="good"} 18
model_version_info 1.0`,
            checklist: [
                "FastAPI /metrics endpoint returns Prometheus-formatted text",
                "Prometheus is scraping the API target (status: UP at localhost:9090/targets)",
                "Grafana is accessible at http://localhost:3000",
                "Prometheus is added as a data source in Grafana",
                "Dashboard panels show: request rate, latency histogram, prediction distribution",
                "Alert rules are configured for high latency or error rate spikes"
            ]
        }
    ];

    /* ----------------------------------------------------------
       STATE
       ---------------------------------------------------------- */
    let currentStage = 0; // 0-indexed

    /* ----------------------------------------------------------
       DOM REFS
       ---------------------------------------------------------- */
    const pipelineTrack   = document.getElementById("pipelineTrack");
    const detailCard      = document.getElementById("detailCard");
    const detailGrid      = document.getElementById("detailGrid");
    const progressFill    = document.getElementById("progressFill");
    const progressLabel   = document.getElementById("progressLabel");
    const btnPrev         = document.getElementById("btnPrev");
    const btnNext         = document.getElementById("btnNext");
    const navDotsContainer = document.getElementById("navDots");

    /* ----------------------------------------------------------
       BUILD PIPELINE NODES
       ---------------------------------------------------------- */
    function buildPipelineTrack() {
        // Remove existing nodes and connectors (keep SVGs)
        const existingNodes = pipelineTrack.querySelectorAll(".pipeline-node, .node-connector");
        existingNodes.forEach(n => n.remove());

        STAGES.forEach((stage, i) => {
            // Node
            const node = document.createElement("div");
            node.className = "pipeline-node";
            node.dataset.index = i;
            node.innerHTML = `
                <div class="node-circle">${String(stage.id).padStart(2, "0")}</div>
                <div class="node-label">${stage.shortLabel}</div>
            `;
            node.addEventListener("click", () => goToStage(i));
            pipelineTrack.appendChild(node);

            // Connector (except after last)
            if (i < STAGES.length - 1) {
                const conn = document.createElement("div");
                conn.className = "node-connector";
                conn.dataset.after = i;
                pipelineTrack.appendChild(conn);
            }
        });
    }

    /* ----------------------------------------------------------
       BUILD NAV DOTS
       ---------------------------------------------------------- */
    function buildNavDots() {
        navDotsContainer.innerHTML = "";
        STAGES.forEach((_, i) => {
            const dot = document.createElement("div");
            dot.className = "nav-dot";
            dot.dataset.index = i;
            dot.addEventListener("click", () => goToStage(i));
            navDotsContainer.appendChild(dot);
        });
    }

    /* ----------------------------------------------------------
       DRAW RETRAINING LOOP
       ---------------------------------------------------------- */
    function drawRetrainingLoop() {
        const svg = document.getElementById("retrainingLoop");
        if (!svg) return;

        // We will redraw after nodes are laid out
        requestAnimationFrame(() => {
            const nodes = pipelineTrack.querySelectorAll(".pipeline-node");
            if (nodes.length < 2) return;

            const firstNode = nodes[0];
            const lastNode  = nodes[nodes.length - 1];

            const trackRect = pipelineTrack.getBoundingClientRect();
            const firstRect = firstNode.querySelector(".node-circle").getBoundingClientRect();
            const lastRect  = lastNode.querySelector(".node-circle").getBoundingClientRect();

            const x1 = lastRect.left + lastRect.width / 2 - trackRect.left;
            const y1 = lastRect.bottom - trackRect.top + 8;
            const x2 = firstRect.left + firstRect.width / 2 - trackRect.left;
            const y2 = firstRect.bottom - trackRect.top + 8;

            const midY = y1 + 36;

            svg.setAttribute("viewBox", `0 0 ${trackRect.width} 56`);
            svg.style.width = trackRect.width + "px";

            svg.innerHTML = `
                <defs>
                    <marker id="arrowLoop" viewBox="0 0 10 7" refX="10" refY="3.5"
                        markerWidth="8" markerHeight="6" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 3.5 L 0 7 z" fill="#a78bfa" opacity="0.6"/>
                    </marker>
                </defs>
                <path class="retraining-path"
                    d="M ${x1} ${y1 - trackRect.top - (trackRect.height - 56)}
                       C ${x1} ${midY - trackRect.top}, ${x2} ${midY - trackRect.top}, ${x2} ${y2 - trackRect.top - (trackRect.height - 56)}"
                    marker-end="url(#arrowLoop)"
                />
            `;
        });
    }

    /* ----------------------------------------------------------
       SYNTAX HIGHLIGHTING (simple CSS-based)
       ---------------------------------------------------------- */
    function highlightCode(code, lang) {
        // Escape HTML first
        let escaped = code
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        if (lang === "python") {
            // Comments
            escaped = escaped.replace(/(#[^\n]*)/g, '<span class="syn-comment">$1</span>');
            // Strings (double quotes)
            escaped = escaped.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="syn-string">$1</span>');
            // Strings (single quotes)
            escaped = escaped.replace(/('(?:[^'\\]|\\.)*')/g, '<span class="syn-string">$1</span>');
            // Keywords
            const pyKeywords = /\b(import|from|as|def|class|return|if|else|elif|for|in|with|not|and|or|True|False|None|assert|print|exit)\b/g;
            escaped = escaped.replace(pyKeywords, '<span class="syn-keyword">$1</span>');
            // Numbers
            escaped = escaped.replace(/\b(\d+\.?\d*)\b/g, '<span class="syn-number">$1</span>');
            // f-strings markers
            escaped = escaped.replace(/\bf(&lt;|")/g, '<span class="syn-keyword">f</span>$1');
        } else if (lang === "bash") {
            // Comments
            escaped = escaped.replace(/(#[^\n]*)/g, '<span class="syn-comment">$1</span>');
            // Strings
            escaped = escaped.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="syn-string">$1</span>');
            escaped = escaped.replace(/('(?:[^'\\]|\\.)*')/g, '<span class="syn-string">$1</span>');
            // Flags
            escaped = escaped.replace(/(\s)(--?\w[\w-]*)/g, '$1<span class="syn-flag">$2</span>');
            // Commands at start of line
            escaped = escaped.replace(/^(\w+)/gm, '<span class="syn-function">$1</span>');
        } else if (lang === "yaml") {
            // Comments
            escaped = escaped.replace(/(#[^\n]*)/g, '<span class="syn-comment">$1</span>');
            // Keys
            escaped = escaped.replace(/^(\s*)([\w-]+)(:)/gm, '$1<span class="syn-keyword">$2</span>$3');
            // Strings
            escaped = escaped.replace(/("(?:[^"\\]|\\.)*")/g, '<span class="syn-string">$1</span>');
            // Booleans
            escaped = escaped.replace(/\b(true|false)\b/gi, '<span class="syn-number">$1</span>');
        }

        return escaped;
    }

    /* ----------------------------------------------------------
       RENDER STAGE DETAIL
       ---------------------------------------------------------- */
    function renderStage(index) {
        const stage = STAGES[index];

        // Header
        document.getElementById("detailStageNumber").textContent = String(stage.id).padStart(2, "0");
        document.getElementById("detailTitle").textContent = stage.title;
        document.getElementById("detailTagline").textContent = stage.tagline;

        // What happens here
        document.getElementById("blockWhat").innerHTML = `<p>${stage.what}</p>`;

        // Key concepts
        const conceptsList = document.getElementById("blockConcepts");
        conceptsList.innerHTML = stage.concepts.map(c => `<li>${c}</li>`).join("");

        // Tools
        const toolsGrid = document.getElementById("blockTools");
        toolsGrid.innerHTML = stage.tools.map(t =>
            `<div class="tool-chip">
                <span>${t.name}</span>
                <span class="tool-desc">${t.desc}</span>
            </div>`
        ).join("");

        // Code blocks
        const codeContainer = document.getElementById("blockCode");
        codeContainer.innerHTML = stage.code.map((block, ci) => {
            const highlighted = highlightCode(block.code, block.lang);
            const codeId = `code-${index}-${ci}`;
            return `
                <div class="code-container">
                    <div class="code-label">${block.label}</div>
                    <pre class="code-block" id="${codeId}">${highlighted}</pre>
                    <button class="copy-btn" data-target="${codeId}" onclick="copyCode(this, '${codeId}')">Copy</button>
                </div>
            `;
        }).join("");

        // Expected output
        document.getElementById("blockOutput").innerHTML =
            `<div class="output-block">${stage.output}</div>`;

        // Checklist
        const checkList = document.getElementById("blockCheck");
        checkList.innerHTML = stage.checklist.map(c => `<li>${c}</li>`).join("");

        // Re-trigger staggered animation
        const blocks = detailGrid.querySelectorAll(".detail-block");
        blocks.forEach(b => {
            b.style.animation = "none";
            // Force reflow
            void b.offsetHeight;
            b.style.animation = "";
        });
    }

    /* ----------------------------------------------------------
       UPDATE UI STATE
       ---------------------------------------------------------- */
    function updateUI() {
        // Pipeline nodes
        const nodes = pipelineTrack.querySelectorAll(".pipeline-node");
        nodes.forEach((node, i) => {
            node.classList.remove("active", "completed");
            if (i === currentStage) node.classList.add("active");
            else if (i < currentStage) node.classList.add("completed");
        });

        // Connectors
        const connectors = pipelineTrack.querySelectorAll(".node-connector");
        connectors.forEach((conn, i) => {
            conn.classList.toggle("active", i < currentStage);
        });

        // Nav dots
        const dots = navDotsContainer.querySelectorAll(".nav-dot");
        dots.forEach((dot, i) => {
            dot.classList.toggle("active", i === currentStage);
        });

        // Progress
        const pct = ((currentStage + 1) / STAGES.length) * 100;
        progressFill.style.width = pct + "%";
        progressLabel.textContent = `Stage ${currentStage + 1} of ${STAGES.length}`;

        // Nav buttons
        btnPrev.disabled = currentStage === 0;
        btnNext.disabled = currentStage === STAGES.length - 1;

        // Scroll active node into view
        const activeNode = pipelineTrack.querySelector(".pipeline-node.active");
        if (activeNode) {
            activeNode.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
        }

        // Render content
        renderStage(currentStage);
    }

    /* ----------------------------------------------------------
       NAVIGATION
       ---------------------------------------------------------- */
    function goToStage(index) {
        if (index < 0 || index >= STAGES.length) return;
        currentStage = index;
        updateUI();

        // Smooth scroll detail into view on mobile
        if (window.innerWidth < 768) {
            detailCard.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    btnPrev.addEventListener("click", () => goToStage(currentStage - 1));
    btnNext.addEventListener("click", () => goToStage(currentStage + 1));

    // Keyboard navigation
    document.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
            e.preventDefault();
            goToStage(currentStage - 1);
        } else if (e.key === "ArrowRight" || e.key === "ArrowDown") {
            e.preventDefault();
            goToStage(currentStage + 1);
        }
    });

    /* ----------------------------------------------------------
       COPY CODE
       ---------------------------------------------------------- */
    window.copyCode = function (btn, codeId) {
        const codeEl = document.getElementById(codeId);
        if (!codeEl) return;

        const text = codeEl.textContent;
        navigator.clipboard.writeText(text).then(() => {
            btn.textContent = "Copied";
            btn.classList.add("copied");
            setTimeout(() => {
                btn.textContent = "Copy";
                btn.classList.remove("copied");
            }, 2000);
        }).catch(() => {
            // Fallback
            const ta = document.createElement("textarea");
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            btn.textContent = "Copied";
            btn.classList.add("copied");
            setTimeout(() => {
                btn.textContent = "Copy";
                btn.classList.remove("copied");
            }, 2000);
        });
    };

    /* ----------------------------------------------------------
       INITIALIZE
       ---------------------------------------------------------- */
    function init() {
        buildPipelineTrack();
        buildNavDots();
        updateUI();

        // Draw retraining loop after layout settles
        setTimeout(drawRetrainingLoop, 300);
        window.addEventListener("resize", () => {
            clearTimeout(window._retrainDebounce);
            window._retrainDebounce = setTimeout(drawRetrainingLoop, 200);
        });
    }

    // Run on DOM ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
