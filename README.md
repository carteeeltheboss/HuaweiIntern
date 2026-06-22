### Phase 1: Packaging & Serving
Your goal here is to take a model out of a Jupyter Notebook and make it a software service that other applications can talk to.

**Les Grands Classiques (Checkpoints):**
* **The API Wrapper:** Train a dead-simple `scikit-learn` model (like a Random Forest on the Iris dataset). Write a FastAPI application with a `/predict` endpoint that takes in JSON data, passes it to the model, and returns the prediction. 
* **The Container:** Write a `Dockerfile` for your FastAPI app. Build the image locally and ensure it runs without crashing.
* **The Local Deployment:** Deploy that newly built container directly into your existing Portainer environment. Send a `curl` request or use Postman from your host machine to verify the containerized API is responding correctly.

---

### Phase 2: Tracking & Versioning
Your goal is to stop relying on memory or messy file names (like `model_final_v3_really.pkl`) to keep track of your experiments and data.

**Les Grands Classiques (Checkpoints):**
* **The MLflow Run:** Spin up a local MLflow tracking server. Write a Python script that trains a model with three different sets of hyperparameters. Log the parameters, the final accuracy, and the model artifact itself to MLflow so you can compare them in the UI.
* **The DVC Commit:** Initialize DVC in a local Git repository. Track a sample dataset (a simple CSV file) using `dvc add`. Push the actual data to a local dummy remote (just another folder on your machine) and commit the resulting `.dvc` file to Git. 

---

### Phase 3: CI/CD (Automation)
Your goal is to automate the testing and building of your model's code so that you catch errors before they ever reach production.

**Les Grands Classiques (Checkpoints):**
* **The Unit Test:** Write a `pytest` script that explicitly tests your FastAPI `/predict` endpoint to ensure it handles bad inputs (like missing JSON fields) gracefully instead of crashing.
* **The Pipeline:** Create a GitHub Actions workflow file (`.github/workflows/main.yml`). Configure it so that every time you push code to the repository, a runner automatically spins up, installs your requirements, and executes your `pytest` suite.

---

### Phase 4: Orchestration & Infrastructure
Your goal is to schedule and chain tasks together reliably, rather than manually running scripts in order.

**Les Grands Classiques (Checkpoints):**
* **The Airflow DAG:** Install Apache Airflow locally. Write a Directed Acyclic Graph (DAG) consisting of three distinct Python Operators: 
    1. Download/Generate dummy data.
    2. Train a model on that data.
    3. Save the model to a specific directory.
* **The Execution:** Trigger the DAG from the Airflow UI and verify that all three tasks succeed in the correct sequence.

---

### Phase 5: Monitoring & Observability
Your goal is to keep an eye on your model in the wild to ensure it isn't hogging resources or making suddenly inaccurate predictions.

**Les Grands Classiques (Checkpoints):**
* **The Metrics Exporter:** Add the `prometheus-fastapi-instrumentator` library to your Phase 1 API. This automatically exposes a `/metrics` endpoint.
* **The Dashboard:** Spin up Prometheus and Grafana. Connect Prometheus to your API's `/metrics` endpoint, and build a simple Grafana dashboard showing your API's "Requests Per Second" and "Average Response Time."
* **The Local Integration:** As a bonus, point Prometheus at your local Ollama instance and build a panel to watch its resource consumption spike during text generation.