# Data Drift Detection & Monitoring

Study of data drift in production ML systems — both theoretical foundations and practical detection.

## Contents

| File | Description |
|------|-------------|
| `data_drift_theory.pdf` | LaTeX document covering the theory of covariate, label, and concept drift |
| `data_drift_theory.tex` | Source LaTeX for the theory document |
| `data_drift_analysis.ipynb` | Jupyter notebook demonstrating drift detection on a synthetic credit default dataset |

## Drift Types Studied

- **Covariate drift** — the input distribution P(X) changes while P(Y\|X) remains constant
- **Label drift** — the target distribution P(Y) shifts while P(X\|Y) is stable
- **Concept drift** — the relationship P(Y\|X) itself changes, even if P(X) stays the same

## Dataset

The notebook uses a synthetic **credit default prediction** dataset with features including age, income, credit score, loan amount, employment years, and region.
