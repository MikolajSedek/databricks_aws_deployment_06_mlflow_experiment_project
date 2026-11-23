---
name: databricks-mlflow-evaluation
description: "Authoritative skill for evaluating classical ML models (XGBoost classifiers and regressors via mlflow.evaluate) and GenAI applications/agents (mlflow.genai.evaluate) on Databricks with MLflow 3, holdout validation gating, and Champion/Challenger promotion."
compatibility: "MLflow (>= 3.5.1), scikit-learn (>= 1.5.0), xgboost (>= 2.1.3)"
metadata:
  version: "1.0.0"
category: Machine Learning & MLOps
tags:
  - mlflow-evaluation
  - mlflow-3
  - champion-challenger
  - model-governance
---

# Databricks MLflow 3 Model Evaluation & Champion/Challenger Governance

---

## 1. Classical ML Evaluation (`mlflow.evaluate`)

Evaluate trained `XGBClassifier` (`model_type="classifier"`) and `XGBRegressor` (`model_type="regressor"`) models on an untouched Holdout Test Set:

```python
import mlflow
import pandas as pd

eval_df = X_test.copy()
eval_df["is_churn_candidate"] = y_test.values

eval_result = mlflow.evaluate(
    model=lambda df: final_xgb_classifier.predict(df[feature_cols]),
    data=eval_df,
    targets="is_churn_candidate",
    model_type="classifier",
    evaluators=["default"],
)
```

---

## 2. Gated `@Champion` / `@Challenger` Alias Promotion in Unity Catalog

```python
from mlflow.tracking import MlflowClient


def promote_model_version_if_better(
    client: MlflowClient,
    uc_model_name: str,
    candidate_version: str,
    candidate_metric: float,
    metric_key: str,
    min_threshold: float,
    higher_is_better: bool = True,
) -> str:
    """Compare candidate against current @Champion and assign @Champion or @Challenger alias."""
    try:
        current_champ = client.get_model_version_by_alias(name=uc_model_name, alias="Champion")
        champ_run = client.get_run(current_champ.run_id)
        champ_metric = float(
            champ_run.data.metrics.get(
                metric_key,
                float("-inf") if higher_is_better else float("inf"),
            )
        )
    except Exception:
        current_champ = None
        champ_metric = float("-inf") if higher_is_better else float("inf")

    passes_threshold = (
        candidate_metric >= min_threshold if higher_is_better else candidate_metric <= min_threshold
    )
    beats_champion = (
        candidate_metric >= champ_metric if higher_is_better else candidate_metric <= champ_metric
    )

    if passes_threshold and beats_champion:
        if current_champ and str(current_champ.version) != str(candidate_version):
            client.set_registered_model_alias(
                name=uc_model_name,
                alias="PreviousChampion",
                version=str(current_champ.version),
            )
        client.set_registered_model_alias(
            name=uc_model_name,
            alias="Champion",
            version=str(candidate_version),
        )
        return "Champion"

    client.set_registered_model_alias(
        name=uc_model_name,
        alias="Challenger",
        version=str(candidate_version),
    )
    return "Challenger"
```
