---
name: mlflow-xgboost-python-api
description: "Authoritative skill and Python API reference for mlflow.xgboost, MLflow 3 Logged Models, ModelSignature inference, UBJSON ('ubj') model serialization, autologging, and Unity Catalog Model Registry integration."
compatibility: "MLflow (>= 3.5.1), xgboost (>= 2.1.3, <= 3.4.1), Python (>= 3.10)"
metadata:
  version: "1.0.0"
category: Machine Learning & MLOps
tags:
  - mlflow
  - mlflow-xgboost
  - xgboost
  - logged-model
  - model-signature
  - ubj
---

# MLflow 3 Python API & `mlflow.xgboost` Authoritative Reference

Grounded in official MLflow Python API documentation (`https://mlflow.org/docs/latest/api_reference/python_api/index.html` and `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.xgboost.html`) and Databricks MLflow 3 documentation (`https://docs.databricks.com/aws/en/mlflow/`).

---

## 1. Core `mlflow.xgboost` API Functions

### 1.1 `mlflow.xgboost.autolog()`
Compatible with `2.1.3 <= xgboost <= 3.4.1` (both `xgboost.train` Booster API and `xgboost.sklearn` `XGBClassifier` / `XGBRegressor` estimators).

```python
import mlflow.xgboost

mlflow.xgboost.autolog(
    importance_types=["weight", "gain", "cover"],
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False,  # Set False during Optuna HPO trials to avoid registering every trial
    log_datasets=True,
    model_format="ubj",  # Universal Binary JSON (recommended default for cross-platform safety)
    silent=True,
)
```

### 1.2 `mlflow.xgboost.log_model()`
Logs an XGBoost model (`xgboost.Booster`, `XGBClassifier`, or `XGBRegressor`) as an MLflow 3 `LoggedModel` artifact with both `mlflow.xgboost` and `mlflow.pyfunc` flavors:

```python
from mlflow.models import infer_signature
import mlflow.xgboost

signature = infer_signature(X_train, model.predict(X_train))

model_info = mlflow.xgboost.log_model(
    xgb_model=model,
    name="model",  # Use `name="model"` (note: `artifact_path` is deprecated in MLflow 3)
    model_format="ubj",  # Defaults to "ubj" (UBJSON); also supports "json" and "xgb"
    signature=signature,
    input_example=X_train.iloc[:5],
    registered_model_name="data_engineering_dev.ml_models.xgb_customer_churn_classifier",
    params=best_params,
    tags={"model_family": "xgboost", "task_type": "binary_classification"},
)
```

### 1.3 `mlflow.xgboost.load_model()` & `mlflow.pyfunc.spark_udf()`
```python
import mlflow.pyfunc
import mlflow.xgboost

# Load native XGBoost estimator / Booster
xgb_native = mlflow.xgboost.load_model(
    "models:/data_engineering_dev.ml_models.xgb_customer_churn_classifier@Champion"
)

# Load as distributed PySpark UDF for batch inference
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri="models:/data_engineering_dev.ml_models.xgb_customer_churn_classifier@Champion",
    env_manager="local",
)
```

---

## 2. MLflow 3 Logged Models & Unity Catalog Lifecycle

1. **MLflow 3 `LoggedModel`**:
   - Every `log_model()` call creates a first-class `LoggedModel` entity with a unique `model_id` that persists across runs and environments, linking parameters, metrics, traces, and evaluation results directly on the Unity Catalog model version page.
2. **Model Signature Enforcement**:
   - Always supply `signature=infer_signature(X_train, predictions)` and `input_example=X_train.iloc[:5]` so Unity Catalog and Databricks Model Serving validate column types and nullability contracts.
3. **Movable Aliases (`@Champion`, `@Challenger`, `@PreviousChampion`)**:
   - Manage lifecycle promotions with `MlflowClient(registry_uri="databricks-uc").set_registered_model_alias()`.
