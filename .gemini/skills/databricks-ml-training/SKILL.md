---
name: databricks-ml-training
description: "Train ML models on Databricks with MLflow 3 and Unity Catalog. Use for: classification/regression (XGBoost, LightGBM, scikit-learn, PyTorch) with Optuna nested runs, @Champion/@Challenger/@prod UC aliases, batch scoring (mlflow.pyfunc.spark_udf for plain models, fe.score_batch for Feature Store models), Unity Catalog Feature Engineering (FeatureEngineeringClient, FeatureLookup, create_training_set, fe.log_model), declarative Feature Views, and custom PyFunc models."
compatibility: "Databricks Runtime ML, MLflow (>= 3.5.1), xgboost (>= 2.1.3, <= 3.4.1), databricks-feature-engineering (>= 0.16.0)"
metadata:
  version: "1.0.0"
category: Machine Learning & MLOps
tags:
  - mlflow
  - mlflow-3
  - xgboost
  - optuna
  - feature-store
  - unity-catalog
  - databricks-ml
---

# Databricks ML Training with MLflow 3, XGBoost & Unity Catalog

Train with MLflow 3 -> register to Unity Catalog (`databricks-uc`) -> promote movable aliases (`@Champion`, `@Challenger`) -> consume the governed model via distributed Spark batch inference (`fe.score_batch` or `mlflow.pyfunc.spark_udf`) or Databricks Model Serving.

---

## 1. End-to-End Architecture

```
data_engineering_dev.gold.vw_ml_* (Upstream Gold Feature Views from Project 05)
        │
        ▼
  Functional PySpark Feature Engineering (.withColumns, Column -> Column, .transform)
        │
        ▼
  Unity Catalog Feature Table (FeatureEngineeringClient.create_table / write_table)
        │
        ▼
  FeatureLookup + fe.create_training_set()
        │
        ▼
  Optuna Hyperparameter Study (Parent MLflow Run + Nested Child Runs)
   ├── mlflow.xgboost.autolog(log_models=False) during HPO trials
   ├── Retrain winning XGBClassifier / XGBRegressor on Train + Validation
   ├── Evaluate on untouched Holdout Test Set (mlflow.evaluate + feature importance)
   ├── Register to Unity Catalog via fe.log_model(..., flavor=mlflow.xgboost)
   └── Promote movable alias (@Champion / @Challenger) via MlflowClient
        │
        ▼
  Distributed Batch Inference (fe.score_batch / mlflow.pyfunc.spark_udf)
        │
        ▼
  data_engineering_dev.ml_predictions.pred_*_xgboost
```

---

## 2. Critical MLflow 3 & Unity Catalog Rules

1. **Always Set Unity Catalog Registry URI**:
   - Call `mlflow.set_registry_uri("databricks-uc")` before any experiment or model registration call. Without it, models land in the deprecated workspace registry.
2. **Pre-Create Experiment Parent Directories**:
   - `mlflow.set_experiment("/Shared/my_project/experiment")` requires `/Shared/my_project` (or `/Users/<email>/...`) to exist in the workspace.
3. **Movable Aliases Instead of Deprecated Stages**:
   - Unity Catalog Model Registry dropped `Staging` and `Production` stages. Always use `MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(model_name, "Champion", version)` and reference via `models:/<catalog>.<schema>.<model>@Champion`.
4. **Prevent HPO Trial Registry Pollution**:
   - During Optuna trials, disable model registration (`mlflow.xgboost.autolog(log_models=False)` or omit `registered_model_name`). Only log and register the final retrained candidate model in `"ubj"` (Universal Binary JSON) format.
5. **Feature Store Lineage vs. Direct PyFunc UDF**:
   - Models logged with `fe.log_model(..., training_set=training_set)` embed Unity Catalog feature lineage and are scored with `fe.score_batch(model_uri=..., df=keys_only_df)`.
   - Plain models logged with `mlflow.xgboost.log_model(..., model_format="ubj")` are scored with `mlflow.pyfunc.spark_udf(spark, model_uri=..., env_manager="local")`.

---

## 3. Canonical XGBoost + Optuna + MLflow 3 Pattern

```python
import mlflow
import mlflow.xgboost
import optuna
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/mlflow_xgboost_customer_churn")

FULL_MODEL_NAME = "data_engineering_dev.ml_models.xgb_customer_churn_classifier"

# Disable automatic model artifact logging during Optuna HPO trials
mlflow.xgboost.autolog(log_models=False, silent=True)


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "random_state": 42,
        "eval_metric": "logloss",
    }
    with mlflow.start_run(run_name=f"xgb_trial_{trial.number:02d}", nested=True):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_probs))
        mlflow.log_metric("val_roc_auc", val_auc)
        return val_auc


with mlflow.start_run(run_name="xgb_optuna_hpo_study"):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    best_params = study.best_trial.params

# Retrain final candidate on Train + Validation and register single LoggedModel
with mlflow.start_run(run_name="xgb_final_candidate"):
    final_model = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    final_model.fit(X_trainval, y_trainval)
    preds = final_model.predict(X_test)
    signature = infer_signature(X_trainval, preds)
    model_info = mlflow.xgboost.log_model(
        xgb_model=final_model,
        name="model",
        model_format="ubj",
        signature=signature,
        input_example=X_trainval.iloc[:5],
        registered_model_name=FULL_MODEL_NAME,
    )

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(FULL_MODEL_NAME, "Champion", str(model_info.registered_model_version))
```

---

## 4. Reference Guides

- [Feature Store & FeatureLookup Guide](references/feature-store.md)
- [Declarative Feature Views Guide](references/feature-views.md)
- [Custom PyFunc Models from Code Guide](references/custom-pyfunc.md)
