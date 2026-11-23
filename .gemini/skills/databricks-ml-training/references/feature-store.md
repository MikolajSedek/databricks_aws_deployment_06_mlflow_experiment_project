# Unity Catalog Feature Engineering (`FeatureLookup` & `FeatureEngineeringClient`)

## 1. Overview & Architecture
Unity Catalog Feature Engineering prevents training-serving skew by registering a governed Delta table with a primary key constraint as a **Unity Catalog Feature Table**. The same `FeatureLookup` definition builds the training dataset and automatically joins features during `fe.score_batch()` and Databricks Model Serving inference.

```
data_engineering_dev.gold.vw_ml_customer_churn_features
  -> Functional PySpark Feature Engineering (.withColumns)
  -> fe.create_table() / fe.write_table(mode="merge")
  -> FeatureLookup + fe.create_training_set()
  -> fe.log_model(..., flavor=mlflow.xgboost, training_set=training_set)
  -> fe.score_batch(model_uri="models:/...@Champion", df=keys_only_df)
```

---

## 2. Creating and Updating Unity Catalog Feature Tables

```python
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")
mlflow.set_registry_uri("databricks-uc")

FEATURE_TABLE = "data_engineering_dev.ml_features.fe_customer_churn_features"

if not spark.catalog.tableExists(FEATURE_TABLE):
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["user_id"],
        df=engineered_features_df,
        description="Governed customer churn features derived from Kimball Gold DWH",
    )
else:
    fe.write_table(
        name=FEATURE_TABLE,
        df=engineered_features_df,
        mode="merge",
    )
```

---

## 3. Building a Lineage-Tracked Training Set & Logging XGBoost Model

```python
feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key="user_id",
        feature_names=CHURN_FEATURE_COLUMNS,
    )
]

training_set = fe.create_training_set(
    df=labels_df.select("user_id", "is_churn_candidate"),
    feature_lookups=feature_lookups,
    label="is_churn_candidate",
    exclude_columns=["user_id"],
)

training_pdf = training_set.load_df().toPandas()

# Inside final MLflow run:
fe.log_model(
    model=final_xgb_model,
    artifact_path="model",
    flavor=mlflow.xgboost,
    training_set=training_set,
    registered_model_name="data_engineering_dev.ml_models.xgb_customer_churn_classifier",
)
```

---

## 4. Distributed Batch Scoring with `fe.score_batch()`

```python
keys_df = spark.table(FEATURE_TABLE).select("user_id")
scored_df = fe.score_batch(
    model_uri="models:/data_engineering_dev.ml_models.xgb_customer_churn_classifier@Champion",
    df=keys_df,
    result_type="int",
)
```
