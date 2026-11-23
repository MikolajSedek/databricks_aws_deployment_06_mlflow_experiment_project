---
name: databricks-model-serving
description: "Databricks Model Serving endpoint lifecycle and operations. Use when creating, updating, querying, or performing zero-downtime Champion version swaps on Databricks Model Serving endpoints."
compatibility: "Databricks CLI (>= 1.0.0), MLflow (>= 3.5.1)"
metadata:
  version: "1.0.0"
category: Machine Learning & MLOps
tags:
  - model-serving
  - databricks
  - mlflow-deployments
  - unity-catalog
---

# Databricks Model Serving Endpoint Lifecycle

---

## 1. Zero-Downtime Version Swap with Unity Catalog `@Champion` Alias

When promoting a new Unity Catalog model version to `@Champion`, update both the registered model alias and the serving endpoint configuration:

```python
from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient

FULL_NAME = "data_engineering_dev.ml_models.xgb_customer_churn_classifier"
MODEL_SHORT_NAME = "xgb_customer_churn_classifier"
ENDPOINT_NAME = "xgb-customer-churn-endpoint"

registry = MlflowClient(registry_uri="databricks-uc")
deploy = get_deploy_client("databricks")

registry.set_registered_model_alias(FULL_NAME, "Champion", new_version)
deploy.update_endpoint(
    endpoint=ENDPOINT_NAME,
    config={
        "served_entities": [
            {
                "entity_name": FULL_NAME,
                "entity_version": str(new_version),
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": f"{MODEL_SHORT_NAME}-{new_version}",
                    "traffic_percentage": 100,
                }
            ]
        },
    },
)
```

---

## 2. Two-Field Readiness Verification

Always check both `state.ready == "READY"` and `state.config_update == "NOT_UPDATING"` before querying a newly updated endpoint:

```bash
databricks serving-endpoints get xgb-customer-churn-endpoint --profile dev \
  | jq '{ready: .state.ready, config_update: .state.config_update}'
```
