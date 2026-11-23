# Gemini Developer & Agent Guidelines

Project: **Enterprise MLflow 3 & XGBoost Experimentation, Feature Store & Model Governance on Databricks AWS**
Upstream Data Platform: **[`databricks_aws_deployment_05_data_vault_and_dwh`](https://github.com/MikolajSedek/databricks_aws_deployment_05_data_vault_and_dwh)**

This document provides architectural standards, imperative sources of truth, critical engineering constraints, and development workflows for Gemini agents working in this repository.

---

## TOP PRIORITY: Mandatory Security Guardrails (Zero Secret Leakage)

> [!CAUTION]
> ### STRICT PROHIBITION ON CREDENTIAL LEAKS & MANDATORY SECRET SCANNING
>
> 1. **NEVER Place Databricks Service Principal Secrets, PAT Tokens, or AWS Secret Keys in Any File**:
>    - **Absolute Prohibition**: Never write, hardcode, paste, commit, or reference plaintext Databricks Service Principal secrets (`DATABRICKS_CLIENT_SECRET`, `ARM_CLIENT_SECRET`), AWS secret access keys (`AWS_SECRET_ACCESS_KEY`), session tokens (`AWS_SESSION_TOKEN`), Databricks Personal Access Tokens (`DATABRICKS_TOKEN`, `dapi...`), passwords, or private keys into any newly created or updated files (e.g., `databricks.yml`, `.mcp.json`, `.vscode/mcp.json`, `.cursor/mcp.json`, Terraform configs, GitHub Actions workflows, Python modules, shell scripts, notebooks, or documentation).
>    - **MLflow MCP Config Guardrail**: When configuring the MLflow MCP server (`mlflow mcp run`), **NEVER** include `"DATABRICKS_TOKEN": "<your-token>"` or plaintext secrets in `.mcp.json` or IDE configs. Supply authentication exclusively through secure runtime environment variables (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`) or Databricks CLI profiles.
>    - **Runtime & Vault Injection Only**: All sensitive credentials must be supplied exclusively through secure runtime environment variables or vault secret managers (e.g., GitHub Actions Secrets, Databricks Secret Scopes `dbutils.secrets.get()`, or AWS Secrets Manager).
>
> 2. **Service Principal IDs for Deployment Require Explicit Purpose Comments**:
>    - **Deployment Configuration**: When a Databricks Service Principal Application/Client ID (`service_principal_name`, `client_id`, `DATABRICKS_CLIENT_ID`) or AWS Account ID / IAM Role ARN is legitimately required for deployment configurations (e.g., `databricks.yml` `run_as` blocks, Terraform provider definitions, or CI/CD job configurations), you **MUST** accompany it with a clear, descriptive comment explaining its purpose and explicitly stating that the corresponding secret must never be committed to code or configuration files.
>    - **Compliant Pattern**:
>      ```yaml
>      # Databricks Service Principal Application ID used for automated headless bundle deployment and MLflow job execution.
>      # NOTE: The client secret must NEVER be placed in code or config; provide via DATABRICKS_CLIENT_SECRET environment variable.
>      run_as:
>        service_principal_name: "<service-principal-client-id>"
>      ```
>
> 3. **Proactive Secret Scanning on All Newly Created or Updated Files**:
>    - **Pre-Save & Pre-Commit Verification**: Always thoroughly scan newly created or updated files for accidental secrets, tokens, or credential patterns before finalizing edits, staging, or committing.
>    - **Patterns to Inspect**:
>      - Databricks tokens (e.g., `dapi...`)
>      - AWS Access Key IDs (e.g., `AKIA...`, `ASIA...`) and Secret Access Keys
>      - OAuth client secrets, private keys, high-entropy tokens, or password strings
>    - **Verification**: Run pre-commit hooks (`detect-secrets`, `detect-aws-credentials`, `bandit -c bandit.yaml`) and secret scanning checks. Immediately sanitize, scrub, and replace any detected credentials with environment variable bindings before saving.

---

## 1. Imperative Sources of Truth

All MLflow 3, XGBoost, Unity Catalog Feature Engineering, and MCP implementations in this project **MUST** adhere to these authoritative references:
1. **Databricks MLflow 3 Documentation**: [`https://docs.databricks.com/aws/en/mlflow/`](https://docs.databricks.com/aws/en/mlflow/)
2. **MLflow Python API Reference (`mlflow`, `mlflow.xgboost`, `mlflow.models`, `mlflow.pyfunc`, `mlflow.tracing`)**: [`https://mlflow.org/docs/latest/api_reference/python_api/index.html`](https://mlflow.org/docs/latest/api_reference/python_api/index.html)
3. **MLflow MCP Server Documentation**: [`https://mlflow.org/docs/latest/genai/mcp/`](https://mlflow.org/docs/latest/genai/mcp/)
4. **Databricks MLflow 3 Tracing & MCP Server**: [`https://docs.databricks.com/aws/en/mlflow3/genai/tracing/mlflow-mcp`](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/mlflow-mcp)
5. **Databricks AI Dev Kit & Agent Skills**: [`https://github.com/databricks-solutions/ai-dev-kit/tree/main`](https://github.com/databricks-solutions/ai-dev-kit/tree/main)
6. **Upstream Data Platform (Project 05 — Data Vault 2.0 & Kimball DWH)**: [`https://github.com/MikolajSedek/databricks_aws_deployment_05_data_vault_and_dwh`](https://github.com/MikolajSedek/databricks_aws_deployment_05_data_vault_and_dwh)

---

## 2. Architecture Overview & Upstream Data Contracts

This repository consumes governed Gold Kimball Star Schema views produced by **Project 05 (`databricks_aws_deployment_05_data_vault_and_dwh`)** in catalog `data_engineering_dev` and implements an end-to-end **MLflow 3 + XGBoost** MLOps lifecycle across two machine learning domains:

1. **Upstream Gold Views (`data_engineering_dev.gold`)**:
   - `data_engineering_dev.gold.vw_ml_customer_churn_features`:
     - **Grain**: One row per active customer (`sk_user`, `user_id`).
     - **Attributes**: `country_code`, `loyalty_tier`, `calculated_customer_segment`, `total_orders`, `total_net_revenue_usd`, `avg_order_value_usd`, `total_events_count`, `distinct_sessions_count`, `avg_session_duration_sec`, `page_view_count`, `add_to_cart_count`, `checkout_start_count`, `purchase_event_count`, `product_search_count`, `days_since_last_order`, `days_since_last_event`, `cart_to_purchase_rate`, `churn_risk_score`, and binary target `is_churn_candidate`.
   - `data_engineering_dev.gold.vw_ml_product_affinity_features`:
     - **Grain**: One row per active product (`sk_product`, `product_id`).
     - **Attributes**: `product_name`, `category`, `subcategory`, `unit_price`, `gross_margin_pct`, `total_orders`, `distinct_buyers_count`, `total_units_sold`, `total_net_revenue_usd`, `total_events_count`, `repeat_purchase_ratio`, `category_revenue_share_pct`, `cart_to_purchase_rate`, and continuous target `affinity_index_score`.
2. **Feature Engineering & Unity Catalog Feature Store (`data_engineering_dev.ml_features`)**:
   - Pure functional PySpark feature engineering in `src/python/modules/features/feature_engineering.py` encodes categorical attributes and derives interaction ratios using `.withColumns({...})` and `.transform()`.
   - Governed Unity Catalog Feature Tables registered via `FeatureEngineeringClient`:
     - `data_engineering_dev.ml_features.fe_customer_churn_features` (Primary Key: `user_id`)
     - `data_engineering_dev.ml_features.fe_product_affinity_features` (Primary Key: `product_id`)
3. **Optuna Hyperparameter Optimization & MLflow 3 Experiment Tracking (`mlflow.xgboost` + `@mlflow.trace`)**:
   - **Customer Churn Classification (`XGBClassifier`)**: Experiment `/Shared/mlflow_xgboost_customer_churn`, optimizing validation `val_roc_auc` across nested Optuna TPE child runs with class-imbalance `scale_pos_weight`.
   - **Product Affinity Regression (`XGBRegressor`)**: Experiment `/Shared/mlflow_xgboost_product_affinity`, optimizing validation `val_rmse` / `val_r2` across nested Optuna TPE child runs.
   - Instrumented with `@mlflow.trace` and `mlflow.start_span()` for programmatic trace inspection via the **MLflow MCP Server (`mlflow-mcp`)**.
4. **Unity Catalog Model Registry (`data_engineering_dev.ml_models`) & `@Champion` / `@Challenger` Governance**:
   - Final candidates retrained on Train + Validation splits, evaluated on untouched Holdout Test sets with `mlflow.evaluate()`, logged in Universal Binary JSON (`model_format="ubj"`) with inferred `ModelSignature` and `input_example`, and registered to Unity Catalog (`databricks-uc`):
     - `data_engineering_dev.ml_models.xgb_customer_churn_classifier`
     - `data_engineering_dev.ml_models.xgb_product_affinity_regressor`
   - Automated Champion/Challenger promotion assigns movable `@Champion`, `@PreviousChampion`, and `@Challenger` aliases via `MlflowClient.set_registered_model_alias()`.
5. **Distributed Batch Inference (`data_engineering_dev.ml_predictions`)**:
   - Distributed Spark scoring via `fe.score_batch()` and `mlflow.pyfunc.spark_udf(..., env_manager="local")` writing governed prediction tables:
     - `data_engineering_dev.ml_predictions.pred_customer_churn_xgboost`
     - `data_engineering_dev.ml_predictions.pred_product_affinity_xgboost`

---

## 3. Critical Engineering Constraints

0. **Zero Credential Leakage & Proactive Secret Scanning**:
   - Never commit secrets, tokens, or keys (including in `.mcp.json`), always comment Service Principal IDs, and scan all modified/new files.
1. **MLflow 3 & `mlflow.xgboost` Standards**:
   - Always configure `mlflow.set_registry_uri("databricks-uc")`.
   - Use `mlflow.xgboost.autolog(log_models=False, silent=True)` during Optuna trials so individual HPO trials do not pollute the Unity Catalog Model Registry.
   - Log final candidate models with `name="model"`, `model_format="ubj"`, `signature=infer_signature(...)`, and `input_example=...` (via `fe.log_model` when bound to a UC `TrainingSet`, or `mlflow.xgboost.log_model`).
   - Use movable aliases (`@Champion`, `@Challenger`, `@PreviousChampion`) via `MlflowClient.set_registered_model_alias()`; never use deprecated stages (`Staging`, `Production`).
2. **MLflow 3 Tracing & MLflow MCP Server (`mlflow[databricks,mcp]>=3.5.1`)**:
   - Decorate experiment orchestration, HPO studies, evaluation, and alias governance functions with `@mlflow.trace` and nested `mlflow.start_span()` blocks so coding agents can inspect spans via `mlflow mcp run`.
3. **Functional PySpark Principles (Palantir & Mikołaj Sędek Standards)**:
   - Pure, single-responsibility functions residing in `src/python/modules/`.
   - Tier 1 `Column -> Column` expression builders + Tier 2 `DataFrame -> DataFrame` step functions composed via `.transform()`.
   - Batch column projections via `.withColumns({...})`; never chain multiple `.withColumn()` calls or mutate DataFrames in `for` loops.
   - Explicit `.select()` schema contracts and `StructType` nullability definitions.

---

## 4. Local Testing & Verification

- Run the complete local unit test suite (PySpark feature engineering + MLflow 3 + XGBoost HPO + alias promotion + batch scoring):
  ```bash
  uv run pytest -v
  ```
- Measure and enforce test coverage:
  ```bash
  uv run pytest --cov=src/python/modules --cov-report=term-missing
  ```
- Run static analysis, formatting, security, and docstring checks:
  ```bash
  uv run ruff check src/python tests
  uv run black --check --line-length=120 src/python tests
  uv run mypy src/python tests
  uv run bandit -c bandit.yaml -r src/python -ll -ii
  uv run interrogate -v --fail-under 95 src/python
  ```

---

## 5. Deployment via Databricks Asset Bundles (DAB)

- Deployment configuration: [`databricks.yml`](databricks.yml).
- Standard bundle commands:
  ```bash
  databricks bundle validate --target dev
  databricks bundle deploy --target dev
  databricks bundle run mlflow_xgboost_end_to_end_workflow --target dev
  ```

---

## 6. Skills Reference

Domain skills are available in `.gemini/skills/` (and mirrored in `.github/copilot/skills/`):
- `databricks-ml-training` (MLflow 3, XGBoost, Optuna, UC Feature Store `FeatureLookup`, `fe.log_model`, `fe.score_batch`, `spark_udf`)
- `mlflow-xgboost-python-api` (`mlflow.xgboost`, `ubj` serialization, `LoggedModel`, `ModelSignature`, autologging)
- `databricks-mlflow-evaluation` (`mlflow.evaluate`, holdout validation gating, `@Champion`/`@Challenger` promotion)
- `databricks-mlflow-tracing` (`@mlflow.trace`, `mlflow.start_span`, and **MLflow MCP Server** `mlflow-mcp`)
- `databricks-model-serving` (Databricks Model Serving endpoints, zero-downtime `@Champion` version swaps)
- `functional-programming`
- `pyspark-style-guide`
- `lakehouse-sql-style-guide`
- `kimball-dimensional-modeling`
- `databricks-data-vault-2-0`
- `databricks-spark-declarative-pipelines`
- `databricks-dqx`
- `databricks-bundles`
- `databricks-jobs`
- `databricks-mcps`
- `databricks-docs`
- `databricks-aws-unity-catalog`
- `databricks-aws-workspace`
- `github-workflows`
- `pytest`
- `pytest-coverage`
