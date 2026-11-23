# GitHub Copilot Instructions

Project: **Enterprise MLflow 3 & XGBoost Experimentation, Feature Store & Model Governance on Databricks AWS**
Upstream Data Platform: **[`databricks_aws_deployment_05_data_vault_and_dwh`](https://github.com/MikolajSedek/databricks_aws_deployment_05_data_vault_and_dwh)**

Follow these authoritative guidelines when suggesting or generating code in this repository:

---

## TOP PRIORITY: Mandatory Security Guardrails (Zero Secret Leakage)

> [!CAUTION]
> ### STRICT PROHIBITION ON CREDENTIAL LEAKS & MANDATORY SECRET SCANNING
>
> 1. **NEVER Place Databricks Service Principal Secrets, PAT Tokens, or AWS Secret Keys in Any File**:
>    - **Absolute Prohibition**: Never write, hardcode, paste, commit, or reference plaintext Databricks Service Principal secrets (`DATABRICKS_CLIENT_SECRET`, `ARM_CLIENT_SECRET`), AWS secret access keys (`AWS_SECRET_ACCESS_KEY`), session tokens (`AWS_SESSION_TOKEN`), Databricks Personal Access Tokens (`DATABRICKS_TOKEN`, `dapi...`), passwords, or private keys into any newly created or updated files (e.g., `databricks.yml`, `.mcp.json`, `.vscode/mcp.json`, `.cursor/mcp.json`, Terraform configs, GitHub Actions workflows, Python modules, shell scripts, notebooks, or documentation).
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
>    - **Verification**: Run pre-commit hooks (`detect-secrets`, `detect-aws-credentials`, `bandit -c bandit.yaml`) and secret scanning checks. Immediately sanitize, scrub, and replace any detected credentials with environment variable bindings before saving.

---

## 1. Architecture & Upstream Data Contract
- **Upstream Source (`data_engineering_dev.gold`)**: Reads governed Gold ML Feature Views created by Project 05 (`vw_ml_customer_churn_features` keyed by `user_id`, and `vw_ml_product_affinity_features` keyed by `product_id`).
- **Feature Store (`data_engineering_dev.ml_features`)**: Functional PySpark transformations prepare numeric/encoded features and upsert Unity Catalog Feature Tables (`fe_customer_churn_features` and `fe_product_affinity_features`) via `FeatureEngineeringClient`.
- **MLflow 3 + Optuna + XGBoost (`mlflow.xgboost`)**:
  - `XGBClassifier` for customer churn binary classification (`is_churn_candidate`).
  - `XGBRegressor` for product affinity regression (`affinity_index_score`).
  - Nested Optuna trials with `@mlflow.trace` instrumentation, `model_format="ubj"`, `infer_signature`, and `mlflow.evaluate()` holdout verification.
- **Unity Catalog Model Registry (`data_engineering_dev.ml_models`)**:
  - Models registered under `databricks-uc` and promoted via movable aliases (`@Champion`, `@Challenger`, `@PreviousChampion`).
- **Distributed Spark Batch Inference (`data_engineering_dev.ml_predictions`)**:
  - Batch scoring via `fe.score_batch()` and `mlflow.pyfunc.spark_udf(..., env_manager="local")`.

---

## 2. Key Engineering Constraints
0. **Zero Credential Leakage & Proactive Secret Scanning**:
   - Never commit secrets (including in `.mcp.json`), always comment Service Principal IDs, and scan all modified/new files.
1. **MLflow 3 & Unity Catalog**:
   - Always call `mlflow.set_registry_uri("databricks-uc")`.
   - Use `mlflow.xgboost.autolog(log_models=False)` during Optuna trials; log only the final retrained candidate model with `name="model"`, `model_format="ubj"`, `signature`, and `input_example`.
   - Use `MlflowClient.set_registered_model_alias()` (`@Champion`, `@Challenger`); never use deprecated model stages.
2. **PySpark Style**:
   - Composable pure functions (`Column -> Column` and `DataFrame -> DataFrame`) in `src/python/modules/` chained via `.transform()`.
   - Single-projection `.withColumns({...})` dictionaries; zero chained `.withColumn()` calls.
   - Explicit `.select()` column contracts and `StructType` nullability.
3. **Testing**:
   - Test all feature transformations, Optuna + MLflow 3 XGBoost runs, alias governance, and Spark batch inference locally with `pytest`.

---

## 3. Skills Reference
Domain skills are configured in `.github/copilot/skills/` (matching `.gemini/skills/`).
