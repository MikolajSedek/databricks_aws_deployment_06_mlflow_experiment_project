---
name: databricks-mlflow-tracing
description: "Authoritative skill for MLflow 3 Tracing (@mlflow.trace, mlflow.start_span) and the MLflow Model Context Protocol (MCP) Server (mlflow[databricks,mcp]>=3.5.1) on Databricks AWS."
compatibility: "MLflow (>= 3.5.1), mlflow[databricks,mcp]>=3.5.1"
metadata:
  version: "1.0.0"
category: Observability & AI Tooling
tags:
  - mlflow-tracing
  - mlflow-mcp
  - observability
  - databricks
---

# Databricks MLflow 3 Tracing & MLflow MCP Server (`mlflow-mcp`)

Grounded in official documentation:
- `https://mlflow.org/docs/latest/genai/mcp/`
- `https://docs.databricks.com/aws/en/mlflow3/genai/tracing/mlflow-mcp`

---

## 1. Instrumenting ML Workflows with `@mlflow.trace` and `mlflow.start_span`

Use `@mlflow.trace` and `mlflow.start_span()` to record structured observability traces across feature preparation, Optuna hyperparameter search, holdout evaluation, and Unity Catalog promotion:

```python
import mlflow


@mlflow.trace(name="run_xgboost_optuna_experiment", span_type="CHAIN")
def run_traced_experiment(dataset_name: str, n_trials: int) -> dict[str, object]:
    with mlflow.start_span(name="prepare_splits") as span:
        span.set_inputs({"dataset_name": dataset_name, "n_trials": n_trials})
        # Execute split and HPO
        result = {"status": "completed", "best_metric": 0.92}
        span.set_outputs(result)
    return result
```

---

## 2. MLflow MCP Server (`mlflow-mcp`) Setup & Security

### 2.1 Installation
Requires `mlflow[databricks,mcp]>=3.5.1`:
```bash
uv add "mlflow[databricks,mcp]>=3.5.1"
```

### 2.2 Secure MCP Configuration (Zero Secret Leakage)
> [!CAUTION]
> **NEVER** hardcode plaintext `DATABRICKS_TOKEN` or `DATABRICKS_CLIENT_SECRET` inside `.mcp.json`, `.vscode/mcp.json`, or `.cursor/mcp.json`. Always rely on runtime environment variables (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` or `DATABRICKS_CONFIG_PROFILE`).

```json
{
  "mcpServers": {
    "mlflow-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mlflow[databricks,mcp]>=3.5.1",
        "mlflow",
        "mcp",
        "run"
      ],
      "env": {
        "MLFLOW_TRACKING_URI": "databricks"
      }
    }
  }
}
```

### 2.3 Available MLflow MCP Tools
| Tool | Description |
|---|---|
| `search_traces` | Search traces by `experiment_id`, `filter_string`, `max_results`, and `extract_fields`. |
| `get_trace` | Retrieve full span hierarchy, inputs, outputs, and assessments for a `trace_id`. |
| `set_trace_tag` / `delete_trace_tag` | Attach or remove metadata tags on a trace. |
| `log_feedback` / `log_expectation` | Record quality ratings, human/agent feedback, and ground-truth assessments on traces. |
| `get_assessment` / `update_assessment` / `delete_assessment` | Inspect or manage trace assessments. |
| `delete_traces` | Clean up traces by `experiment_id`, `trace_ids`, or `max_timestamp_millis`. |
