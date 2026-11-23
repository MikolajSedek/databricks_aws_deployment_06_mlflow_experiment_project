---
name: databricks-mcps
description: "Authoritative reference and setup guide for Model Context Protocol (MCP) servers operating on Databricks Lakehouse: MLflow MCP Server (mlflow-mcp), Genie One MCP, Databricks AI Dev Kit MCP (44 tools), Databricks Labs MCP, Databricks Agent Kit (dak), DQX MCP, GitHub MCP, and AWS Docs MCP."
compatibility: "Model Context Protocol (MCP), FastMCP, MLflow (>= 3.5.1), Databricks Unity Gateway, Python"
metadata:
  version: "2.0.0"
category: AI & Agent Tooling
tags:
  - mcp
  - mlflow-mcp
  - model-context-protocol
  - genie-one
  - ai-dev-kit
  - databricks
  - unity-catalog
  - github-mcp
  - aws-mcp
---

# Authoritative MCP Servers for Databricks MLflow & Lakehouse Projects

This guide documents the authoritative **Model Context Protocol (MCP)** servers that act as primary knowledge and operational bridges for AI coding agents (Claude Code, Cursor, Gemini, Copilot, Antigravity) working on Databricks MLflow 3 and Lakehouse projects.

---

## 1. MLflow MCP Server (`mlflow-mcp` — `mlflow[databricks,mcp]>=3.5.1`)

### Overview
Grounded in [`https://mlflow.org/docs/latest/genai/mcp/`](https://mlflow.org/docs/latest/genai/mcp/) and [`https://docs.databricks.com/aws/en/mlflow3/genai/tracing/mlflow-mcp`](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/mlflow-mcp), the **MLflow MCP server** enables AI coding assistants to programmatically search, inspect, tag, and log assessments/feedback on MLflow 3 experiment traces stored in Databricks.

### Available Tools:
| Tool | Description |
|---|---|
| `search_traces` | Search traces by `experiment_id`, `filter_string`, `max_results`, and `extract_fields`. |
| `get_trace` | Retrieve a trace and its span tree by `trace_id` and optional `extract_fields`. |
| `delete_traces` | Delete traces by `experiment_id`, `trace_ids`, or `max_timestamp_millis`. |
| `set_trace_tag` / `delete_trace_tag` | Manage key-value metadata tags on traces. |
| `log_feedback` / `log_expectation` | Log human or automated quality feedback and expected outputs onto traces. |
| `get_assessment` / `update_assessment` / `delete_assessment` | Retrieve, update, or remove trace assessments. |

### Secure Configuration for Databricks (Zero Secret Leakage):
> [!CAUTION]
> **NEVER** write plaintext `DATABRICKS_TOKEN` or `DATABRICKS_CLIENT_SECRET` into `.mcp.json`, `.vscode/mcp.json`, or `.cursor/mcp.json`. Supply authentication exclusively via runtime environment variables (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, or `DATABRICKS_CONFIG_PROFILE`).

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

---

## 2. Databricks Genie One MCP (`system.ai.genie_one_mcp`)

### Overview
The **Genie One MCP server** is an official Databricks-provided MCP hosted on **Unity Gateway**. It connects AI agents directly to Databricks Genie, allowing natural language queries grounded in your enterprise **Genie Ontology** (governed semantic layer, metric definitions, and entity relationships).

### Available Tools:
| Tool | Description |
|---|---|
| `genie_ask` | Ask Genie a natural-language data question. Initiates asynchronous query generation and execution. |
| `genie_poll_response` | Polls in-flight execution status, progress narration, and query results. |
| `genie_get_query_result` | Retrieves complete SQL result rows and schema when answers are truncated. |
| `genie_cancel_response` | Cancels an in-flight Genie execution. |
| `view_ask` | Interactive view ask for clients supporting MCP Apps UI extensions. |

### Connection Configuration:
- **Unity Gateway Endpoint**:
  `https://<workspace-hostname>/ai-gateway/mcp-services/system.ai.genie_one_mcp?o=<workspace-id>`
- **Authentication**: OAuth on-behalf-of user token or Service Principal OAuth with `ai-gateway` scope.

---

## 3. Databricks AI Dev Kit MCP Server (`databricks-mcp-server`)

### Overview
Provided by [`github.com/databricks-solutions/ai-dev-kit`](https://github.com/databricks-solutions/ai-dev-kit) (and paired with official skills from [`databricks/databricks-agent-skills`](https://github.com/databricks/databricks-agent-skills) and [`mlflow/skills`](https://github.com/mlflow/skills) via `databricks aitools install`), this FastMCP server exposes **44 authoritative tools** directly into your IDE/agent, enabling the agent to inspect UC volumes/tables, query SQL warehouses, manage Lakeflow pipelines, and orchestrate ML jobs.

### Key Tool Categories:
1. **SQL & Data Querying**:
   - `execute_sql`: Runs queries on a Databricks SQL Warehouse.
   - `get_table_stats_and_schema`: Retrieves row counts, column schemas, and data statistics (ideal for inspecting `data_engineering_dev.gold.vw_ml_*` and `ml_features.fe_*`).
   - `get_volume_folder_details`: Inspects files and structure inside Unity Catalog Volumes.
2. **Spark Declarative Pipelines (Lakeflow)**:
   - `manage_pipeline` (`create`, `create_or_update`, `get`, `update`, `delete`, `find_by_name`).
   - `manage_pipeline_run` (`start`, `get`, `stop`, `get_events`).
3. **Unity Catalog Governance**:
   - `manage_uc_objects`: Catalogs, schemas, volumes, external locations.
   - `manage_uc_grants`: Access controls, permissions.
   - `manage_uc_monitors`: Data quality and model drift monitoring in Lakehouse.
4. **Workflows & Jobs**:
   - `manage_jobs`: Scaffolds and modifies Multi-Task Jobs (e.g., feature engineering -> Optuna HPO -> Champion promotion -> batch scoring).
   - `manage_job_runs`: Triggers, monitors, and validates job runs.

---

## 4. Databricks Labs MCP ([`databrickslabs/mcp`](https://github.com/databrickslabs/mcp))

### Overview
The official Databricks Labs repository for Model Context Protocol integrations. It provides:
- Unity Catalog functions and metadata tooling.
- SQL query execution endpoints over Databricks SQL Warehouses.
- Templates for hosting custom internal MCP servers directly on **Databricks Apps**.

---

## 5. Databricks Agent Kit ([`alexmarco/databricks-agent-kit`](https://github.com/alexmarco/databricks-agent-kit) – `dak`)

### Overview
A lightweight, security-hardened MCP server designed for coding agents that connects over `stdio` or Streamable HTTP, operating strictly over public REST APIs and enforcing Unity Catalog read-only permissions.

---

## 6. Databricks Labs DQX MCP Server

### Overview
Provided by **Databricks Labs DQX** ([databrickslabs.github.io/dqx](https://databrickslabs.github.io/dqx/)), this server equips agents with automated tools for profiling feature tables, validating data contracts, and inspecting data quality metrics.

---

## 7. Official GitHub MCP Server ([`@modelcontextprotocol/server-github`](https://github.com/modelcontextprotocol/servers/tree/main/src/github))

### Overview
Enables coding agents to interact with GitHub issues, branches, and GitHub Actions CI/CD workflows directly.

---

## 8. AWS Documentation & Architecture MCP ([`awslabs/mcp-server-aws`](https://github.com/awslabs/mcp-server-aws))

### Overview
Provides semantic search across official AWS documentation and AWS Well-Architected Framework guidance for S3, IAM, and KMS supporting Databricks Unity Catalog on AWS.
