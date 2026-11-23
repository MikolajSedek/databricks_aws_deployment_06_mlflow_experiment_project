---
name: databricks-jobs
description: "Authoritative skill and patterns for authoring, configuring, orchestrating, and troubleshooting Databricks Multi-Task Jobs and Workflows using Databricks Asset Bundles and Python/SQL tasks."
compatibility: "Databricks Jobs API 2.1, Databricks Asset Bundles"
metadata:
  version: "1.0.0"
category: Orchestration
tags:
  - databricks-jobs
  - workflows
  - orchestration
  - dabs
---

# Databricks Workflows & Multi-Task Jobs

This skill provides patterns and configurations for orchestrating Lakehouse ETL workflows, Data Vault ingestion, and Kimball DWH refresh jobs using **Databricks Workflows**.

---

## 1. Multi-Task DAG Architecture

In a Data Vault + Kimball architecture, orchestration tasks are organized as a directed acyclic graph (DAG):

```
┌────────────────────────────────┐
│ Task 1: Ingest Raw Events      │
│ (Volume Ingestion Check / DLT) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Task 2: Raw Vault Pipelines    │
│ (Hubs, Links, Satellites)      │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Task 3: Business Vault & PIT   │
│ (Point-in-Time & Bridge Tables)│
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Task 4: Kimball Star Schema    │
│ (Fact Tables & Dimension SCD2) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Task 5: DQX Data Quality Audit │
│ (Referential Integrity Checks) │
└────────────────────────────────┘
```

---

## 2. Defining Jobs in DAB (`resources/jobs.yml`)

```yaml
resources:
  jobs:
    nightly_data_vault_etl:
      name: "[${bundle.target}] Nightly Data Vault & DWH Pipeline"
      schedule:
        quartz_cron_expression: "0 0 2 * * ?"
        timezone_id: "UTC"
      tasks:
        - task_key: "run_pipeline"
          pipeline_task:
            pipeline_id: ${resources.pipelines.data_vault_pipeline.id}

        - task_key: "run_dq_checks"
          depends_on:
            - task_key: "run_pipeline"
          spark_python_task:
            python_file: ../src/python/tasks/run_dqx_audit.py
            parameters:
              - "--catalog"
              - ${var.catalog}
          environment_key: "serverless_env"

      environments:
        - environment_key: "serverless_env"
          spec:
            client: "1"
```
