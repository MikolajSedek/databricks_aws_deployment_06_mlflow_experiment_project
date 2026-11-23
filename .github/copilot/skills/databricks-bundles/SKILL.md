---
name: databricks-bundles
description: "Authoritative skill and patterns for scaffolding, configuring, validating, deploying, and managing Databricks Asset Bundles (DAB) across dev, staging, and prod environments using CI/CD and the Databricks CLI."
compatibility: "Databricks CLI (>= 0.220.0), Databricks Asset Bundles"
metadata:
  version: "1.0.0"
category: CI/CD & Infrastructure
tags:
  - databricks-bundles
  - dab
  - asset-bundles
  - databricks-cli
  - ci-cd
---

# Databricks Asset Bundles (DAB) Guide

This skill provides comprehensive instructions for declaring, packaging, deploying, and automating Databricks resources (pipelines, jobs, dashboards, alerts) using **Databricks Asset Bundles (DAB)**.

---

## 1. Directory Structure

```
.
├── databricks.yml           # Root bundle configuration & environment targets
├── resources/               # Resource YAML definitions
│   ├── pipelines.yml        # Lakeflow Declarative Pipelines
│   ├── jobs.yml             # Scheduled and triggered Workflows
│   └── dashboards.yml       # Lakeview Dashboards
└── src/                     # Source code (notebooks, python modules, SQL)
    ├── python/
    │   └── pipelines/
    └── sql/
```

---

## 2. Root Manifest (`databricks.yml`)

```yaml
bundle:
  name: databricks_data_vault_and_dwh

include:
  - resources/*.yml

variables:
  catalog:
    description: "Default Unity Catalog name"
    default: "lakehouse_dev"
  schema:
    description: "Default schema name"
    default: "raw_vault"
  warehouse_id:
    description: "Serverless SQL Warehouse ID"
    lookup:
      warehouse: "Serverless Starter Warehouse"

targets:
  dev:
    default: true
    mode: development
    workspace:
      host: https://adb-xxxxxxxx.azuredatabricks.net
    variables:
      catalog: "lakehouse_dev"
      schema: "raw_vault"

  prod:
    mode: production
    workspace:
      host: https://adb-xxxxxxxx.azuredatabricks.net
      root_path: /Shared/.bundle/${bundle.name}/${bundle.target}
    variables:
      catalog: "lakehouse_prod"
      schema: "raw_vault"
    run_as:
      service_principal_name: "<service-principal-client-id>"
```

---

## 3. Pipeline Resource (`resources/pipelines.yml`)

```yaml
resources:
  pipelines:
    data_vault_pipeline:
      name: "[${bundle.target}] Lakeflow Data Vault & DWH Pipeline"
      target: ${var.schema}
      catalog: ${var.catalog}
      continuous: false
      channel: preview
      serverless: true
      configuration:
        pipeline.catalog: ${var.catalog}
        pipeline.schema: ${var.schema}
        pipeline.raw_volume_path: "/Volumes/${var.catalog}/raw_data/event_landing"
      libraries:
        - file:
            path: ../src/python/pipelines/01_raw_vault_pipeline.py
        - file:
            path: ../src/python/pipelines/02_business_vault_pipeline.py
        - file:
            path: ../src/python/pipelines/03_kimball_dwh_pipeline.py
      notifications:
        - email_recipients:
            - alerts@example.com
          on_failure: true
```

---

## 4. CLI Workflow Commands

```bash
# Validate bundle schema and variables (dry run)
databricks bundle validate

# Deploy to development environment (default)
databricks bundle deploy

# Deploy to production environment
databricks bundle deploy --target prod

# Trigger pipeline run
databricks bundle run data_vault_pipeline

# Destroy deployed bundle resources
databricks bundle destroy --auto-approve
```
