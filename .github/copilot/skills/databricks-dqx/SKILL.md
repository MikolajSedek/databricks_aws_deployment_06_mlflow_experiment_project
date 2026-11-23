---
name: databricks-dqx
description: "Authoritative skill and patterns for implementing automated data quality checks, data contracts, and anomaly detection across Databricks Lakehouse pipelines using Databricks Labs DQX (Data Quality Extension) and Lakeflow Expectations."
compatibility: "Python (>= 3.10), PySpark (>= 3.4), Databricks Labs DQX, Delta Lake"
metadata:
  version: "1.0.0"
category: Data Quality & Governance
tags:
  - dqx
  - databricks-labs
  - data-quality
  - expectations
  - data-contracts
  - unity-catalog
---

# Databricks Labs DQX: Data Quality & Governance for the Lakehouse

This skill provides comprehensive instructions for deploying and automating data quality checks across Lakehouse ETL pipelines using **Databricks Labs DQX** and **Lakeflow Declarative Expectations**.

---

## 1. Overview of DQX

**DQX** ([databrickslabs.github.io/dqx](https://databrickslabs.github.io/dqx/)) is the open-source data quality framework developed by Databricks Labs. It enables teams to define, monitor, and act on data quality issues across batch and streaming pipelines.

### Key Capabilities:
- **PySpark Native**: Works directly on PySpark DataFrames (batch and Structured Streaming).
- **Lakeflow / DLT Integration**: Seamlessly maps declarative rules to Lakeflow expectations.
- **Data Contracts**: Validates schemas and business constraints against Open Data Contract Standard (ODCS) YAML specs.
- **Custom Failure Reactions**: Drop, quarantine, or alert on invalid rows.
- **Profiling & Candidate Generation**: Profiles source datasets to auto-generate rule candidates.
- **AI Coding Agent Integration (MCP)**: Exposes DQX tools over MCP for agents to query quality rules and inspect test failures.

---

## 2. Declarative Lakeflow Expectations

In Lakeflow Pipelines (`pyspark.pipelines` or SQL), apply data quality constraints directly on streaming tables and materialized views:

```python
from pyspark import pipelines as dp

@dp.table(
    name="silver.clean_customers",
    comment="Cleansed customer records with quality enforcement"
)
@dp.expect("valid_customer_id", "customer_id IS NOT NULL AND customer_id != ''")
@dp.expect_or_drop("valid_email", "email RLIKE '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'")
@dp.expect_or_fail("valid_status", "status IN ('ACTIVE', 'PENDING', 'SUSPENDED')")
def clean_customers():
    return spark.readStream.table("bronze.raw_customers")
```

| Action | Behavior | Best Used For |
|---|---|---|
| `@dp.expect` | Logs violation metrics to pipeline event log without dropping rows | Monitoring informational checks |
| `@dp.expect_or_drop` | Silently drops violating records from target table | Staging / Silver ingestion (filtering junk) |
| `@dp.expect_or_fail` | Fails the pipeline run immediately upon violation | Critical business keys and financial balances |

---

## 3. DQX Core: Rule Definitions in Code & Config

DQX allows defining quality suites declaratively in JSON/YAML configuration or in pure Python code:

### 3.1 Defining Rules in Python
```python
from dqx import QualityCheck, CheckSuite, Rule

# Define a validation suite
suite = CheckSuite(
    name="customer_quality_suite",
    checks=[
        QualityCheck(
            column="customer_id",
            rule=Rule.IS_NOT_NULL,
            action="drop"
        ),
        QualityCheck(
            column="age",
            rule=Rule.BETWEEN,
            params={"min": 18, "max": 120},
            action="quarantine"
        ),
        QualityCheck(
            column="country_code",
            rule=Rule.REGEX_MATCH,
            params={"pattern": "^[A-Z]{2}$"},
            action="alert"
        )
    ]
)

# Apply suite to a PySpark DataFrame
validated_df, quarantine_df, summary_metrics = suite.evaluate(input_df)
```

### 3.2 Quarantine Table Pattern
When bad rows arrive, rather than failing the whole pipeline, quarantine them into a dedicated table (`quarantine.customers_bad_rows`) with the failed rule metadata, allowing engineers to audit and remediate issues asynchronously.

---

## 4. Referential Integrity Checks for Data Vault

Data Vault 2.0 requires strict relationship validation between Hubs, Links, and Satellites:

```sql
-- DQX SQL Assertion: Every Link foreign key must exist in corresponding Hub
SELECT 
    l.hk_customer_order,
    l.hk_customer
FROM raw_vault.lnk_customer_order l
LEFT JOIN raw_vault.hub_customer h ON l.hk_customer = h.hk_customer
WHERE h.hk_customer IS NULL;
-- Expectation: 0 rows returned
```

---

## 5. DQX MCP Server Integration

When running in an AI-assisted environment (Claude Code, Cursor, Genie Code, Antigravity), the DQX MCP server exposes tools to:
- `dqx_profile_data(table_name)`: Profiles distributions, null counts, and candidate quality rules.
- `dqx_validate_contract(contract_path, table_name)`: Verifies table schema against an ODCS contract.
- `dqx_get_metrics(pipeline_id)`: Fetches data quality pass/fail rates from recent runs.
