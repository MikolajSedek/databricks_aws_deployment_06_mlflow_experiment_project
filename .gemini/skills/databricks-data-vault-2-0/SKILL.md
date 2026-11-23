---
name: databricks-data-vault-2-0
description: "Authoritative skill and patterns for architecting, modeling, implementing, and optimizing Data Vault 2.0 (Hubs, Links, Satellites, PIT tables, Bridge tables) on the Databricks Lakehouse with Lakeflow Declarative Pipelines and Delta Lake."
compatibility: "Databricks Runtime (DBR >= 15.0), Lakeflow Declarative Pipelines / Delta Live Tables, Delta Lake 3.x, Unity Catalog"
metadata:
  version: "1.0.0"
category: Data Architecture & Modeling
tags:
  - data-vault-2
  - lakehouse
  - databricks
  - delta-lake
  - lakeflow
  - raw-vault
  - business-vault
---

# Data Vault 2.0 on the Databricks Lakehouse

This skill provides comprehensive architectural patterns, production recipes, and operational guardrails for designing and implementing **Data Vault 2.0** on the **Databricks Lakehouse Platform** using **Lakeflow Declarative Pipelines (SDP/DLT)**, **Delta Lake**, and **Unity Catalog**.

---

## 1. Architectural Foundations & Authoritative Sources

All Data Vault 2.0 implementations on Databricks must align with these core standards and authoritative guides:

- **Databricks Data Vault Best Practices on the Lakehouse**:
  [Databricks Engineering Blog: Data Vault Best Practice Implementation on the Lakehouse](https://www.databricks.com/blog/data-vault-best-practice-implementation-lakehouse)
  *Definitive guidance on mapping Raw Vault and Business Vault to the Medallion architecture with Delta Live Tables / Lakeflow Pipelines.*
- **Databricks Prescriptive Guidance for Data Vault**:
  [Prescriptive Guidance for Implementing a Data Vault Model on the Databricks Lakehouse Platform](https://www.databricks.com/blog/2022/06/24/prescriptive-guidance-for-implementing-a-data-vault-model-on-the-databricks-lakehouse-platform.html)
- **Data Vault 2.0 Standard (Dan Linstedt)**:
  *Core principles: SHA-2 / SHA-1 surrogate hash keys, insert-only auditability, decoupled business keys (Hubs), relationships (Links), and context (Satellites).*
- **Datavault4dbt Patterns & Standards**:
  *Standard hashing algorithms, hash diff generation, ghost records, Point-in-Time (PIT) structures, and Bridge table acceleration.*

---

## 2. Lakehouse Medallion vs Data Vault Mapping

Data Vault 2.0 integrates cleanly into the Databricks Medallion Architecture:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             BRONZE LAYER                                 │
│  Raw Ingestion from Managed Volumes (Auto Loader / cloudFiles)           │
│  Append-only, exact source copy + _metadata (file_path, ingested_at)     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         SILVER LAYER (DATA VAULT)                        │
│                                                                          │
│  ┌───────────────────────────────┐     ┌──────────────────────────────┐  │
│  │           RAW VAULT           │     │        BUSINESS VAULT        │  │
│  │  • Hubs (Business Keys)       │────▶│  • PIT Tables (Point-in-Time)│  │
│  │  • Links (Relationships)      │     │  • Bridge Tables (Key Graph) │  │
│  │  • Satellites (Context/Hist)  │     │  • Business Satellites (BV)  │  │
│  │  Insert-only, audited fact    │     │  • Multi-Active Satellites   │  │
│  └───────────────────────────────┘     └──────────────────────────────┘  │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              GOLD LAYER                                  │
│  Information Marts & Kimball Dimensional Models (Star Schemas)           │
│  Fact Tables (Grain declared) + Dimension Tables (SCD 1 & SCD 2)         │
│  Optimized for BI dashboards, SQL Warehouses, and ML Feature Tables      │
└──────────────────────────────────────────────────────────────────────────┘
```

| Layer | Component | Target Schema Pattern | Retention / Mutability |
|---|---|---|---|
| **Bronze** | Raw Staging | `catalog.bronze_schema.stg_<source>` | Append-only, raw types, metadata captured |
| **Silver** | Raw Vault (Hubs, Links, Sats) | `catalog.raw_vault_schema.hub_*`, `lnk_*`, `sat_*` | Insert-only, audited, 100% historical fidelity |
| **Silver** | Business Vault (PIT, Bridge, BV Sats) | `catalog.biz_vault_schema.pit_*`, `br_*`, `sat_*_bv` | Materialized views or incremental tables |
| **Gold** | Kimball Star Schemas / Info Marts | `catalog.gold_schema.fct_*`, `dim_*` | Star schema, pre-aggregated, SCD Type 1 & 2 |

---

## 3. Data Vault 2.0 Core Entities & Schemas

### 3.1 Hubs (Core Business Concepts)
A Hub models a core business concept (Customer, Product, Order, Account). It contains only the unique business key, its hash key, load timestamp, and record source. Hubs never update; they are strictly **insert-only**.

**Columns**:
1. `hk_<entity>`: Hash Key (SHA-256 binary or 32/64 char string).
2. `bk_<entity>`: Natural Business Key (trimmed, upper-cased).
3. `_ldts`: Load Date Timestamp (`current_timestamp()`).
4. `_rsrc`: Record Source (e.g. `'crm.customers'`, `'erp.orders'`).

**PySpark / Lakeflow Pattern**:
```python
from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    name="raw_vault.hub_customer",
    comment="Hub storing unique customer business keys",
    cluster_by=["hk_customer"]
)
def hub_customer():
    return (
        spark.readStream.table("bronze.stg_customers")
        .select(
            F.sha2(F.upper(F.trim(F.col("customer_id"))), 256).alias("hk_customer"),
            F.upper(F.trim(F.col("customer_id"))).alias("bk_customer"),
            F.current_timestamp().alias("_ldts"),
            F.lit("volume.events.customers").alias("_rsrc")
        )
        .filter(F.col("bk_customer").isNotNull())
        .dropDuplicates(["hk_customer"])
    )
```

### 3.2 Links (Relationships)
A Link represents an association or transaction between two or more Hubs (e.g. Customer places Order, Order contains Product). Links are many-to-many by default and strictly **insert-only**.

**Columns**:
1. `hk_<link_name>`: Link Hash Key (SHA-256 of concatenated foreign business keys).
2. `hk_<parent1>`: Foreign Hash Key referencing Hub 1.
3. `hk_<parent2>`: Foreign Hash Key referencing Hub 2.
4. `_ldts`: Load Date Timestamp.
5. `_rsrc`: Record Source.

**PySpark / Lakeflow Pattern**:
```python
@dp.table(
    name="raw_vault.lnk_customer_order",
    comment="Link associating customers with orders",
    cluster_by=["hk_customer_order", "hk_customer"]
)
def lnk_customer_order():
    return (
        spark.readStream.table("bronze.stg_orders")
        .select(
            F.sha2(
                F.concat_ws("^^",
                    F.coalesce(F.upper(F.trim(F.col("customer_id"))), F.lit("-1")),
                    F.coalesce(F.upper(F.trim(F.col("order_id"))), F.lit("-1"))
                ), 256
            ).alias("hk_customer_order"),
            F.sha2(F.upper(F.trim(F.col("customer_id"))), 256).alias("hk_customer"),
            F.sha2(F.upper(F.trim(F.col("order_id"))), 256).alias("hk_order"),
            F.current_timestamp().alias("_ldts"),
            F.lit("volume.events.orders").alias("_rsrc")
        )
        .filter(F.col("hk_customer").isNotNull() & F.col("hk_order").isNotNull())
        .dropDuplicates(["hk_customer_order"])
    )
```

### 3.3 Satellites (Context & Historical Attributes)
Satellites store descriptive context over time for a Hub or Link. Whenever an attribute changes, a new row is appended with the new load timestamp. No row is ever updated or deleted.

**Columns**:
1. `hk_<parent>`: Hash Key referencing parent Hub or Link.
2. `hashdiff`: Hash of all descriptive payload columns (used to detect changes).
3. Payload columns: `name`, `email`, `status`, `address`, etc.
4. `_ldts`: Load Date Timestamp.
5. `_rsrc`: Record Source.

**Hashdiff Computation**:
```python
def compute_hashdiff(columns: list[str]) -> F.Column:
    """Computes SHA-256 hashdiff across sorted descriptive attributes with null safety."""
    parts = [
        F.coalesce(F.upper(F.trim(F.col(c).cast("string"))), F.lit("^^NULL^^"))
        for c in sorted(columns)
    ]
    return F.sha2(F.concat_ws("^^", *parts), 256)
```

**PySpark / Lakeflow Pattern (Append-only change capture)**:
```python
@dp.table(
    name="raw_vault.sat_customer",
    comment="Raw Satellite capturing historical customer profile changes",
    cluster_by=["hk_customer", "_ldts"]
)
def sat_customer():
    payload_cols = ["first_name", "last_name", "email", "phone", "country"]
    return (
        spark.readStream.table("bronze.stg_customers")
        .select(
            F.sha2(F.upper(F.trim(F.col("customer_id"))), 256).alias("hk_customer"),
            compute_hashdiff(payload_cols).alias("hashdiff"),
            *[F.col(c) for c in payload_cols],
            F.current_timestamp().alias("_ldts"),
            F.lit("volume.events.customers").alias("_rsrc")
        )
        .dropDuplicates(["hk_customer", "hashdiff"])
    )
```

---

## 4. Business Vault Patterns

The Business Vault applies business logic, calculations, and performance acceleration structures without altering the Raw Vault facts.

### 4.1 Point-in-Time (PIT) Tables
In Data Vault, querying multiple Satellites historically requires complex window joins (`_ldts <= query_date`). **Point-in-Time (PIT)** tables precompute the surrogate hash keys and exact satellite `_ldts` for regular snapshots (e.g. daily, hourly), turning multi-table historical scans into ultra-fast equi-joins.

**Schema**:
- `hk_customer`
- `as_of_date` (Snapshot timestamp)
- `sat_customer_core_ldts` (Exact matching `_ldts` in `sat_customer_core`)
- `sat_customer_profile_ldts` (Exact matching `_ldts` in `sat_customer_profile`)

**SQL Implementation**:
```sql
CREATE OR REFRESH MATERIALIZED VIEW biz_vault.pit_customer
CLUSTER BY (as_of_date, hk_customer)
AS
WITH snapshot_dates AS (
    SELECT explode(sequence(to_date('2026-01-01'), current_date(), interval 1 day)) AS as_of_date
),
pit_base AS (
    SELECT 
        h.hk_customer,
        d.as_of_date,
        MAX(s1._ldts) AS sat_customer_core_ldts,
        MAX(s2._ldts) AS sat_customer_profile_ldts
    FROM raw_vault.hub_customer h
    CROSS JOIN snapshot_dates d
    LEFT JOIN raw_vault.sat_customer_core s1 
        ON h.hk_customer = s1.hk_customer AND s1._ldts <= d.as_of_date
    LEFT JOIN raw_vault.sat_customer_profile s2 
        ON h.hk_customer = s2.hk_customer AND s2._ldts <= d.as_of_date
    GROUP BY ALL
)
SELECT * FROM pit_base;
```

### 4.2 Bridge Tables
A Bridge table pre-joins multiple Links and Hubs for a business domain (e.g. `Order -> Customer -> Product -> Store`), resolving all foreign hash keys into a single high-performance structure.

---

## 5. Standard Rules for Data Vault 2.0 on Databricks

1. **Insert-Only in Raw Vault**: Never execute `UPDATE` or `DELETE` on Raw Vault tables. History is strictly preserved.
2. **Standard Hash Delimiter**: Always use `^^` as the delimiter in `concat_ws` to prevent collision attacks when concatenating natural keys.
3. **Null Handling in Keys**: Always coalesce natural keys to a standard sentinel (e.g. `'-1'` or `'^^NULL^^'`) before hashing.
4. **Liquid Clustering**:
   - Cluster Hubs by `hk_<entity>`.
   - Cluster Links by `hk_<link>`, `hk_<entity1>`.
   - Cluster Satellites by `hk_<parent>`, `_ldts`.
   - Cluster PIT tables by `as_of_date`, `hk_<entity>`.
5. **Ghost Records**: Seed `-1` (all zeros or known hash) ghost records for missing/null references to allow outer joins without losing records.
6. **Parallel Ingestion**: Hubs and Satellites from the same source file must be loaded in parallel streams.
