---
name: kimball-dimensional-modeling
description: "Authoritative skill and patterns for architecting, designing, and implementing Kimball Dimensional Data Warehouses (Star Schemas, SCD Type 1/2, Fact Tables, Conformed Dimensions) on Databricks Lakehouse for BI, SQL Analytics, and ML Feature Stores."
compatibility: "Databricks Runtime (DBR >= 15.0), Lakeflow Declarative Pipelines, Databricks SQL Warehouses, Unity Catalog"
metadata:
  version: "1.0.0"
category: Data Architecture & Modeling
tags:
  - kimball
  - dimensional-modeling
  - star-schema
  - fact-tables
  - scd-type-2
  - databricks
  - lakehouse
---

# Kimball Dimensional Data Modeling on the Databricks Lakehouse

This skill provides comprehensive architectural patterns, production recipes, and quality guardrails for building **Kimball Dimensional Models (Star Schemas)** on the **Databricks Lakehouse Platform** on top of Data Vault 2.0 or cleansed staging tables.

---

## 1. Architectural Role in the Lakehouse

In the Medallion Architecture, Kimball Dimensional Models constitute the **Gold Layer**:

```
┌─────────────────────────────────┐
│   SILVER LAYER (DATA VAULT)     │
│   • Raw Vault: Hubs, Links, Sats│
│   • Biz Vault: PIT, Bridges, BV │
└────────────────┬────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        GOLD LAYER (KIMBALL DWH)                        │
│                                                                        │
│   ┌─────────────────────┐               ┌──────────────────────────┐   │
│   │  CONFORMED DIMENSIONS│               │       FACT TABLES        │   │
│   │  • dim_date         │               │  • fct_events (Trans)    │   │
│   │  • dim_customer     │──────────────▶│  • fct_daily_snapshot    │   │
│   │  • dim_product      │               │  • fct_order_lifecycle   │   │
│   │  (SCD Type 1 & 2)   │               │  Declared explicit grain │   │
│   └─────────────────────┘               └──────────────────────────┘   │
└────────────────┬───────────────────────────────────────────────────────┘
                 │
                 ├───────────────────────────────┐
                 ▼                               ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│         DATA ANALYSTS           │   │          ML ENGINEERS            │
│  • Databricks SQL Warehouses    │   │  • Unity Catalog Feature Store   │
│  • PowerBI / Tableau Dashboards │   │  • Point-in-time train sets      │
│  • Genie Spaces (Semantic Layer)│   │  • Low-latency online inference  │
└─────────────────────────────────┘   └──────────────────────────────────┘
```

---

## 2. The 4-Step Kimball Design Process

1. **Select the Business Process**: Identify the operational event or activity being measured (e.g. order placement, payment transaction, user click event, web session).
2. **Declare the Grain**: State explicitly what a single row in the fact table represents. **Never proceed without a written grain statement**:
   - *"One row represents one customer order line item."*
   - *"One row represents one clickstream event captured from a client application."*
   - *"One row represents one product's inventory balance at midnight UTC."*
3. **Identify the Dimensions**: Choose the descriptive context answering *who, what, where, when, why, and how* (e.g. `dim_date`, `dim_customer`, `dim_product`, `dim_device`).
4. **Identify the Numeric Facts**: Determine the additive, semi-additive, or non-additive measurements taken at the declared grain (e.g. `quantity`, `unit_price`, `discount_amt`, `net_revenue`).

---

## 3. Fact Table Patterns

### 3.1 Transaction Fact Tables
Capture an event that occurs at a specific point in time. Records are append-only.
- **Grain**: One row per discrete event.
- **Measures**: Fully additive across all dimensions (sum, avg, count).
- **Naming**: `fct_<event_name>` (e.g. `fct_orders`, `fct_transactions`).

```sql
CREATE OR REFRESH MATERIALIZED VIEW gold.fct_orders
CLUSTER BY (order_date_key, customer_key)
AS
SELECT
    d.date_key AS order_date_key,
    c.customer_key,
    p.product_key,
    o.order_id,
    o.quantity,
    o.unit_price,
    o.discount_amount,
    (o.quantity * o.unit_price) - o.discount_amount AS net_amount,
    current_timestamp() AS _created_at
FROM biz_vault.pit_customer pit
JOIN raw_vault.sat_orders o ON pit.hk_order = o.hk_order
JOIN gold.dim_customer c ON c.hk_customer = pit.hk_customer AND c.is_current = TRUE
JOIN gold.dim_product p ON p.hk_product = o.hk_product AND p.is_current = TRUE
JOIN gold.dim_date d ON d.calendar_date = to_date(o.order_timestamp);
```

### 3.2 Periodic Snapshot Fact Tables
Capture the cumulative state of a business process at predetermined intervals (daily, weekly, monthly).
- **Grain**: One row per period per entity (e.g. one row per account per month).
- **Measures**: Semi-additive (can be summed across customers, but NOT across time — use average or closing balance across time).
- **Naming**: `fct_monthly_account_balance`, `fct_daily_inventory_snapshot`.

### 3.3 Accumulating Snapshot Fact Tables
Track the progress of a predictable multi-step lifecycle (e.g. order created → payment approved → item picked → shipped → delivered).
- **Grain**: One row per entire lifecycle instance (one row per order).
- **Dates**: Multiple foreign keys to `dim_date` (e.g. `order_date_key`, `ship_date_key`, `delivery_date_key`).
- **Lag Metrics**: Computed durations between stages (e.g. `days_to_ship`, `days_to_deliver`).

### 3.4 Factless Fact Tables
Record events where no numeric measure exists, or define valid coverage relationships.
- **Examples**: Student attendance (`student_key`, `course_key`, `date_key`), marketing campaign eligibility.

---

## 4. Dimension Table Patterns & SCD Management

### 4.1 Surrogate Keys vs Business Keys
- Every dimension table MUST have an integer or SHA surrogate primary key (`customer_key`, `product_key`).
- Natural business keys (`customer_id`, `sku`) must be retained as separate attributes.
- Use `row_number()` or MD5/SHA-256 for deterministic surrogate keys in distributed environments.

### 4.2 Slowly Changing Dimensions (SCD)

#### SCD Type 1 (Overwrite In-Place)
Used when historical tracking is unnecessary or for correcting typographical errors.
```python
# SCD Type 1 in Lakeflow Pipelines
dp.create_auto_cdc_flow(
    source="silver.stg_customers",
    target="gold.dim_customer_scd1",
    keys=["customer_id"],
    sequence_by="updated_at"
)
```

#### SCD Type 2 (Historical Tracking with Temporal Validity)
Used when history must be preserved for accurate historical reporting.
Databricks Lakeflow Declarative Pipelines generates temporal validity columns natively:
- `__START_AT`: Timestamp when the record version became effective.
- `__END_AT`: Timestamp when the record version expired (`NULL` for current active row).

**Lakeflow PySpark Implementation**:
```python
from pyspark import pipelines as dp

dp.create_streaming_table(
    name="gold.dim_customer_scd2",
    comment="Customer Dimension with SCD Type 2 historical tracking",
    cluster_by=["customer_key", "is_current"]
)

dp.create_auto_cdc_flow(
    source="silver.stg_customers",
    target="gold.dim_customer_scd2",
    keys=["customer_id"],
    sequence_by="updated_at",
    track_history_column_list=["segment", "region", "status"],
    track_history_except_column_list=["phone_number", "last_login"]
)
```

**Querying SCD Type 2 in SQL**:
```sql
-- Querying current state only
SELECT * FROM gold.dim_customer_scd2
WHERE __END_AT IS NULL;

-- Historical point-in-time join (as of transaction date)
SELECT 
    f.order_id,
    f.net_amount,
    c.segment,
    c.region
FROM gold.fct_orders f
JOIN gold.dim_customer_scd2 c 
    ON f.customer_id = c.customer_id
    AND f.order_timestamp >= c.__START_AT 
    AND (f.order_timestamp < c.__END_AT OR c.__END_AT IS NULL);
```

---

## 5. Tailoring for ML Engineers & Feature Stores

1. **Point-in-Time Correctness**:
   - Dimensional tables with SCD Type 2 (`__START_AT`, `__END_AT`) prevent **data leakage** when constructing offline ML training sets.
   - Training features are joined as of the observation event timestamp: `event_time >= __START_AT AND event_time < __END_AT`.
2. **Unity Catalog Feature Store Registration**:
   - Dimension and snapshot tables in Gold can be registered directly as UC Feature Tables using Databricks Feature Engineering client (`FeatureEngineeringClient()`).
3. **Pre-aggregated Feature Tables**:
   - Create accumulating metrics (e.g. `user_orders_last_30d`, `avg_spend_last_90d`) as gold materialized views with Liquid Clustering on the entity key.

---

## 6. Dimensional Modeling Quality Checklist

- [ ] Every fact table has an explicitly declared, documented grain in table comments.
- [ ] Foreign keys in fact tables connect to dimension surrogate keys.
- [ ] No `NULL` values in fact foreign keys (use `-1` Unknown / Ghost records).
- [ ] Additive measures are cleanly separated from non-additive rates or ratios (store numerator and denominator, compute ratio in view).
- [ ] Conformed dimensions are reused across all fact tables in the enterprise data warehouse.
- [ ] Star schema design is maintained (avoid excessive snowflaking to minimize join penalties in distributed execution).
- [ ] Tables use Liquid Clustering (`CLUSTER BY`) on join keys and primary temporal filters.
