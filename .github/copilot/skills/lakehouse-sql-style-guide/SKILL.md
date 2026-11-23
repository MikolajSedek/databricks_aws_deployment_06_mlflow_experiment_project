---
name: lakehouse-sql-style-guide
description: "Authoritative skill and coding standards for writing readable, high-performance SQL for Lakehouse Analytics and Databricks SQL Warehouses, based on Mikołaj Sędek's Lakehouse SQL principles."
compatibility: "Databricks SQL Warehouses, Spark SQL, Delta Lake, Unity Catalog"
metadata:
  version: "1.0.0"
category: SQL & Analytics Engineering
tags:
  - sql
  - databricks-sql
  - lakehouse
  - performance-tuning
  - qualify
  - ctes
---

# Lakehouse SQL Style Guide & Query Optimization Standards

This skill provides definitive standards for writing maintainable, low-cognitive-load, and high-performance SQL queries for **Databricks SQL Warehouses**, **Lakeflow Pipelines**, and **Delta Lake Analytics**, authored by **Mikołaj Sędek**.

---

## 1. Eliminate Cognitive Complexity with CTEs

Deeply nested subqueries hide logic, complicate debugging, and hinder peer review. Use modular Common Table Expressions (`WITH ... AS (...)`) to tell a linear, step-by-step story:

```sql
-- BAD: Deeply nested subqueries
SELECT customer_id, total_spend
FROM (
    SELECT customer_id, SUM(amount) AS total_spend
    FROM (
        SELECT customer_id, amount
        FROM bronze.raw_orders
        WHERE status = 'COMPLETED'
    )
    GROUP BY customer_id
)
WHERE total_spend > 1000;

-- GOOD: Modular, self-documenting CTEs
WITH completed_orders AS (
    SELECT 
        customer_id, 
        amount
    FROM bronze.raw_orders
    WHERE status = 'COMPLETED'
),
customer_spending AS (
    SELECT 
        customer_id, 
        SUM(amount) AS total_spend
    FROM completed_orders
    GROUP BY customer_id
)
SELECT 
    customer_id, 
    total_spend
FROM customer_spending
WHERE total_spend > 1000;
```

---

## 2. Leverage the `QUALIFY` Clause

In standard SQL, filtering on the result of a window function (like `ROW_NUMBER()` or `RANK()`) forces an extra subquery or CTE. Modern Databricks SQL natively supports `QUALIFY`, filtering window calculations directly in the same statement:

```sql
-- BAD: Extra CTE required just to filter row_number
WITH ranked_events AS (
    SELECT 
        user_id,
        event_name,
        timestamp,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) AS rn
    FROM silver.events
)
SELECT user_id, event_name, timestamp
FROM ranked_events
WHERE rn = 1;

-- GOOD: Clean, idiomatic QUALIFY clause
SELECT 
    user_id,
    event_name,
    timestamp
FROM silver.events
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) = 1;
```

---

## 3. Avoid the Distributed Sort Shuffle Trap

In a single-node database (MySQL, PostgreSQL), `ORDER BY` is cheap. In distributed engines (Databricks, Apache Spark), `ORDER BY` forces a **global shuffle** that pulls all data across worker nodes onto a single executor or serialized shuffle stage, causing severe network congestion and disk spilling.

**Rules**:
1. **Never sort in intermediate CTEs or staging views**.
2. Only apply `ORDER BY` at the final presentation step if strictly necessary (e.g. `LIMIT 100`).
3. For window ordering, keep partition keys granular so sorting is localized to individual partitions.

---

## 4. Preserve File Pruning & Liquid Clustering

Applying functions or type conversions to clustered or partitioned columns in `WHERE` clauses disables data skipping and file pruning, forcing full-table scans.

```sql
-- BAD: Casting or function prevents file skipping on cluster key 'event_date'
SELECT * FROM gold.fct_events
WHERE CAST(event_date AS STRING) = '2026-11-03';

-- BAD: Date function on timestamp cluster key
SELECT * FROM gold.fct_events
WHERE to_date(event_timestamp) = '2026-11-03';

-- GOOD: Match native column data type directly
SELECT * FROM gold.fct_events
WHERE event_date = DATE '2026-11-03';

-- GOOD: Range query on native timestamp
SELECT * FROM gold.fct_events
WHERE event_timestamp >= TIMESTAMP '2026-11-03 00:00:00'
  AND event_timestamp <  TIMESTAMP '2026-11-04 00:00:00';
```

---

## 5. Modern SQL Expressions

### 5.1 Use `GROUP BY ALL`
Eliminate redundant, error-prone repetition of dimensional column lists:
```sql
-- GOOD: Automatically groups by all non-aggregate columns
SELECT 
    country,
    region,
    product_category,
    SUM(net_amount) AS total_revenue,
    COUNT(DISTINCT customer_id) AS unique_buyers
FROM gold.fct_orders
GROUP BY ALL;
```

### 5.2 Use `COALESCE` Over Verbose `CASE WHEN`
```sql
-- BAD
CASE WHEN email IS NOT NULL THEN email ELSE 'unknown@domain.com' END

-- GOOD
COALESCE(email, 'unknown@domain.com')
```

### 5.3 Beware Non-Equi Joins
Joins with inequality conditions (`<`, `>`, `!=`, `BETWEEN`) often trigger broadcast nested-loop joins or Cartesian explosions. Whenever possible, join on exact equality keys (e.g. surrogate keys, hash keys, date keys) and filter boundaries in `WHERE` or via Point-in-Time (PIT) structures.

---

## 6. Formatting & Style Rules

1. **Keywords in UPPERCASE**: `SELECT`, `FROM`, `JOIN`, `ON`, `WHERE`, `GROUP BY ALL`, `HAVING`, `QUALIFY`.
2. **Identifiers in snake_case**: Tables, columns, and aliases must use lowercase snake_case (`customer_id`, `fct_orders`).
3. **Leading Commas or Trailing Commas**: Be consistent across the project (trailing commas preferred with one column per line).
4. **Meaningful Aliases**: Avoid single-letter aliases (`a`, `b`, `c`). Use intuitive table abbreviations (`cust`, `ord`, `pit`).
