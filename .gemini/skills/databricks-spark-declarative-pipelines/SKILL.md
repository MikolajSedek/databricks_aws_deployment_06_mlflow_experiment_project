---
name: databricks-spark-declarative-pipelines
description: "Authoritative skill and patterns for authoring, configuring, running, and testing Databricks Lakeflow Spark Declarative Pipelines (SDP/LDP) on serverless compute. Covers streaming tables, materialized views, Auto Loader ingestion from Volumes, CDC, SCD Type 1/2, expectations, and Liquid Clustering."
compatibility: "Databricks Runtime (DBR >= 15.0), Lakeflow Declarative Pipelines, Delta Lake, Unity Catalog"
metadata:
  version: "1.0.0"
category: Data Engineering
tags:
  - databricks
  - lakeflow
  - declarative-pipelines
  - streaming-tables
  - materialized-views
  - cdc
  - scd2
---

# Databricks Lakeflow Spark Declarative Pipelines (SDP)

This skill provides authoritative patterns, architectural best practices, and recipes for building **Lakeflow Declarative Pipelines** (formerly Delta Live Tables / DLT) using modern PySpark declarative APIs (`pyspark.pipelines`) and Databricks SQL on serverless compute.

---

## 1. Golden Rules of Lakeflow Pipelines

1. **Syntax: `CREATE OR REFRESH` (Never `CREATE OR REPLACE`)**:
   - Streaming Tables: `CREATE OR REFRESH STREAMING TABLE <name>`
   - Materialized Views: `CREATE OR REFRESH MATERIALIZED VIEW <name>`
   - `CREATE OR REPLACE` is standard interactive SQL and will fail in pipeline execution.
2. **Modern API: `pyspark.pipelines` (Never `import dlt`)**:
   - Always import: `from pyspark import pipelines as dp`
   - Use decorators: `@dp.table()`, `@dp.materialized_view()`, `@dp.temporary_view()`
   - Imperative CDC: `dp.create_streaming_table()`, `dp.create_auto_cdc_flow()`
3. **Data Ingestion: `read_files()` with `STREAM` Keyword**:
   - In SQL: `FROM STREAM read_files('/Volumes/catalog/schema/volume/path', format => 'json')`
   - In Python: `spark.readStream.format("cloudFiles").option("cloudFiles.format", "json").load("/Volumes/...")`
   - Always point to a folder path, never a single file.
4. **Liquid Clustering (Never Hive Partitioning + Z-Order)**:
   - Always specify `CLUSTER BY (col1, col2)` on tables.
   - Liquid clustering adapts automatically to ingestion skew and query filtering.
5. **Data Quality Expectations**:
   - Enforce governance at ingestion:
     `CONSTRAINT valid_key EXPECT (key IS NOT NULL) ON VIOLATION DROP ROW`
     `CONSTRAINT non_negative_price EXPECT (price >= 0) ON VIOLATION FAIL UPDATE`

---

## 2. Ingestion from Unity Catalog Managed Volumes

Reading sample events directly from a Unity Catalog managed volume using Auto Loader (`cloudFiles`):

### SQL Pattern:
```sql
CREATE OR REFRESH STREAMING TABLE bronze.raw_events
CLUSTER BY (event_date, event_type)
COMMENT "Raw event stream ingested from UC Managed Volume via Auto Loader"
AS
SELECT
    *,
    _metadata.file_path AS source_file_path,
    _metadata.file_name AS source_file_name,
    _metadata.file_modification_time AS source_file_time,
    current_timestamp() AS _ingested_at
FROM STREAM read_files(
    '/Volumes/lakehouse_dev/raw_data/event_landing/',
    format => 'json',
    schemaHints => 'event_id STRING, timestamp TIMESTAMP, user_id STRING'
);
```

### PySpark Pattern:
```python
from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    name="bronze.raw_events",
    comment="Raw event stream ingested from UC Managed Volume",
    cluster_by=["event_type", "event_date"]
)
def raw_events():
    volume_path = "/Volumes/lakehouse_dev/raw_data/event_landing/"
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaLocation", "/Volumes/lakehouse_dev/raw_data/checkpoints/schema_events")
        .load(volume_path)
        .select(
            "*",
            F.col("_metadata.file_path").alias("source_file_path"),
            F.col("_metadata.file_modification_time").alias("source_file_time"),
            F.current_timestamp().alias("_ingested_at")
        )
    )
```

---

## 3. Streaming Tables vs Materialized Views

| Feature | Streaming Table (`@dp.table`) | Materialized View (`@dp.materialized_view`) |
|---|---|---|
| **Underlying Engine** | Structured Streaming | Spark Batch Engine |
| **Data Processing** | Incremental / append-only | Full recalculation on each refresh |
| **Source Requirements** | Must read from a stream (`readStream`, `STREAM`) | Reads static tables, views, or files |
| **Aggregations** | Windowed aggregations with watermark (`window()`) | Any SQL aggregation (totals, group by, distinct) |
| **Best For** | Bronze raw ingestion, Silver CDC & cleaning | Gold summary statistics, KPI tables, PIT snapshots |

---

## 4. Change Data Capture (CDC) & Slowly Changing Dimensions

Lakeflow Pipelines provides built-in CDC engine via `dp.create_auto_cdc_flow` in Python and `AUTO CDC INTO` in SQL.

### 4.1 SCD Type 1 (Merge In-Place)
```python
dp.create_streaming_table(
    name="silver.dim_user_current",
    comment="SCD Type 1 current user state",
    cluster_by=["user_id"]
)

dp.create_auto_cdc_flow(
    source="bronze.stg_users",
    target="silver.dim_user_current",
    keys=["user_id"],
    sequence_by="updated_at",
    apply_as_deletes="operation = 'DELETE'"
)
```

### 4.2 SCD Type 2 (Historical Tracking)
Lakeflow automatically injects `__START_AT` and `__END_AT` columns:
```python
dp.create_streaming_table(
    name="gold.dim_user_scd2",
    comment="SCD Type 2 historical user dimension",
    cluster_by=["user_id"]
)

dp.create_auto_cdc_flow(
    source="silver.stg_users",
    target="gold.dim_user_scd2",
    keys=["user_id"],
    sequence_by="updated_at",
    apply_as_deletes="operation = 'DELETE'",
    track_history_column_list=["email", "status", "tier"]
)
```

---

## 5. Configuration & Multi-Environment Management

Access pipeline parameters configured in `databricks.yml` dynamically:
```python
catalog_name = spark.conf.get("pipeline.catalog", "lakehouse_dev")
volume_path  = spark.conf.get("pipeline.raw_volume_path", "/Volumes/lakehouse_dev/raw/events")

@dp.table(name=f"{catalog_name}.silver.cleaned_events")
def cleaned_events():
    ...
```

---

## 6. Local Testing Strategy

Transformations should be decoupled from the Lakeflow orchestration globals. Keep business logic in pure helper functions (`transformations.py`) and test them using `pytest` and a local `SparkSession` (`local[1]`):

```python
# Pure transformation function
def clean_events_df(df: DataFrame) -> DataFrame:
    return (
        df.filter(F.col("event_type").isNotNull())
        .withColumn("event_date", F.to_date("timestamp"))
    )

# Tested in pytest without a Databricks cluster
def test_clean_events_df(spark):
    sample_df = spark.createDataFrame([("1", "CLICK", "2026-11-03 10:00:00")], ["event_id", "event_type", "timestamp"])
    result_df = clean_events_df(sample_df)
    assert result_df.count() == 1
    assert "event_date" in result_df.columns
```
