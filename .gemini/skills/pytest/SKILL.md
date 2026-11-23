---
name: pytest
description: "Run pytest, interpret failures, and write fast local unit tests for pure PySpark transformations and data generation without requiring a remote Databricks cluster."
compatibility: "pytest (>= 7.0), PySpark (>= 3.4)"
metadata:
  version: "1.0.0"
category: Testing & Quality Assurance
tags:
  - pytest
  - pyspark-testing
  - unit-testing
---

# PySpark Local Unit Testing with pytest

This skill provides patterns for testing Lakehouse transformations, Data Vault hash calculations, and data generation modules locally using `pytest` without connecting to a remote Databricks cluster.

---

## 1. Local SparkSession Fixture (`tests/conftest.py`)

A session-scoped local `SparkSession` minimizes JVM startup overhead by launching only once per test run:

```python
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("pytest-local-spark")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()
```

---

## 2. Testing Pure Transformations

```python
from pyspark.sql import Row
from src.python.modules.transformations import add_audit_metadata, compute_hash_key

def test_compute_hash_key(spark):
    data = [Row(customer_id="C100"), Row(customer_id="c100"), Row(customer_id=None)]
    df = spark.createDataFrame(data)
    
    result_df = compute_hash_key(df, key_col="customer_id", hash_col="hk_customer")
    rows = result_df.collect()
    
    # Assert case-insensitivity
    assert rows[0]["hk_customer"] == rows[1]["hk_customer"]
    # Assert null handling produces valid hash
    assert rows[2]["hk_customer"] is not None
```

---

## 3. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run fast pure-Python tests only (no Spark JVM)
pytest tests/ -m "not spark" -v

# Run with test coverage
pytest tests/ --cov=src --cov-report=term-missing
```
