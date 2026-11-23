---
name: functional-programming
description: "Authoritative skill for writing clean, testable, and maintainable Python and PySpark code using functional programming principles: pure functions, Column-first composition, .withColumns() projections, functools.reduce pipelines, and immutable configs."
compatibility: "Python (>= 3.10), PySpark (>= 3.4, >= 4.0)"
metadata:
  version: "2.0.0"
category: Python Engineering
tags:
  - functional-programming
  - pyspark
  - withColumns
  - transform
  - functools-reduce
  - pure-functions
  - immutability
---

# Functional Programming for Python & PySpark Data Engineering (v2.0)

This skill defines functional programming (FP) patterns for Python and PySpark Lakehouse engineering, grounded in **Mikołaj Sędek's Functional PySpark methodology** and the **Palantir PySpark Style Guide**.

---

## 1. Core Functional Principles in PySpark

1. **Pure Transformation Functions (`DataFrame -> DataFrame` and `Column -> Column`)**:
   - Every transformation function is a pure mathematical mapping from input to output with zero side effects, zero global state, and zero I/O calls.
2. **Atomic Responsibility (Micro-Transformations)**:
   - Break large transformations into small, composable functions where each function performs a **single logical operation** (e.g., `filter_valid_users`, `standardize_user_attributes`, `deduplicate_latest_records`).
   - No function should exceed 35–50 lines or chain more than 3–5 DataFrame operations.
3. **Batch Column Projections via `.withColumns()`**:
   - Never mutate a DataFrame reference repeatedly with `.withColumn()`. Construct a declarative dictionary `dict[str, Column]` and apply `.withColumns(col_dict)` in a single Catalyst projection.
4. **Declarative Composition via `.transform()` and `functools.reduce`**:
   - Chain atomic `DataFrame -> DataFrame` functions using `.transform()` or `functools.reduce` over a tuple of transformation functions.
5. **Immutable Domain Configurations & Schemas**:
   - Use `@dataclass(frozen=True)` or `NamedTuple` for all pipeline configurations, DQ rule summaries, and explicit `StructType` schema contracts with strict `nullable=False/True` definitions.

---

## 2. Two-Tier Functional Architecture: `Column` Functions + `DataFrame` Functions

### Tier 1: Pure `Column -> Column` Expression Builders
Encapsulate domain formulas, string normalizations, and conditional classifications as pure functions returning `pyspark.sql.Column`:

```python
from pyspark.sql import Column, functions as F

def trim_upper(col_name: str) -> Column:
    """Normalize a string column by trimming whitespace and converting to uppercase."""
    return F.upper(F.trim(F.col(col_name)))

def safe_ratio(numerator: Column, denominator: Column, scale: int = 4) -> Column:
    """Compute a division ratio safely guarding against zero or null denominators."""
    is_valid_denom = denominator.isNotNull() & (denominator > F.lit(0))
    return F.when(is_valid_denom, F.round(numerator / denominator, scale)).otherwise(F.lit(0.0))
```

### Tier 2: Atomic `DataFrame -> DataFrame` Step Functions
Use Tier 1 `Column` builders inside focused `DataFrame` step functions using `.withColumns()`:

```python
from functools import reduce
from typing import Callable
from pyspark.sql import DataFrame

TransformStep = Callable[[DataFrame], DataFrame]

def run_pipeline_steps(df: DataFrame, *steps: TransformStep) -> DataFrame:
    """Sequentially apply a pipeline of pure DataFrame transformation functions."""
    return reduce(lambda acc, step: acc.transform(step), steps, df)
```

---

## 3. Eliminating Loops over DataFrames

Never iterate over `df.columns` with a `for` loop that reassigns `df = df.withColumn(...)`. Instead:
- Use `df.withColumns({col_name: expr(col_name) for col_name in target_cols})` for adding/updating columns.
- Use `df.select(*[F.col(c).alias(c, metadata=...) for c in df.columns])` for single-pass schema/metadata projection.
- Use Spark SQL Higher-Order Functions (`F.transform`, `F.filter`, `F.exists`, `F.aggregate`) for array column manipulation.
