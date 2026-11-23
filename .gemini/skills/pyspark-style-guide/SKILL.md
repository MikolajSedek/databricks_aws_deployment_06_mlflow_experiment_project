---
name: pyspark-style-guide
description: "Authoritative skill and coding standards for writing idiomatic, performant, and testable PySpark code on Databricks Lakehouse, combining the Palantir PySpark Style Guide and Mikołaj Sędek's Lakehouse PySpark engineering principles."
compatibility: "PySpark (>= 3.4, >= 4.0), Databricks Runtime, Lakeflow Declarative Pipelines"
metadata:
  version: "2.0.0"
category: Python & PySpark Engineering
tags:
  - pyspark
  - palantir-style-guide
  - mikolaj-sedek
  - withColumns
  - column-first-design
  - functional-programming
  - lakehouse
  - clean-code
---

# PySpark Lakehouse Style Guide & Engineering Standards (v2.0)

This skill is the authoritative engineering reference for authoring, reviewing, and refactoring PySpark code on the Databricks Lakehouse. It synthesizes the complete **Palantir Foundry PySpark Style Guide** (`https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-style-guide`) and **Mikołaj Sędek's Functional PySpark Architecture** (*"How to write really good PySpark code"*, `https://medium.com/@mikoajsdek/how-to-write-really-good-pyspark-code-ca2e8781952d`).

---

## 1. The Ten Commandments of Production PySpark

1. **Never Chain Multiple `.withColumn()` Calls — Always Use `.withColumns({...})`**:
   - Every `.withColumn()` call appends a new `Project` node to Spark's Catalyst unresolved logical plan. Chaining `.withColumn()` (or calling it inside a `for` loop) causes quadratic plan analysis overhead and driver `StackOverflowError`s.
   - Group all column derivations into a dictionary `dict[str, Column]` and apply them in a **single projection** via `df.withColumns(column_map)`.
2. **Operate on `Column` Objects First, Attach to `DataFrame` Last**:
   - Define `Column` expressions and pure helper functions (`Column -> Column`) first. Compose complex expressions at the `Column` level, and attach them to the `DataFrame` at the end via `.withColumns()` or `.select()`.
3. **Limit Method Chaining & Isolate Contexts (Max 3–5 Chained Calls)**:
   - Never mix heterogeneous operations (filtering + column creation + multi-table joins + dropping) in one giant 15-line method chain.
   - Keep chains to **at most 3–5 statements** of a single logical concern, or decompose into atomic pure helper functions composed via `.transform()` or `functools.reduce`.
4. **Explicit `StructType` Contracts & Explicit Nullability**:
   - Declare `StructField(name, DataType, nullable=False/True)` explicitly on all schemas:
     - Primary / business keys, surrogate hash keys (`hk_*`, `sk_*`), foreign keys, and mandatory audit timestamps (`_ldts`, `_rsrc`, `updated_at`) MUST be `nullable=False`.
     - Optional descriptive attributes or left-joined nullable metrics should explicitly declare `nullable=True`.
5. **Use `select()` as an Explicit Schema Contract**:
   - Place a clean `.select(...)` at the end of a transformation to explicitly define the output schema contract and column order rather than relying on sprawling `.drop()` or `.withColumnRenamed()` chains.
6. **Decompose Complex Logical Conditions (Max 3 Conditions Inline)**:
   - Never write deeply nested boolean expressions inside `.filter(...)` or `F.when(...)`. Extract sub-predicates into named `Column` variables or `Column -> Column` helper functions.
7. **Strict Join Discipline**:
   - Always specify `how="..."` explicitly (`how="inner"`, `how="left"`).
   - Never use `how="right"` — swap DataFrame order and use `how="left"`.
   - Trim/project right-side DataFrames before joining or use `.alias()` to prevent duplicate column collisions.
   - Never mask a join explosion with `.dropDuplicates()` or `.distinct()`.
8. **Explicit Window Frames & Null Ordering**:
   - Always specify explicit window ordering (`F.col(...).asc_nulls_last()` / `desc_nulls_last()`) and explicit frame bounds (`rowsBetween`) for rolling/cumulative windows. Never use an empty `Window.partitionBy()`.
9. **Zero Driver-Side Collects or Python UDFs in ETL**:
   - Never call `.collect()`, `.first()`, `.head()`, `.take()`, or `.show()` inside pipeline transformations.
   - Avoid Python UDFs in favor of native `pyspark.sql.functions` and Higher-Order Functions (`F.transform`, `F.filter`, `F.aggregate`).
10. **Strict File & Function Size Limits**:
    - Keep files under **250 lines** and individual functions under **70 lines** (ideally 15–35 lines). Separate Data Vault and Kimball DWH pipelines into dedicated modular files.

---

## 2. `.withColumns()` vs. Chained `.withColumn()` (Mikołaj Sędek Pattern)

### Anti-Pattern (BAD — Multiple Catalyst `Project` Nodes & Loops)
```python
# BAD: 6 separate Catalyst Project nodes!
df = (
    df.withColumn("user_id", F.upper(F.trim(F.col("user_id"))))
    .withColumn("email", F.lower(F.trim(F.col("email"))))
    .withColumn("first_name", F.initcap(F.trim(F.col("first_name"))))
    .withColumn("last_name", F.initcap(F.trim(F.col("last_name"))))
    .withColumn("country_code", F.upper(F.trim(F.col("country_code"))))
    .withColumn("loyalty_tier", F.upper(F.trim(F.col("loyalty_tier"))))
)

# BAD: .withColumn inside a for-loop!
for col_name in df.columns:
    df = df.withColumn(col_name, F.col(col_name).alias(col_name, metadata={"comment": desc}))
```

### Best Practice (GOOD — Single `.withColumns()` Dictionary Projection)
```python
from pyspark.sql import Column, DataFrame, functions as F

def norm_upper_col(col_name: str) -> Column:
    """Return a trimmed, uppercase Column expression."""
    return F.upper(F.trim(F.col(col_name)))

def norm_lower_col(col_name: str) -> Column:
    """Return a trimmed, lowercase Column expression."""
    return F.lower(F.trim(F.col(col_name)))

def norm_title_col(col_name: str) -> Column:
    """Return a trimmed, title-case Column expression."""
    return F.initcap(F.trim(F.col(col_name)))

def standardize_user_columns(df: DataFrame) -> DataFrame:
    """Normalize all user string attributes in a single Catalyst projection."""
    return df.withColumns(
        {
            "user_id": norm_upper_col("user_id"),
            "email": norm_lower_col("email"),
            "first_name": norm_title_col("first_name"),
            "last_name": norm_title_col("last_name"),
            "country_code": norm_upper_col("country_code"),
            "loyalty_tier": norm_upper_col("loyalty_tier"),
        }
    )

def attach_column_descriptions(df: DataFrame, descriptions: dict[str, str]) -> DataFrame:
    """Attach Unity Catalog comments to all columns in a single .select() projection."""
    return df.select(
        *[
            F.col(c).alias(c, metadata={"comment": descriptions.get(c, f"Governed attribute {c}.")})
            for c in df.columns
        ]
    )
```

---

## 3. Operate on `Column` Objects, Not `DataFrame` Objects (Palantir Pattern)

### 3.1 Write Helper Functions That Accept & Return `Column` Objects
Instead of writing helpers that mutate a `DataFrame` one column at a time, write composable `Column -> Column` builders:

```python
from pyspark.sql import Column, functions as F

def calc_net_amount(gross_col: Column, discount_col: Column, tax_col: Column) -> Column:
    """Compute net amount as (gross - discount + tax) rounded to 2 decimals."""
    return F.round(gross_col - discount_col + tax_col, 2)

def calc_discount_pct(gross_col: Column, discount_col: Column) -> Column:
    """Compute percentage discount relative to gross amount."""
    has_positive_gross = gross_col > F.lit(0.0)
    return F.when(has_positive_gross, F.round((discount_col / gross_col) * F.lit(100.0), 2)).otherwise(F.lit(0.0))

def classify_order_tier(net_col: Column) -> Column:
    """Classify order value tier from net revenue."""
    is_high = net_col >= F.lit(250.0)
    is_medium = net_col >= F.lit(100.0)
    return (
        F.when(is_high, F.lit("HIGH_VALUE"))
        .when(is_medium, F.lit("MEDIUM_VALUE"))
        .otherwise(F.lit("STANDARD_VALUE"))
    )
```

### 3.2 Delay Attaching Columns & Never Shadow Existing Column Names Unintentionally
Build `Column` expressions as local variables and attach them once via `.withColumns({...})`:

```python
def enrich_order_metrics(df: DataFrame) -> DataFrame:
    """Attach derived financial metrics to orders in a single projection."""
    net_amt = calc_net_amount(F.col("gross_amount"), F.col("discount_amount"), F.col("tax_amount"))
    return df.withColumns(
        {
            "net_amount": net_amt,
            "order_date": F.to_date(F.col("order_timestamp")),
        }
    )
```

---

## 4. Pipeline Composition with `.transform()` and `functools.reduce`

Decompose multi-step transformations into small, single-purpose functions (each doing **one** thing: filtering, normalizing columns, deduplicating, or projecting) and compose them declaratively:

```python
from functools import reduce
from typing import Callable
from pyspark.sql import DataFrame

TransformFn = Callable[[DataFrame], DataFrame]

def compose_transforms(df: DataFrame, steps: tuple[TransformFn, ...]) -> DataFrame:
    """Apply an ordered tuple of pure DataFrame transformations using functools.reduce."""
    return reduce(lambda acc_df, fn: acc_df.transform(fn), steps, df)

def clean_users_df(df: DataFrame) -> DataFrame:
    """Execute the complete user cleaning pipeline via composable atomic steps."""
    return (
        df.transform(filter_non_empty_user_keys)
        .transform(standardize_user_columns)
        .transform(deduplicate_users_by_latest_update)
        .transform(attach_column_descriptions)
    )
```

---

## 5. Refactoring Complex Logical Predicates

Never inline more than 3 boolean conditions inside `.filter()` or `F.when()`. Extract named predicate `Column` variables first:

```python
# BAD: Unreadable nested boolean predicate
df = df.filter(
    F.col("order_id").isNotNull()
    & F.col("user_id").isNotNull()
    & F.col("product_id").isNotNull()
    & (F.col("quantity") > 0)
    & (F.col("gross_amount") >= 0.0)
)

# GOOD: Named Column predicates combined cleanly
def filter_valid_order_rows(df: DataFrame) -> DataFrame:
    """Retain orders with non-null business keys and valid positive quantities/amounts."""
    has_required_keys = (
        F.col("order_id").isNotNull()
        & F.col("user_id").isNotNull()
        & F.col("product_id").isNotNull()
    )
    has_valid_measures = (F.col("quantity") > 0) & (F.col("gross_amount") >= 0.0)
    return df.filter(has_required_keys & has_valid_measures)
```

---

## 6. Explicit Schema Contracts & `StructType` Nullability

1. **Mandatory vs. Nullable Fields in `StructType`**:
   - Primary keys, foreign keys, timestamps, and core measures must specify `nullable=False`.
   - Optional attributes (e.g., `subcategory`, `page_url`) may specify `nullable=True`.
2. **Typed Null Placeholders**:
   - When creating an empty column to satisfy a schema contract, always use `F.lit(None).cast("string")` (or appropriate type), never `F.lit("")` or `F.lit("NA")`.
3. **Final `select()` Schema Contract**:
   - Place simple column names first, followed by aliased or cast expressions on separate lines. Never embed `.when()` conditionals inside `.select()` — compute them in `.withColumns()` first.

---

## 7. Code Review Checklist (Validation Gate)

Before approving any PySpark module, verify:
- [ ] **Zero chained `.withColumn()` calls**: Are all multi-column additions/modifications wrapped in `.withColumns({...})`?
- [ ] **Zero `for` loops mutating a DataFrame**: Are column loops replaced with `.select(*[...])` list comprehensions or `.withColumns({...})` dictionary comprehensions?
- [ ] **Short transformation chains**: Is every method chain $\le 5$ operations, separated by logical concern (filter vs. compute vs. join vs. select)?
- [ ] **`Column`-level helper functions**: Are reusable calculations written as `Column -> Column` functions rather than `DataFrame -> DataFrame` wrappers?
- [ ] **Named boolean predicates**: Are complex `.filter()` and `F.when()` expressions factored into named `Column` variables?
- [ ] **Explicit `how=` on all `.join()` calls**: Are all joins using explicit `how="inner"` or `how="left"`, with zero right joins?
- [ ] **File & function size limits**: Is every file $< 250$ lines and every function $< 70$ lines?
