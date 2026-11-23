"""Pure functional PySpark feature engineering for Customer Churn and Product Affinity XGBoost models.

Follows Palantir PySpark Style Guide and Mikołaj Sędek's Functional PySpark standards:
- Tier 1 `Column -> Column` expression builders
- Single-projection `.withColumns({...})` dictionaries (zero chained `.withColumn()`)
- Composable `DataFrame -> DataFrame` steps via `.transform()`
- Explicit `.select()` schema contracts, label-free Feature Store tables, dedicated label spines,
  point-in-time `snapshot_date` timeseries keys, and Unity Catalog metadata attachment
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    COL_PRODUCT_ID,
    COL_SK_PRODUCT,
    COL_SK_USER,
    COL_SNAPSHOT_DATE,
    COL_USER_ID,
    TARGET_AFFINITY_COL,
    TARGET_CHURN_COL,
)
from src.python.modules.utils.metadata import attach_column_descriptions

# =============================================================================
# Tier 1: Pure Column-Level Feature Builders (Column -> Column)
# =============================================================================


def calc_safe_ratio(numerator: Column, denominator: Column, scale: int = 4) -> Column:
    """Compute a bounded division ratio safely guarding against zero or null denominators."""
    has_positive_denom = denominator.isNotNull() & (denominator > F.lit(0))
    return F.when(has_positive_denom, F.round(numerator / denominator, scale)).otherwise(F.lit(0.0))


def encode_loyalty_tier(tier_col: Column) -> Column:
    """Encode customer `loyalty_tier` string into an ordinal integer code (1..4)."""
    normalized = F.upper(F.trim(F.coalesce(tier_col, F.lit("BRONZE"))))
    is_platinum = normalized == F.lit("PLATINUM")
    is_gold = normalized == F.lit("GOLD")
    is_silver = normalized == F.lit("SILVER")
    return (
        F.when(is_platinum, F.lit(4)).when(is_gold, F.lit(3)).when(is_silver, F.lit(2)).otherwise(F.lit(1)).cast("int")
    )


def encode_customer_segment(segment_col: Column) -> Column:
    """Encode `calculated_customer_segment` string into an ordinal integer code (1..3)."""
    normalized = F.upper(F.trim(F.coalesce(segment_col, F.lit("NEW_OR_LOW_ACTIVITY"))))
    is_vip = normalized.isin("VIP_REPEAT", "HIGH_VALUE", "VIP")
    is_core = normalized.isin("CORE_ACTIVE", "ACTIVE", "REGULAR")
    return F.when(is_vip, F.lit(3)).when(is_core, F.lit(2)).otherwise(F.lit(1)).cast("int")


def encode_product_category(category_col: Column) -> Column:
    """Encode product `category` string into a deterministic ordinal integer code (1..6)."""
    normalized = F.upper(F.trim(F.coalesce(category_col, F.lit("OTHER"))))
    return (
        F.when(normalized == F.lit("ELECTRONICS"), F.lit(1))
        .when(normalized == F.lit("APPAREL"), F.lit(2))
        .when(normalized.isin("HOME", "HOME_GARDEN"), F.lit(3))
        .when(normalized == F.lit("SPORTS"), F.lit(4))
        .when(normalized == F.lit("BOOKS"), F.lit(5))
        .otherwise(F.lit(6))
        .cast("int")
    )


def encode_hash_bucket(text_col: Column, num_buckets: int = 16) -> Column:
    """Map a categorical string column to a deterministic non-negative modulo hash bucket."""
    normalized = F.upper(F.trim(F.coalesce(text_col, F.lit("UNKNOWN"))))
    return F.pmod(F.abs(F.xxhash64(normalized)), F.lit(num_buckets)).cast("int")


def _resolve_snapshot_date_col(df: DataFrame, snapshot_date: date | str | None = None) -> Column:
    """Resolve the point-in-time `snapshot_date` column (`DATE`), preserving existing column when present."""
    if snapshot_date is not None:
        return F.to_date(F.lit(str(snapshot_date)))
    if COL_SNAPSHOT_DATE in df.columns:
        return F.coalesce(F.to_date(F.col(COL_SNAPSHOT_DATE)), F.current_date())
    return F.current_date()


# =============================================================================
# Tier 2: Atomic DataFrame Feature & Label Transformations (DataFrame -> DataFrame)
# =============================================================================


def filter_valid_churn_rows(df: DataFrame) -> DataFrame:
    """Retain customer churn records with non-null, non-empty `user_id` primary keys."""
    has_key = F.col(COL_USER_ID).isNotNull() & (F.length(F.trim(F.col(COL_USER_ID))) > F.lit(0))
    return df.filter(has_key)


def derive_churn_engineered_columns(df: DataFrame, snapshot_date: date | str | None = None) -> DataFrame:
    """Derive encoded categoricals, behavioral ratios, and `snapshot_date` in a single `.withColumns()` call."""
    return df.withColumns(
        {
            COL_SNAPSHOT_DATE: _resolve_snapshot_date_col(df, snapshot_date),
            "loyalty_tier_code": encode_loyalty_tier(F.col("loyalty_tier")),
            "customer_segment_code": encode_customer_segment(F.col("calculated_customer_segment")),
            "country_bucket_code": encode_hash_bucket(F.col("country_code"), num_buckets=16),
            "revenue_per_event_usd": calc_safe_ratio(
                F.col("total_net_revenue_usd").cast("double"),
                F.col("total_events_count").cast("double"),
                scale=2,
            ),
            "checkout_completion_ratio": calc_safe_ratio(
                F.col("purchase_event_count").cast("double"),
                F.col("checkout_start_count").cast("double"),
            ),
        }
    )


def engineer_customer_churn_features(
    raw_churn_view_df: DataFrame,
    snapshot_date: date | str | None = None,
) -> DataFrame:
    """Transform `vw_ml_customer_churn_features` into a label-free, leak-free UC Feature Table."""
    selected_cols = [COL_USER_ID, COL_SK_USER, COL_SNAPSHOT_DATE, *CHURN_FEATURE_COLUMNS]
    return (
        raw_churn_view_df.transform(filter_valid_churn_rows)
        .transform(lambda d: derive_churn_engineered_columns(d, snapshot_date=snapshot_date))
        .select(*selected_cols)
        .transform(attach_column_descriptions)
    )


def extract_customer_churn_labels(
    raw_churn_view_df: DataFrame,
    snapshot_date: date | str | None = None,
) -> DataFrame:
    """Extract the dedicated Customer Churn label spine (`user_id`, `sk_user`, `snapshot_date`, `is_churn_candidate`)."""
    return (
        raw_churn_view_df.transform(filter_valid_churn_rows)
        .filter(F.col(TARGET_CHURN_COL).isNotNull())
        .withColumns(
            {
                COL_SNAPSHOT_DATE: _resolve_snapshot_date_col(raw_churn_view_df, snapshot_date),
                TARGET_CHURN_COL: F.col(TARGET_CHURN_COL).cast("int"),
            }
        )
        .select(COL_USER_ID, COL_SK_USER, COL_SNAPSHOT_DATE, TARGET_CHURN_COL)
        .transform(attach_column_descriptions)
    )


def filter_valid_affinity_rows(df: DataFrame) -> DataFrame:
    """Retain product affinity records with non-null, non-empty `product_id` primary keys."""
    has_key = F.col(COL_PRODUCT_ID).isNotNull() & (F.length(F.trim(F.col(COL_PRODUCT_ID))) > F.lit(0))
    return df.filter(has_key)


def derive_affinity_engineered_columns(df: DataFrame, snapshot_date: date | str | None = None) -> DataFrame:
    """Derive product category codes, unit/margin ratios, and `snapshot_date` in a single `.withColumns()` call."""
    margin_per_unit = F.round(
        F.col("unit_price").cast("double") * (F.col("gross_margin_pct").cast("double") / F.lit(100.0)),
        2,
    )
    return df.withColumns(
        {
            COL_SNAPSHOT_DATE: _resolve_snapshot_date_col(df, snapshot_date),
            "category_code": encode_product_category(F.col("category")),
            "subcategory_bucket_code": encode_hash_bucket(F.col("subcategory"), num_buckets=32),
            "units_per_order_ratio": calc_safe_ratio(
                F.col("total_units_sold").cast("double"),
                F.col("total_orders").cast("double"),
            ),
            "revenue_per_buyer_usd": calc_safe_ratio(
                F.col("total_net_revenue_usd").cast("double"),
                F.col("distinct_buyers_count").cast("double"),
                scale=2,
            ),
            "estimated_margin_per_unit_usd": margin_per_unit,
        }
    )


def engineer_product_affinity_features(
    raw_affinity_view_df: DataFrame,
    snapshot_date: date | str | None = None,
) -> DataFrame:
    """Transform `vw_ml_product_affinity_features` into a label-free, leak-free UC Feature Table."""
    selected_cols = [COL_PRODUCT_ID, COL_SK_PRODUCT, COL_SNAPSHOT_DATE, *AFFINITY_FEATURE_COLUMNS]
    return (
        raw_affinity_view_df.transform(filter_valid_affinity_rows)
        .transform(lambda d: derive_affinity_engineered_columns(d, snapshot_date=snapshot_date))
        .select(*selected_cols)
        .transform(attach_column_descriptions)
    )


def extract_product_affinity_labels(
    raw_affinity_view_df: DataFrame,
    snapshot_date: date | str | None = None,
) -> DataFrame:
    """Extract the dedicated Product Affinity label spine (`product_id`, `sk_product`, `snapshot_date`, target)."""
    return (
        raw_affinity_view_df.transform(filter_valid_affinity_rows)
        .filter(F.col(TARGET_AFFINITY_COL).isNotNull())
        .withColumns(
            {
                COL_SNAPSHOT_DATE: _resolve_snapshot_date_col(raw_affinity_view_df, snapshot_date),
                TARGET_AFFINITY_COL: F.col(TARGET_AFFINITY_COL).cast("double"),
            }
        )
        .select(COL_PRODUCT_ID, COL_SK_PRODUCT, COL_SNAPSHOT_DATE, TARGET_AFFINITY_COL)
        .transform(attach_column_descriptions)
    )


def sync_uc_feature_table(
    spark: SparkSession,
    features_df: DataFrame,
    table_fqn: str,
    primary_key: str,
    description: str,
    fe_client: Any | None = None,
    timeseries_columns: list[str] | None = None,
) -> str:
    """Create or merge a Unity Catalog Feature Table via `FeatureEngineeringClient` with Delta fallback."""
    if fe_client is not None:
        if not spark.catalog.tableExists(table_fqn):
            create_kwargs: dict[str, Any] = {
                "name": table_fqn,
                "primary_keys": [primary_key, *(timeseries_columns or [])],
                "df": features_df,
                "description": description,
            }
            if timeseries_columns:
                create_kwargs["timeseries_columns"] = timeseries_columns
            fe_client.create_table(**create_kwargs)
            return "created_uc_feature_table"
        fe_client.write_table(
            name=table_fqn,
            df=features_df,
            mode="merge",
        )
        return "merged_uc_feature_table"

    features_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_fqn)
    return "saved_delta_table"


def sync_label_table(
    labels_df: DataFrame,
    table_fqn: str,
) -> str:
    """Persist a dedicated entity label spine table separately from the UC Feature Table."""
    labels_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_fqn)
    return "saved_label_table"
