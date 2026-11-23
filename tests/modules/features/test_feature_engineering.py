"""Unit tests for pure PySpark feature engineering, label spine extraction, and Feature Table syncing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession
from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
    extract_customer_churn_labels,
    extract_product_affinity_labels,
    sync_label_table,
    sync_uc_feature_table,
)
from src.python.modules.features.synthetic_gold_views import (
    GOLD_AFFINITY_VIEW_SCHEMA,
    GOLD_CHURN_VIEW_SCHEMA,
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)
from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    COL_PRODUCT_ID,
    COL_SK_PRODUCT,
    COL_SK_USER,
    COL_SNAPSHOT_DATE,
    COL_USER_ID,
    LEAKY_AFFINITY_COLUMNS,
    LEAKY_CHURN_COLUMNS,
    TARGET_AFFINITY_COL,
    TARGET_CHURN_COL,
)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
def test_engineer_customer_churn_features_and_labels_contract(spark: SparkSession) -> None:
    """Verify label-free Customer Churn Feature Table contract, dedicated label spine, and ratios."""
    gold_df = build_sample_gold_churn_view_df(spark, num_users=30)
    assert gold_df.schema["sk_user"].nullable is False
    assert gold_df.schema["user_id"].nullable is False
    assert gold_df.schema["snapshot_date"].nullable is False
    assert gold_df.schema == GOLD_CHURN_VIEW_SCHEMA

    fe_df = gold_df.transform(engineer_customer_churn_features)
    expected_cols = [COL_USER_ID, COL_SK_USER, COL_SNAPSHOT_DATE, *CHURN_FEATURE_COLUMNS]
    assert fe_df.columns == expected_cols
    assert TARGET_CHURN_COL not in fe_df.columns
    assert "churn_risk_score" not in fe_df.columns
    assert LEAKY_CHURN_COLUMNS.isdisjoint(fe_df.columns)

    rows = fe_df.collect()
    assert len(rows) == 30
    for row in rows:
        assert row["snapshot_date"] is not None
        assert row["loyalty_tier_code"] in (1, 2, 3, 4)
        assert row["customer_segment_code"] in (1, 2, 3)
        assert 0 <= row["country_bucket_code"] < 16
        assert row["revenue_per_event_usd"] >= 0.0
        assert row["checkout_completion_ratio"] >= 0.0

    labels_df = gold_df.transform(extract_customer_churn_labels)
    assert labels_df.columns == [COL_USER_ID, COL_SK_USER, COL_SNAPSHOT_DATE, TARGET_CHURN_COL]
    label_rows = labels_df.collect()
    assert len(label_rows) == 30
    assert all(r[TARGET_CHURN_COL] in (0, 1) for r in label_rows)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize(
    ("snapshot_override", "expected_exact_date"),
    [
        (None, None),
        ("2026-03-15", "2026-03-15"),
    ],
    ids=["fallback_to_current_date", "explicit_snapshot_override"],
)
def test_churn_snapshot_date_resolution_without_upstream_column(
    spark: SparkSession,
    snapshot_override: str | None,
    expected_exact_date: str | None,
) -> None:
    """Verify upstream Gold views lacking `snapshot_date` fall back to `current_date()` or explicit override."""
    no_date_df = build_sample_gold_churn_view_df(spark, num_users=10).drop(COL_SNAPSHOT_DATE)
    fe_df = engineer_customer_churn_features(no_date_df, snapshot_date=snapshot_override)
    resolved_dates = [r[COL_SNAPSHOT_DATE] for r in fe_df.collect()]
    assert all(d is not None for d in resolved_dates)
    if expected_exact_date is not None:
        assert {str(d) for d in resolved_dates} == {expected_exact_date}


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
def test_engineer_product_affinity_features_and_labels_contract(spark: SparkSession) -> None:
    """Verify label-free Product Affinity Feature Table contract, HOME_GARDEN encoding, and label spine."""
    gold_df = build_sample_gold_affinity_view_df(spark, num_products=24)
    assert gold_df.schema["sk_product"].nullable is False
    assert gold_df.schema["product_id"].nullable is False
    assert gold_df.schema["snapshot_date"].nullable is False
    assert gold_df.schema == GOLD_AFFINITY_VIEW_SCHEMA

    fe_df = gold_df.transform(engineer_product_affinity_features)
    expected_cols = [COL_PRODUCT_ID, COL_SK_PRODUCT, COL_SNAPSHOT_DATE, *AFFINITY_FEATURE_COLUMNS]
    assert fe_df.columns == expected_cols
    assert TARGET_AFFINITY_COL not in fe_df.columns
    assert LEAKY_AFFINITY_COLUMNS.isdisjoint(fe_df.columns)

    rows = fe_df.collect()
    assert len(rows) == 24
    for row in rows:
        assert row["snapshot_date"] is not None
        assert row["category_code"] in (1, 2, 3, 4, 5, 6)
        assert 0 <= row["subcategory_bucket_code"] < 32
        assert row["units_per_order_ratio"] >= 0.0
        assert row["revenue_per_buyer_usd"] >= 0.0
        assert row["estimated_margin_per_unit_usd"] >= 0.0

    # Verify HOME_GARDEN maps to category_code 3 (same bucket as HOME)
    home_garden_ids = {r[COL_PRODUCT_ID] for r in gold_df.filter("category = 'HOME_GARDEN'").collect()}
    assert home_garden_ids, "Expected HOME_GARDEN rows in synthetic affinity dataset"
    home_garden_codes = {r["category_code"] for r in rows if r[COL_PRODUCT_ID] in home_garden_ids}
    assert home_garden_codes == {3}

    labels_df = gold_df.transform(extract_product_affinity_labels)
    assert labels_df.columns == [COL_PRODUCT_ID, COL_SK_PRODUCT, COL_SNAPSHOT_DATE, TARGET_AFFINITY_COL]
    assert labels_df.count() == 24


@pytest.mark.parametrize(
    ("table_exists", "use_fe_client", "expected_status"),
    [
        (False, True, "created_uc_feature_table"),
        (True, True, "merged_uc_feature_table"),
        (True, False, "saved_delta_table"),
    ],
    ids=["create_uc_feature_table", "merge_existing_uc_feature_table", "delta_fallback_without_fe_client"],
)
def test_sync_uc_feature_table_modes(
    table_exists: bool,
    use_fe_client: bool,
    expected_status: str,
) -> None:
    """Verify `sync_uc_feature_table` creates with timeseries key, merges when present, or falls back to Delta."""
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = table_exists
    mock_fe = MagicMock() if use_fe_client else None
    mock_df = MagicMock()

    status = sync_uc_feature_table(
        spark=mock_spark,
        features_df=mock_df,
        table_fqn="cat.ml_features.fe_customer_churn_features",
        primary_key="user_id",
        description="Churn features",
        fe_client=mock_fe,
        timeseries_columns=[COL_SNAPSHOT_DATE],
    )
    assert status == expected_status

    if expected_status == "created_uc_feature_table" and mock_fe is not None:
        mock_fe.create_table.assert_called_once_with(
            name="cat.ml_features.fe_customer_churn_features",
            primary_keys=["user_id", COL_SNAPSHOT_DATE],
            df=mock_df,
            description="Churn features",
            timeseries_columns=[COL_SNAPSHOT_DATE],
        )
        mock_fe.write_table.assert_not_called()
    elif expected_status == "merged_uc_feature_table" and mock_fe is not None:
        mock_fe.write_table.assert_called_once_with(
            name="cat.ml_features.fe_customer_churn_features",
            df=mock_df,
            mode="merge",
        )
    else:
        mock_df.write.mode.return_value.option.return_value.saveAsTable.assert_called_once_with(
            "cat.ml_features.fe_customer_churn_features"
        )


def test_sync_label_table() -> None:
    """Verify `sync_label_table` persists the dedicated label spine table as Delta."""
    mock_lbl_df = MagicMock()
    status = sync_label_table(labels_df=mock_lbl_df, table_fqn="cat.ml_features.lbl_customer_churn")
    assert status == "saved_label_table"
    mock_lbl_df.write.mode.return_value.option.return_value.saveAsTable.assert_called_once_with(
        "cat.ml_features.lbl_customer_churn"
    )
