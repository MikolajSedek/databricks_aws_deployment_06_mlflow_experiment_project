"""Unit tests for dataset splitting (out-of-time & fallback), class weights, feature importance, and leakage guard."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
import xgboost as xgb
from pyspark.sql import SparkSession
from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
    extract_customer_churn_labels,
    extract_product_affinity_labels,
)
from src.python.modules.features.synthetic_gold_views import (
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)
from src.python.modules.training.xgboost_datasets import (
    DatasetSplits,
    compute_scale_pos_weight,
    detect_single_feature_leakage,
    extract_feature_importance_df,
    split_classification_dataset,
    split_regression_dataset,
)
from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    COL_PRODUCT_ID,
    COL_SNAPSHOT_DATE,
    COL_USER_ID,
    TARGET_AFFINITY_COL,
    TARGET_CHURN_COL,
)


@pytest.fixture(scope="module")
def _cached_joined_churn_pdf(spark: SparkSession) -> pd.DataFrame:
    """Build the deterministic Customer Churn Pandas DataFrame once per module to avoid redundant Spark joins."""
    raw_df = build_sample_gold_churn_view_df(spark, num_users=150, num_snapshots=5)
    fe_df = raw_df.transform(engineer_customer_churn_features)
    lbl_df = raw_df.transform(extract_customer_churn_labels).select(COL_USER_ID, COL_SNAPSHOT_DATE, TARGET_CHURN_COL)
    return fe_df.join(lbl_df, on=[COL_USER_ID, COL_SNAPSHOT_DATE], how="inner").toPandas()


@pytest.fixture(scope="module")
def _cached_joined_affinity_pdf(spark: SparkSession) -> pd.DataFrame:
    """Build the deterministic Product Affinity Pandas DataFrame once per module to avoid redundant Spark joins."""
    raw_df = build_sample_gold_affinity_view_df(spark, num_products=120, num_snapshots=5)
    fe_df = raw_df.transform(engineer_product_affinity_features)
    lbl_df = raw_df.transform(extract_product_affinity_labels).select(
        COL_PRODUCT_ID, COL_SNAPSHOT_DATE, TARGET_AFFINITY_COL
    )
    return fe_df.join(lbl_df, on=[COL_PRODUCT_ID, COL_SNAPSHOT_DATE], how="inner").toPandas()


@pytest.fixture()
def joined_churn_pdf(_cached_joined_churn_pdf: pd.DataFrame) -> pd.DataFrame:
    """Return an isolated copy of the cached Customer Churn Pandas DataFrame."""
    return _cached_joined_churn_pdf.copy()


@pytest.fixture()
def joined_affinity_pdf(_cached_joined_affinity_pdf: pd.DataFrame) -> pd.DataFrame:
    """Return an isolated copy of the cached Product Affinity Pandas DataFrame."""
    return _cached_joined_affinity_pdf.copy()


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize(
    ("task_type", "max_periods", "time_col", "expected_strategy"),
    [
        ("classification", 5, COL_SNAPSHOT_DATE, "temporal_out_of_time"),
        ("classification", 2, COL_SNAPSHOT_DATE, "stratified_random"),
        ("classification", 5, None, "stratified_random"),
        ("regression", 5, COL_SNAPSHOT_DATE, "temporal_out_of_time"),
        ("regression", 5, None, "random"),
    ],
    ids=[
        "classification_temporal_out_of_time",
        "classification_two_periods_fallback",
        "classification_no_time_col_fallback",
        "regression_temporal_out_of_time",
        "regression_no_time_col_fallback",
    ],
)
def test_dataset_split_strategies(
    joined_churn_pdf: pd.DataFrame,
    joined_affinity_pdf: pd.DataFrame,
    task_type: Literal["classification", "regression"],
    max_periods: int,
    time_col: str | None,
    expected_strategy: str,
) -> None:
    """Verify chronological out-of-time splitting and stratified/random fallback behavior."""
    if task_type == "classification":
        pdf = joined_churn_pdf
        if max_periods < 5:
            selected_dates = sorted(pdf[COL_SNAPSHOT_DATE].unique())[:max_periods]
            pdf = pdf[pdf[COL_SNAPSHOT_DATE].isin(selected_dates)]
        splits = split_classification_dataset(
            pdf=pdf,
            feature_cols=CHURN_FEATURE_COLUMNS,
            target_col=TARGET_CHURN_COL,
            time_col=time_col,
        )
    else:
        pdf = joined_affinity_pdf
        splits = split_regression_dataset(
            pdf=pdf,
            feature_cols=AFFINITY_FEATURE_COLUMNS,
            target_col=TARGET_AFFINITY_COL,
            time_col=time_col,
        )

    assert splits.split_strategy == expected_strategy
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == len(pdf)
    assert len(splits.X_trainval) == len(splits.X_train) + len(splits.X_val)


@pytest.mark.parametrize(
    ("labels", "expected_weight"),
    [
        ([0, 0, 0], 1.0),
        ([1, 1, 1], 1.0),
        ([0, 0, 0, 1], 3.0),
        ([0, 1, 1, 1], 0.3333),
    ],
    ids=["all_negatives", "all_positives", "3_to_1_imbalance", "1_to_3_imbalance"],
)
def test_compute_scale_pos_weight(labels: list[int], expected_weight: float) -> None:
    """Verify `compute_scale_pos_weight` returns balanced weights and defaults to 1.0 for single-class inputs."""
    assert compute_scale_pos_weight(pd.Series(labels)) == pytest.approx(expected_weight)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
def test_extract_feature_importance_df(joined_churn_pdf: pd.DataFrame) -> None:
    """Verify `extract_feature_importance_df` emits sorted gain and weight scores for all features."""
    splits = split_classification_dataset(
        pdf=joined_churn_pdf,
        feature_cols=CHURN_FEATURE_COLUMNS,
        target_col=TARGET_CHURN_COL,
        time_col=COL_SNAPSHOT_DATE,
    )
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, random_state=42)
    model.fit(splits.X_train, splits.y_train)
    imp_df = extract_feature_importance_df(model, CHURN_FEATURE_COLUMNS)
    assert list(imp_df.columns) == ["feature", "importance_gain", "importance_weight"]
    assert len(imp_df) == len(CHURN_FEATURE_COLUMNS)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize(
    "task_type",
    ["classification", "regression"],
    ids=["churn_classification_leakage_guard", "affinity_regression_leakage_guard"],
)
def test_single_feature_leakage_guard_ci(
    joined_churn_pdf: pd.DataFrame,
    joined_affinity_pdf: pd.DataFrame,
    task_type: Literal["classification", "regression"],
) -> None:
    """CI leakage guard: clean feature contracts pass (<0.98) while injected target leakage is caught."""
    if task_type == "classification":
        feature_cols = CHURN_FEATURE_COLUMNS
        splits = split_classification_dataset(
            pdf=joined_churn_pdf,
            feature_cols=feature_cols,
            target_col=TARGET_CHURN_COL,
            time_col=COL_SNAPSHOT_DATE,
        )
        assert detect_single_feature_leakage(splits, feature_cols, "classification", threshold=0.98) == {}

        leaky_splits = DatasetSplits(
            X_train=splits.X_train.assign(leaked_signal=splits.y_train.astype(float) * 100.0),
            X_val=splits.X_val.assign(leaked_signal=splits.y_val.astype(float) * 100.0),
            X_trainval=splits.X_trainval.assign(leaked_signal=splits.y_trainval.astype(float) * 100.0),
            X_test=splits.X_test.assign(leaked_signal=splits.y_test.astype(float) * 100.0),
            y_train=splits.y_train,
            y_val=splits.y_val,
            y_trainval=splits.y_trainval,
            y_test=splits.y_test,
            split_strategy=splits.split_strategy,
        )
    else:
        feature_cols = AFFINITY_FEATURE_COLUMNS
        splits = split_regression_dataset(
            pdf=joined_affinity_pdf,
            feature_cols=feature_cols,
            target_col=TARGET_AFFINITY_COL,
            time_col=COL_SNAPSHOT_DATE,
        )
        assert detect_single_feature_leakage(splits, feature_cols, "regression", threshold=0.98) == {}

        # Two-level target proxy guarantees depth-1 regression stump R2 == 1.0
        step_y_train = (splits.y_train >= splits.y_train.median()).astype(float) * 50.0
        step_y_val = (splits.y_val >= splits.y_train.median()).astype(float) * 50.0
        leaky_splits = DatasetSplits(
            X_train=splits.X_train.assign(leaked_signal=step_y_train),
            X_val=splits.X_val.assign(leaked_signal=step_y_val),
            X_trainval=splits.X_trainval.assign(leaked_signal=0.0),
            X_test=splits.X_test.assign(leaked_signal=0.0),
            y_train=step_y_train,
            y_val=step_y_val,
            y_trainval=splits.y_trainval,
            y_test=splits.y_test,
            split_strategy=splits.split_strategy,
        )

    flagged = detect_single_feature_leakage(
        leaky_splits,
        (*feature_cols, "leaked_signal"),
        task_type,
        threshold=0.98,
    )
    assert "leaked_signal" in flagged
    assert flagged["leaked_signal"] >= 0.98
