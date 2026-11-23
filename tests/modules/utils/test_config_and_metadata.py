"""Unit tests for immutable `MLProjectConfig`, leak-free feature contracts, and Unity Catalog metadata DDLs."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    COL_SNAPSHOT_DATE,
    DEFAULT_ML_CONFIG,
    LEAKY_AFFINITY_COLUMNS,
    LEAKY_CHURN_COLUMNS,
    MLProjectConfig,
)
from src.python.modules.utils.metadata import (
    ML_COLUMN_DESCRIPTIONS,
    attach_column_descriptions,
    build_schema_provisioning_ddls,
)


@pytest.mark.parametrize(
    ("attr_name", "expected_fqn"),
    [
        ("upstream_churn_view_fqn", "test_cat.gold.vw_ml_customer_churn_features"),
        ("upstream_affinity_view_fqn", "test_cat.gold.vw_ml_product_affinity_features"),
        ("churn_feature_table_fqn", "test_cat.ml_features.fe_customer_churn_features"),
        ("affinity_feature_table_fqn", "test_cat.ml_features.fe_product_affinity_features"),
        ("churn_label_table_fqn", "test_cat.ml_features.lbl_customer_churn"),
        ("affinity_label_table_fqn", "test_cat.ml_features.lbl_product_affinity"),
        ("churn_model_fqn", "test_cat.ml_models.xgb_customer_churn_classifier"),
        ("affinity_model_fqn", "test_cat.ml_models.xgb_product_affinity_regressor"),
        ("churn_predictions_table_fqn", "test_cat.ml_predictions.pred_customer_churn_xgboost"),
        ("affinity_predictions_table_fqn", "test_cat.ml_predictions.pred_product_affinity_xgboost"),
        ("churn_trace_prefix_fqn", "test_cat.ml_traces.xgb_customer_churn"),
        ("affinity_trace_prefix_fqn", "test_cat.ml_traces.xgb_product_affinity"),
    ],
)
def test_ml_project_config_fqn_properties(attr_name: str, expected_fqn: str) -> None:
    """Verify 3-level Unity Catalog FQN properties resolve consistently from `MLProjectConfig`."""
    cfg = MLProjectConfig(catalog="test_cat")
    assert getattr(cfg, attr_name) == expected_fqn


def test_default_config_and_leak_free_feature_contracts() -> None:
    """Verify default catalog, alias URI helper, and strict disjointness from leaky upstream formula inputs."""
    cfg = MLProjectConfig(catalog="test_cat")
    assert cfg.trace_sql_warehouse_id is None
    assert cfg.model_alias_uri(cfg.churn_model_fqn, "Champion") == (
        "models:/test_cat.ml_models.xgb_customer_churn_classifier@Champion"
    )
    assert DEFAULT_ML_CONFIG.catalog == "data_engineering_dev"

    # Contract invariant: upstream deterministic formula inputs must never appear in feature tuples
    assert LEAKY_CHURN_COLUMNS.isdisjoint(CHURN_FEATURE_COLUMNS)
    assert LEAKY_AFFINITY_COLUMNS.isdisjoint(AFFINITY_FEATURE_COLUMNS)
    assert COL_SNAPSHOT_DATE in ML_COLUMN_DESCRIPTIONS


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
def test_schema_provisioning_ddls_and_column_descriptions(spark: SparkSession) -> None:
    """Verify schema provisioning DDLs and Unity Catalog column comment attachment."""
    cfg = MLProjectConfig(catalog="test_cat")
    ddls = build_schema_provisioning_ddls(cfg)
    assert len(ddls) == 4
    assert all("CREATE SCHEMA IF NOT EXISTS `test_cat`." in stmt for stmt in ddls)

    raw_df = spark.createDataFrame([("hk_1", 1, "unknown")], ["sk_user", "is_churn_candidate", "custom_col"])
    annotated_df = attach_column_descriptions(raw_df)
    fields_by_name = {field.name: field for field in annotated_df.schema.fields}
    assert fields_by_name["sk_user"].metadata["comment"] == ML_COLUMN_DESCRIPTIONS["sk_user"]
    assert fields_by_name["custom_col"].metadata["comment"] == "Governed ML attribute custom_col."
