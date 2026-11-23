"""Unit tests for Spark batch scoring formatting, scoring_date Delta writes, strict alias resolution, and FS scoring."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException
from pyspark.sql import SparkSession
from src.python.modules.inference.batch_scoring import (
    _SparkUdfBenignWarningFilter,
    configure_silent_spark_udf_logging,
    format_affinity_predictions_df,
    format_churn_predictions_df,
    resolve_scoring_date,
    resolve_scoring_model_uri,
    score_with_feature_store,
    write_predictions_delta_table,
)


@pytest.mark.parametrize(
    ("logger_name", "message", "expected_allow"),
    [
        ("mlflow.pyfunc", 'Calling `spark_udf()` with `env_manager="local"`', False),
        ("databricks.sdk.mixins.files", "Unexpected status resolving file endpoints (403)", False),
        ("mlflow.pyfunc", "Real dependency mismatch warning", True),
    ],
    ids=["suppress_local_env_manager", "suppress_files_403", "keep_real_warning"],
)
def test_spark_udf_benign_warning_filter(logger_name: str, message: str, expected_allow: bool) -> None:
    """Verify `_SparkUdfBenignWarningFilter` suppresses only known benign Serverless warnings."""
    configure_silent_spark_udf_logging()
    flt = _SparkUdfBenignWarningFilter()
    record = logging.LogRecord(logger_name, logging.WARNING, "", 1, message, (), None)
    assert flt.filter(record) is expected_allow


def test_resolve_scoring_date_validation() -> None:
    """Verify `resolve_scoring_date` defaults to UTC today, accepts ISO dates, and rejects malformed inputs."""
    default_date = resolve_scoring_date(None)
    assert date.fromisoformat(default_date).isoformat() == default_date
    assert resolve_scoring_date("2026-05-15") == "2026-05-15"
    with pytest.raises(ValueError):
        resolve_scoring_date("2026-05-15; DROP TABLE x")


def test_resolve_scoring_model_uri_strict_fallback_and_errors(
    missing_alias_error: Callable[[str], MlflowException],
) -> None:
    """Verify strict `@Champion` resolution, opt-in `@Challenger` fallback, and transient error propagation."""
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = missing_alias_error("No Champion")
    with pytest.raises(LookupError, match="@Champion"):
        resolve_scoring_model_uri(mock_client, "cat.ml_models.churn", "Champion")

    mock_client.get_model_version_by_alias.side_effect = [
        missing_alias_error("No Champion"),
        SimpleNamespace(version="7"),
    ]
    uri, alias, version = resolve_scoring_model_uri(
        mock_client, "cat.ml_models.churn", "Champion", fallback_alias="Challenger"
    )
    assert uri == "models:/cat.ml_models.churn@Challenger"
    assert alias == "Challenger"
    assert version == "7"

    mock_client.get_model_version_by_alias.side_effect = ConnectionError("Network failure")
    with pytest.raises(ConnectionError, match="Network failure"):
        resolve_scoring_model_uri(mock_client, "cat.ml_models.churn", "Champion")


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize("include_label", [True, False], ids=["labeled_backtest", "unlabeled_production"])
def test_format_churn_predictions_df(spark: SparkSession, include_label: bool) -> None:
    """Verify Customer Churn prediction tiering, immutable `model_version`, `scoring_date`, and optional label."""
    if include_label:
        raw_df = spark.createDataFrame(
            [
                ("hk_u1", "USR-1", 1, 0.85),
                ("hk_u2", "USR-2", 0, 0.60),
                ("hk_u3", "USR-3", 0, 0.15),
            ],
            ["sk_user", "user_id", "is_churn_candidate", "raw_prediction"],
        )
    else:
        raw_df = spark.createDataFrame(
            [
                ("hk_u1", "USR-1", 0.85),
                ("hk_u2", "USR-2", 0.60),
                ("hk_u3", "USR-3", 0.15),
            ],
            ["sk_user", "user_id", "raw_prediction"],
        )

    formatted = format_churn_predictions_df(
        scored_df=raw_df,
        model_uri="models:/cat.ml_models.churn@Champion",
        model_version="12",
        scoring_date="2026-05-01",
    )
    rows = formatted.collect()
    tiers = {r["sk_user"]: r["churn_action_tier"] for r in rows}
    assert tiers == {
        "hk_u1": "HIGH_RISK_RETENTION",
        "hk_u2": "WATCHLIST",
        "hk_u3": "HEALTHY",
    }
    assert all(r["model_version"] == "12" for r in rows)
    assert all(str(r["scoring_date"]) == "2026-05-01" for r in rows)
    if include_label:
        assert {r["sk_user"]: r["is_churn_candidate"] for r in rows} == {"hk_u1": 1, "hk_u2": 0, "hk_u3": 0}
    else:
        assert all(r["is_churn_candidate"] is None for r in rows)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize("include_label", [True, False], ids=["labeled_backtest", "unlabeled_production"])
def test_format_affinity_predictions_df(spark: SparkSession, include_label: bool) -> None:
    """Verify Product Affinity tiering, immutable `model_version`, `scoring_date`, and optional actuals."""
    if include_label:
        raw_df = spark.createDataFrame(
            [
                ("hk_p1", "PRD-1", 72.0, 70.0),
                ("hk_p2", "PRD-2", 48.0, 50.0),
                ("hk_p3", "PRD-3", 20.0, 22.0),
            ],
            ["sk_product", "product_id", "affinity_index_score", "raw_prediction"],
        )
    else:
        raw_df = spark.createDataFrame(
            [
                ("hk_p1", "PRD-1", 70.0),
                ("hk_p2", "PRD-2", 50.0),
                ("hk_p3", "PRD-3", 22.0),
            ],
            ["sk_product", "product_id", "raw_prediction"],
        )

    formatted = format_affinity_predictions_df(
        scored_df=raw_df,
        model_uri="models:/cat.ml_models.affinity@Champion",
        model_version="9",
        scoring_date="2026-05-01",
    )
    rows = formatted.collect()
    tiers = {r["sk_product"]: r["affinity_tier"] for r in rows}
    assert tiers == {
        "hk_p1": "HERO_PRODUCT",
        "hk_p2": "CORE_ASSORTMENT",
        "hk_p3": "LONG_TAIL",
    }
    assert all(r["model_version"] == "9" for r in rows)
    assert all(str(r["scoring_date"]) == "2026-05-01" for r in rows)
    if include_label:
        assert all(r["affinity_residual"] is not None for r in rows)
    else:
        assert all(r["affinity_index_score"] is None for r in rows)
        assert all(r["affinity_residual"] is None for r in rows)


@pytest.mark.parametrize(
    ("table_exists", "write_mode", "expected_mode", "expected_replace_where"),
    [
        (False, "replace_where", "overwrite", None),
        (True, "replace_where", "overwrite", "scoring_date = DATE '2026-05-01'"),
        (True, "append", "append", None),
    ],
    ids=["initial_create", "existing_replace_where", "existing_append"],
)
def test_write_predictions_delta_table_modes(
    table_exists: bool,
    write_mode: Any,
    expected_mode: str,
    expected_replace_where: str | None,
) -> None:
    """Verify Delta prediction persistence preserves history via `replaceWhere`/`append` without `overwriteSchema`."""
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.return_value = table_exists
    mock_df = MagicMock()
    delta_writer = mock_df.write.format.return_value
    mode_writer = delta_writer.mode.return_value
    option_writer = mode_writer.option.return_value

    resolved = write_predictions_delta_table(
        spark=mock_spark,
        predictions_df=mock_df,
        table_fqn="cat.ml_predictions.pred_churn",
        scoring_date="2026-05-01",
        write_mode=write_mode,
    )
    assert resolved == "2026-05-01"
    mock_df.write.format.assert_called_once_with("delta")
    delta_writer.mode.assert_called_once_with(expected_mode)
    if expected_replace_where is not None:
        mode_writer.option.assert_called_once_with("replaceWhere", expected_replace_where)
        option_writer.saveAsTable.assert_called_once_with("cat.ml_predictions.pred_churn")
    else:
        mode_writer.option.assert_not_called()
        mode_writer.saveAsTable.assert_called_once_with("cat.ml_predictions.pred_churn")


def test_write_predictions_delta_table_rejects_invalid_mode() -> None:
    """Verify unsupported write modes raise `ValueError`."""
    with pytest.raises(ValueError, match="Unsupported prediction write_mode"):
        write_predictions_delta_table(
            spark=MagicMock(),
            predictions_df=MagicMock(),
            table_fqn="cat.ml_predictions.pred_churn",
            scoring_date="2026-05-01",
            write_mode="truncate_all",  # type: ignore[arg-type]
        )


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
def test_score_with_feature_store(spark: SparkSession) -> None:
    """Verify `score_with_feature_store` delegates to `fe_client.score_batch`."""
    keys_df = spark.createDataFrame([("USR-1",)], ["user_id"])
    mock_fe = MagicMock()
    score_with_feature_store(
        fe_client=mock_fe,
        keys_df=keys_df,
        model_uri="models:/cat.ml_models.churn@Champion",
    )
    mock_fe.score_batch.assert_called_once()
