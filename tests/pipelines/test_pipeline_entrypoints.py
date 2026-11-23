"""Unit tests for Databricks pipeline fail-fast guards, leakage aborts, and Click CLI entrypoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner
from pyspark.sql import SparkSession
from src.python.modules.utils.config import MLProjectConfig
from src.python.pipelines import (
    batch_score_predictions,
    prepare_uc_features,
    train_affinity_xgboost,
    train_churn_xgboost,
)


@pytest.mark.parametrize(
    ("pipeline_fn", "kwargs", "table_exists_side_effect", "expected_missing_table"),
    [
        (
            prepare_uc_features.run_feature_preparation_pipeline,
            {"migrate_uc_traces": False, "allow_synthetic": False},
            [False],
            "vw_ml_customer_churn_features",
        ),
        (
            train_churn_xgboost.run_churn_training_pipeline,
            {"allow_synthetic": False},
            [False, False],
            "fe_customer_churn_features",
        ),
        (
            train_churn_xgboost.run_churn_training_pipeline,
            {"allow_synthetic": False},
            [True, False],
            "lbl_customer_churn",
        ),
        (
            train_affinity_xgboost.run_affinity_training_pipeline,
            {"allow_synthetic": False},
            [False, False],
            "fe_product_affinity_features",
        ),
        (
            train_affinity_xgboost.run_affinity_training_pipeline,
            {"allow_synthetic": False},
            [True, False],
            "lbl_product_affinity",
        ),
        (
            batch_score_predictions.run_batch_scoring_pipeline,
            {"persist_tables": False, "allow_synthetic": False},
            [False],
            "fe_customer_churn_features",
        ),
    ],
    ids=[
        "prepare_missing_gold_view",
        "train_churn_missing_features",
        "train_churn_missing_labels",
        "train_affinity_missing_features",
        "train_affinity_missing_labels",
        "batch_score_missing_features",
    ],
)
def test_pipelines_fail_fast_when_tables_missing(
    pipeline_fn: Any,
    kwargs: dict[str, Any],
    table_exists_side_effect: list[bool],
    expected_missing_table: str,
) -> None:
    """Verify every pipeline raises `FileNotFoundError` naming the missing upstream table by default."""
    missing_cfg = MLProjectConfig(catalog="nonexistent_catalog_guard")
    mock_spark = MagicMock()
    mock_spark.catalog.tableExists.side_effect = table_exists_side_effect

    with pytest.raises(FileNotFoundError, match=expected_missing_table):
        pipeline_fn(spark=mock_spark, config=missing_cfg, **kwargs)


@pytest.mark.spark
@pytest.mark.xdist_group("spark_unit")
@pytest.mark.parametrize(
    ("pipeline_module", "runner_attr"),
    [
        (train_churn_xgboost, "run_churn_training_pipeline"),
        (train_affinity_xgboost, "run_affinity_training_pipeline"),
    ],
    ids=["churn_strict_leakage_abort", "affinity_strict_leakage_abort"],
)
def test_training_pipelines_abort_on_strict_leakage_detection(
    spark: SparkSession,
    pipeline_module: Any,
    runner_attr: str,
) -> None:
    """Verify training pipelines abort with `ValueError` when strict single-feature leakage check fires."""
    cfg = MLProjectConfig(catalog="nonexistent_catalog_guard")
    runner_fn = getattr(pipeline_module, runner_attr)

    with (
        patch("pyspark.sql.catalog.Catalog.tableExists", return_value=False),
        patch.object(pipeline_module, "detect_single_feature_leakage", return_value={"leaked_col": 0.995}),
        pytest.raises(ValueError, match="Single-feature target leakage detected"),
    ):
        runner_fn(
            spark=spark,
            config=cfg,
            allow_synthetic=True,
            strict_leakage_check=True,
        )


@pytest.mark.parametrize(
    ("cli_cmd", "expected_flags"),
    [
        (
            prepare_uc_features.cli,
            ("--catalog", "--trace-warehouse-id", "--migrate-uc-traces", "--allow-synthetic"),
        ),
        (
            train_churn_xgboost.cli,
            (
                "--catalog",
                "--trace-warehouse-id",
                "--worker-n-jobs",
                "--early-stopping-rounds",
                "--log-shap-artifacts",
                "--allow-synthetic",
                "--strict-leakage-check",
            ),
        ),
        (
            train_affinity_xgboost.cli,
            (
                "--catalog",
                "--trace-warehouse-id",
                "--worker-n-jobs",
                "--early-stopping-rounds",
                "--log-shap-artifacts",
                "--allow-synthetic",
                "--strict-leakage-check",
            ),
        ),
        (
            batch_score_predictions.cli,
            (
                "--catalog",
                "--trace-warehouse-id",
                "--scoring-date",
                "--write-mode",
                "--allow-synthetic",
                "--allow-challenger-fallback",
            ),
        ),
    ],
    ids=["prepare_cli", "train_churn_cli", "train_affinity_cli", "batch_score_cli"],
)
def test_click_cli_help_flags(cli_cmd: click.Command, expected_flags: tuple[str, ...]) -> None:
    """Verify all four pipeline Click CLIs render `--help` with their required flags."""
    runner = CliRunner()
    help_res = runner.invoke(cli_cmd, ["--help"])
    assert help_res.exit_code == 0
    for flag in expected_flags:
        assert flag in help_res.output


def test_click_pipeline_main_entrypoints() -> None:
    """Verify all four pipeline `main()` entrypoints parse options and tolerate extra Databricks args."""
    # Patch `SparkSession.Builder.getOrCreate` on the class (`SparkSession.builder` returns a fresh instance per access).
    with patch.object(SparkSession.Builder, "getOrCreate", return_value=MagicMock()):
        with patch.object(prepare_uc_features, "run_feature_preparation_pipeline") as mock_prep:
            cfg = prepare_uc_features.main(
                ["--catalog", "cli_cat", "--trace-warehouse-id", "wh_abc", "--unknown-dbx-flag", "ignored"]
            )
            assert cfg.catalog == "cli_cat"
            assert cfg.trace_sql_warehouse_id == "wh_abc"
            mock_prep.assert_called_once()

        with (
            patch.object(train_churn_xgboost.mlflow, "set_registry_uri"),
            patch.object(
                train_churn_xgboost,
                "run_churn_training_pipeline",
                return_value=(MagicMock(), None),
            ) as mock_churn,
        ):
            train_churn_xgboost.main(["--catalog", "cli_cat", "--optuna-trials", "3", "--spark-parallelism", "2"])
            mock_churn.assert_called_once()

        with (
            patch.object(train_affinity_xgboost.mlflow, "set_registry_uri"),
            patch.object(
                train_affinity_xgboost,
                "run_affinity_training_pipeline",
                return_value=(MagicMock(), None),
            ) as mock_aff,
        ):
            train_affinity_xgboost.main(["--catalog", "cli_cat", "--optuna-trials", "3", "--spark-parallelism", "2"])
            mock_aff.assert_called_once()

        with (
            patch.object(batch_score_predictions.mlflow, "set_registry_uri"),
            patch.object(
                batch_score_predictions,
                "run_batch_scoring_pipeline",
                return_value=(MagicMock(), MagicMock()),
            ) as mock_score,
        ):
            batch_score_predictions.main(
                [
                    "--catalog",
                    "cli_cat",
                    "--alias",
                    "Challenger",
                    "--scoring-date",
                    "2026-05-01",
                    "--write-mode",
                    "append",
                    "--allow-challenger-fallback",
                ]
            )
            mock_score.assert_called_once()
