# Databricks notebook source
"""Task 2b: Train Product Affinity `XGBRegressor` with MLflow 3, Spark-Distributed Optuna, SHAP, and UC Governance.

Joins label-free `ml_features.fe_product_affinity_features` with dedicated label spine
`ml_features.lbl_product_affinity` on `(product_id, snapshot_date)`, performs chronological
out-of-time train/val/test splitting (falling back to random split when fewer than 3 snapshot
periods exist), runs an automated single-feature leakage guard, binds experiment traces to Unity
Catalog (`ml_traces.xgb_product_affinity_otel_*`), executes Spark-distributed MLflow 3 + Optuna
hyperparameter tuning with early stopping, logs SHAP explainability artifacts, re-evaluates the
reigning `@Champion` on the exact same holdout via `mlflow.models.evaluate()` +
`mlflow.validate_evaluation_results()`, and assigns `@Champion` or `@Challenger` alias.
"""

from __future__ import annotations

from collections.abc import Sequence

import click
import mlflow
from pyspark.sql import SparkSession

from src.python.modules.evaluation.model_governance import PromotionDecision
from src.python.modules.features.feature_engineering import (
    engineer_product_affinity_features,
    extract_product_affinity_labels,
)
from src.python.modules.features.synthetic_gold_views import build_sample_gold_affinity_view_df
from src.python.modules.training.xgboost_datasets import (
    detect_single_feature_leakage,
    split_regression_dataset,
)
from src.python.modules.training.xgboost_experiment import AFFINITY_TASK_SPEC, ExperimentRunOutcome
from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    COL_PRODUCT_ID,
    TARGET_AFFINITY_COL,
    MLProjectConfig,
)
from src.python.pipelines.training_runner import TrainingPipelineSpec, execute_training_pipeline


def build_affinity_pipeline_spec(config: MLProjectConfig) -> TrainingPipelineSpec:
    """Construct the immutable `TrainingPipelineSpec` for Product Affinity regression."""
    return TrainingPipelineSpec(
        task_spec=AFFINITY_TASK_SPEC,
        entity_key_col=COL_PRODUCT_ID,
        feature_cols=AFFINITY_FEATURE_COLUMNS,
        target_col=TARGET_AFFINITY_COL,
        feature_table_fqn=config.affinity_feature_table_fqn,
        label_table_fqn=config.affinity_label_table_fqn,
        experiment_path=config.affinity_experiment_path,
        trace_table_prefix=config.affinity_trace_table_prefix,
        model_fqn=config.affinity_model_fqn,
        primary_holdout_metric="holdout_r2",
        min_threshold=config.min_affinity_r2,
        min_absolute_change=config.min_affinity_r2_delta,
        higher_is_better=True,
        feature_importance_filename="affinity_xgb_feature_importance.csv",
        leakage_metric_label="R2",
        synthetic_loader=lambda spk: build_sample_gold_affinity_view_df(spk, num_products=160),
        feature_transformer=engineer_product_affinity_features,
        label_extractor=extract_product_affinity_labels,
        dataset_splitter=split_regression_dataset,
    )


def run_affinity_training_pipeline(
    spark: SparkSession,
    config: MLProjectConfig,
    n_trials: int = 10,
    register_in_uc: bool = True,
    use_spark_optuna: bool = True,
    spark_parallelism: int = 4,
    allow_synthetic: bool = False,
    strict_leakage_check: bool = True,
    leakage_threshold: float = 0.98,
    early_stopping_rounds: int | None = 30,
    worker_n_jobs: int = 1,
    log_shap_artifacts: bool = True,
) -> tuple[ExperimentRunOutcome, PromotionDecision | None]:
    """Execute end-to-end Product Affinity XGBoost training, leakage guard, SHAP logging, and alias governance."""
    return execute_training_pipeline(
        spark=spark,
        config=config,
        spec=build_affinity_pipeline_spec(config),
        n_trials=n_trials,
        register_in_uc=register_in_uc,
        use_spark_optuna=use_spark_optuna,
        spark_parallelism=spark_parallelism,
        allow_synthetic=allow_synthetic,
        strict_leakage_check=strict_leakage_check,
        leakage_threshold=leakage_threshold,
        leakage_detector=detect_single_feature_leakage,
        early_stopping_rounds=early_stopping_rounds,
        worker_n_jobs=worker_n_jobs,
        log_shap_artifacts=log_shap_artifacts,
    )


@click.command(
    name="train-affinity-xgboost",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--catalog", default="data_engineering_dev", show_default=True, help="Target Unity Catalog name.")
@click.option("--feature-schema", default="ml_features", show_default=True, help="Unity Catalog Feature Store schema.")
@click.option("--model-schema", default="ml_models", show_default=True, help="Unity Catalog Model Registry schema.")
@click.option(
    "--trace-schema", default="ml_traces", show_default=True, help="Unity Catalog OpenTelemetry trace schema."
)
@click.option(
    "--experiment-path",
    default="/Shared/mlflow_xgboost_product_affinity_uc",
    show_default=True,
    help="Databricks workspace MLflow experiment path bound to Unity Catalog trace tables.",
)
@click.option(
    "--trace-warehouse-id",
    default=None,
    envvar="MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    help="Optional Databricks SQL Warehouse ID for UC trace storage queries.",
)
@click.option("--optuna-trials", default=10, type=int, show_default=True, help="Number of Optuna HPO trials.")
@click.option(
    "--spark-parallelism",
    default=4,
    type=int,
    show_default=True,
    help="Number of parallel Optuna trials per Spark mapInPandas wave.",
)
@click.option(
    "--worker-n-jobs",
    default=1,
    type=int,
    show_default=True,
    help="XGBoost CPU thread cap per Spark worker task to prevent oversubscription.",
)
@click.option(
    "--early-stopping-rounds",
    default=30,
    type=int,
    show_default=True,
    help="Validation early stopping rounds during Optuna HPO (0 disables early stopping).",
)
@click.option(
    "--min-r2",
    default=0.60,
    type=float,
    show_default=True,
    help="Minimum holdout R2 required for @Champion promotion.",
)
@click.option(
    "--allow-synthetic/--no-allow-synthetic",
    default=False,
    show_default=True,
    help="Permit deterministic synthetic training data when UC feature/label tables are absent (dev/test only).",
)
@click.option(
    "--strict-leakage-check/--no-strict-leakage-check",
    default=True,
    show_default=True,
    help="Fail fast when any single feature exceeds the depth-1 stump leakage threshold.",
)
@click.option(
    "--log-shap-artifacts/--no-log-shap-artifacts",
    default=True,
    show_default=True,
    help="Compute TreeSHAP feature attributions and log explainability plots/tables to MLflow.",
)
def cli(
    catalog: str,
    feature_schema: str,
    model_schema: str,
    trace_schema: str,
    experiment_path: str,
    trace_warehouse_id: str | None,
    optuna_trials: int,
    spark_parallelism: int,
    worker_n_jobs: int,
    early_stopping_rounds: int,
    min_r2: float,
    allow_synthetic: bool,
    strict_leakage_check: bool,
    log_shap_artifacts: bool,
) -> tuple[ExperimentRunOutcome, PromotionDecision | None]:
    """Click CLI command training and governing the Product Affinity XGBRegressor with Spark Optuna."""
    config = MLProjectConfig(
        catalog=catalog,
        feature_schema=feature_schema,
        model_schema=model_schema,
        trace_schema=trace_schema,
        affinity_experiment_path=experiment_path,
        trace_sql_warehouse_id=trace_warehouse_id or None,
        min_affinity_r2=min_r2,
    )
    mlflow.set_registry_uri("databricks-uc")
    spark = SparkSession.builder.getOrCreate()
    return run_affinity_training_pipeline(
        spark=spark,
        config=config,
        n_trials=optuna_trials,
        register_in_uc=True,
        use_spark_optuna=True,
        spark_parallelism=spark_parallelism,
        allow_synthetic=allow_synthetic,
        strict_leakage_check=strict_leakage_check,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        worker_n_jobs=worker_n_jobs,
        log_shap_artifacts=log_shap_artifacts,
    )


def main(argv: Sequence[str] | None = None) -> tuple[ExperimentRunOutcome, PromotionDecision | None]:
    """Programmatic and Databricks entrypoint invoking the Click CLI with `standalone_mode=False`."""
    result: tuple[ExperimentRunOutcome, PromotionDecision | None] = cli.main(
        args=list(argv) if argv is not None else None,
        standalone_mode=False,
    )
    return result


if __name__ == "__main__":
    main()
