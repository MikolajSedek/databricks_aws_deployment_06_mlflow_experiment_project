"""Shared training pipeline runner deduplicating Customer Churn and Product Affinity workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession

from src.python.modules.evaluation.model_governance import (
    PromotionDecision,
    log_feature_importance_artifact,
    promote_champion_challenger_alias,
)
from src.python.modules.evaluation.uc_tracing import configure_uc_experiment_tracing
from src.python.modules.training.xgboost_datasets import DatasetSplits, detect_single_feature_leakage
from src.python.modules.training.xgboost_experiment import (
    ExperimentRunOutcome,
    ModelTaskSpec,
    run_xgb_experiment,
)
from src.python.modules.utils.config import COL_SNAPSHOT_DATE, MLProjectConfig


@dataclass(frozen=True)
class TrainingPipelineSpec:
    """Immutable binding of domain tables, transformers, and governance gates for a training pipeline."""

    task_spec: ModelTaskSpec
    entity_key_col: str
    feature_cols: tuple[str, ...]
    target_col: str
    feature_table_fqn: str
    label_table_fqn: str
    experiment_path: str
    trace_table_prefix: str
    model_fqn: str
    primary_holdout_metric: str
    min_threshold: float
    min_absolute_change: float
    higher_is_better: bool
    feature_importance_filename: str
    leakage_metric_label: str
    synthetic_loader: Callable[[SparkSession], DataFrame]
    feature_transformer: Callable[[DataFrame], DataFrame]
    label_extractor: Callable[[DataFrame], DataFrame]
    dataset_splitter: Callable[..., DatasetSplits]


def _load_joined_training_df(
    spark: SparkSession,
    spec: TrainingPipelineSpec,
    allow_synthetic: bool,
) -> tuple[DataFrame, str, str]:
    """Load and join label-free features with the label spine, failing fast if absent unless allowed."""
    has_features = spark.catalog.tableExists(spec.feature_table_fqn)
    has_labels = spark.catalog.tableExists(spec.label_table_fqn)

    if has_features and has_labels:
        # Join on (entity_key, snapshot_date) to enforce point-in-time alignment between features and labels.
        spark_features_df = spark.read.table(spec.feature_table_fqn)
        spark_labels_df = spark.read.table(spec.label_table_fqn).select(
            spec.entity_key_col, COL_SNAPSHOT_DATE, spec.target_col
        )
        training_df = spark_features_df.join(spark_labels_df, on=[spec.entity_key_col, COL_SNAPSHOT_DATE], how="inner")
        return training_df, "gold", f"uc_table://{spec.feature_table_fqn}"

    if allow_synthetic:
        raw_df = spec.synthetic_loader(spark)
        spark_features_df = raw_df.transform(spec.feature_transformer)
        spark_labels_df = raw_df.transform(spec.label_extractor).select(
            spec.entity_key_col, COL_SNAPSHOT_DATE, spec.target_col
        )
        training_df = spark_features_df.join(spark_labels_df, on=[spec.entity_key_col, COL_SNAPSHOT_DATE], how="inner")
        return training_df, "synthetic", f"synthetic://{spec.task_spec.dataset_name_prefix}"

    missing = [
        tbl
        for tbl, exists in (
            (spec.feature_table_fqn, has_features),
            (spec.label_table_fqn, has_labels),
        )
        if not exists
    ]
    raise FileNotFoundError(
        f"Required training table(s) missing: {', '.join(missing)}. "
        "Run prepare_uc_features first or pass --allow-synthetic for local/dev testing."
    )


def _validate_no_single_feature_leakage(
    splits: DatasetSplits,
    spec: TrainingPipelineSpec,
    strict_leakage_check: bool,
    leakage_threshold: float,
    leakage_detector: Callable[..., dict[str, float]],
) -> None:
    """Abort training when any standalone depth-1 feature stump exceeds `leakage_threshold`."""
    leaked_features = leakage_detector(
        splits=splits,
        feature_cols=spec.feature_cols,
        task_type=spec.task_spec.task_type,
        threshold=leakage_threshold,
    )
    if strict_leakage_check and leaked_features:
        raise ValueError(
            f"Single-feature target leakage detected (depth-1 val {spec.leakage_metric_label} >= "
            f"{leakage_threshold:.2f}): {leaked_features}. Remove leaky features before training."
        )


def _govern_registered_candidate(
    spec: TrainingPipelineSpec,
    splits: DatasetSplits,
    exp_result: ExperimentRunOutcome,
    data_source: str,
) -> PromotionDecision:
    """Run same-holdout `@Champion` comparison, assign UC alias, and record optimal decision threshold tag."""
    assert exp_result.registered_model_version is not None  # nosec B101
    holdout_df = splits.X_test.copy()
    holdout_df[spec.target_col] = splits.y_test.to_numpy()
    client = MlflowClient()

    decision = promote_champion_challenger_alias(
        client=client,
        uc_model_name=spec.model_fqn,
        candidate_version=exp_result.registered_model_version,
        candidate_metric=exp_result.holdout_metrics[spec.primary_holdout_metric],
        metric_key=spec.primary_holdout_metric,
        min_threshold=spec.min_threshold,
        higher_is_better=spec.higher_is_better,
        data_source=data_source,
        min_absolute_change=spec.min_absolute_change,
        candidate_eval_result=exp_result.holdout_eval_result,
        candidate_model_uri=exp_result.model_uri,
        holdout_df=holdout_df,
        feature_cols=spec.feature_cols,
        target_col=spec.target_col,
        task_type=spec.task_spec.task_type,
    )
    if exp_result.optimal_decision_threshold is not None:
        client.set_model_version_tag(
            name=spec.model_fqn,
            version=exp_result.registered_model_version,
            key="optimal_decision_threshold",
            value=f"{exp_result.optimal_decision_threshold:.4f}",
        )
    return decision


def execute_training_pipeline(
    spark: SparkSession,
    config: MLProjectConfig,
    spec: TrainingPipelineSpec,
    n_trials: int = 10,
    register_in_uc: bool = True,
    use_spark_optuna: bool = True,
    spark_parallelism: int = 4,
    allow_synthetic: bool = False,
    strict_leakage_check: bool = True,
    leakage_threshold: float = 0.98,
    leakage_detector: Callable[..., dict[str, float]] = detect_single_feature_leakage,
    early_stopping_rounds: int | None = 30,
    worker_n_jobs: int = 1,
    log_shap_artifacts: bool = True,
) -> tuple[ExperimentRunOutcome, PromotionDecision | None]:
    """Execute end-to-end training, leakage guarding, SHAP logging, UC tracing, and alias governance."""
    training_df, data_source, dataset_source_uri = _load_joined_training_df(
        spark=spark,
        spec=spec,
        allow_synthetic=allow_synthetic,
    )

    splits = spec.dataset_splitter(
        pdf=training_df.toPandas(),
        feature_cols=spec.feature_cols,
        target_col=spec.target_col,
        time_col=COL_SNAPSHOT_DATE,
    )
    _validate_no_single_feature_leakage(splits, spec, strict_leakage_check, leakage_threshold, leakage_detector)

    configure_uc_experiment_tracing(
        experiment_name=spec.experiment_path,
        catalog_name=config.catalog,
        schema_name=config.trace_schema,
        table_prefix=spec.trace_table_prefix,
        sql_warehouse_id=config.trace_sql_warehouse_id,
    )

    exp_result = run_xgb_experiment(
        spec=spec.task_spec,
        splits=splits,
        feature_cols=spec.feature_cols,
        n_trials=n_trials,
        registered_model_name=spec.model_fqn if register_in_uc else None,
        spark=spark if use_spark_optuna else None,
        spark_parallelism=spark_parallelism,
        worker_n_jobs=worker_n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        training_dataset_source=dataset_source_uri,
        target_col=spec.target_col,
        log_shap_artifacts=log_shap_artifacts,
    )
    log_feature_importance_artifact(
        feature_importance_df=exp_result.feature_importance_df,
        run_id=exp_result.final_run_id,
        artifact_filename=spec.feature_importance_filename,
    )

    promotion_decision: PromotionDecision | None = None
    if register_in_uc and exp_result.registered_model_version is not None:
        promotion_decision = _govern_registered_candidate(spec, splits, exp_result, data_source)

    if hasattr(mlflow, "flush_trace_async_logging"):
        mlflow.flush_trace_async_logging()

    return exp_result, promotion_decision
