# Databricks notebook source
"""Task 1: Prepare Unity Catalog Feature Tables & Label Spines from Project 05 Kimball Gold ML views.

Provisions Unity Catalog ML schemas (`ml_features`, `ml_models`, `ml_predictions`, `ml_traces`),
optionally migrates legacy experiment traces to Unity Catalog OpenTelemetry tables when
`--migrate-uc-traces` is explicitly requested (one-off migration decoupled from scheduled runs),
reads `gold.vw_ml_customer_churn_features` and `gold.vw_ml_product_affinity_features` (failing
fast when absent unless `--allow-synthetic` is set), applies pure functional PySpark feature
engineering, publishes label-free timeseries-keyed Feature Tables (`fe_*`) keyed by
`(entity_id, snapshot_date)`, and persists dedicated label spine tables (`lbl_*`) separately.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import click
from pyspark.sql import DataFrame, SparkSession

from src.python.modules.evaluation.uc_tracing import migrate_experiment_traces_to_uc
from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
    extract_customer_churn_labels,
    extract_product_affinity_labels,
    sync_label_table,
    sync_uc_feature_table,
)
from src.python.modules.features.synthetic_gold_views import (
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)
from src.python.modules.utils.config import (
    COL_PRODUCT_ID,
    COL_SNAPSHOT_DATE,
    COL_USER_ID,
    MLProjectConfig,
)
from src.python.modules.utils.metadata import build_schema_provisioning_ddls


def _load_or_seed_gold_view(
    spark: SparkSession,
    view_fqn: str,
    synthetic_builder: Callable[[SparkSession], DataFrame],
    allow_synthetic: bool = False,
) -> DataFrame:
    """Load an upstream Project 05 Gold view or fail fast unless `allow_synthetic=True`."""
    if spark.catalog.tableExists(view_fqn):
        return spark.read.table(view_fqn)
    if not allow_synthetic:
        raise FileNotFoundError(
            f"Upstream Gold view '{view_fqn}' does not exist. "
            "Verify Project 05 deployment or pass --allow-synthetic for local/dev testing."
        )
    return synthetic_builder(spark)


def _try_get_fe_client() -> Any | None:
    """Attempt to instantiate Databricks `FeatureEngineeringClient` when running on Databricks."""
    try:
        from databricks.feature_engineering import FeatureEngineeringClient

        return FeatureEngineeringClient()
    except Exception:
        return None


def _migrate_legacy_traces(config: MLProjectConfig) -> None:
    """Run one-off legacy trace migration into UC OpenTelemetry tables for both ML domains."""
    domain_migrations = (
        (config.legacy_churn_experiment_path, config.churn_experiment_path, config.churn_trace_table_prefix),
        (config.legacy_affinity_experiment_path, config.affinity_experiment_path, config.affinity_trace_table_prefix),
    )
    for source_exp, target_exp, table_prefix in domain_migrations:
        migrate_experiment_traces_to_uc(
            source_experiment_name=source_exp,
            target_experiment_name=target_exp,
            catalog_name=config.catalog,
            schema_name=config.trace_schema,
            table_prefix=table_prefix,
            sql_warehouse_id=config.trace_sql_warehouse_id,
        )


def _prepare_and_sync_domain(
    spark: SparkSession,
    raw_df: DataFrame,
    feature_transformer: Callable[[DataFrame], DataFrame],
    label_extractor: Callable[[DataFrame], DataFrame],
    feature_table_fqn: str,
    label_table_fqn: str,
    primary_key: str,
    description: str,
    fe_client: Any | None,
) -> DataFrame:
    """Transform one Gold ML view into a label-free UC Feature Table and separate label spine."""
    features_df = raw_df.transform(feature_transformer)
    labels_df = raw_df.transform(label_extractor)

    # Publish features and labels into separate tables so Feature Store tables stay strictly label-free.
    sync_uc_feature_table(
        spark=spark,
        features_df=features_df,
        table_fqn=feature_table_fqn,
        primary_key=primary_key,
        description=description,
        fe_client=fe_client,
        timeseries_columns=[COL_SNAPSHOT_DATE],
    )
    sync_label_table(labels_df=labels_df, table_fqn=label_table_fqn)
    return features_df


def run_feature_preparation_pipeline(
    spark: SparkSession,
    config: MLProjectConfig,
    fe_client: Any | None = None,
    migrate_uc_traces: bool = False,
    allow_synthetic: bool = False,
) -> tuple[DataFrame, DataFrame]:
    """Execute schema provisioning, optional UC trace migration, feature engineering, and table sync."""
    for ddl in build_schema_provisioning_ddls(config):
        spark.sql(ddl)

    if migrate_uc_traces:
        _migrate_legacy_traces(config)

    raw_churn_df = _load_or_seed_gold_view(
        spark=spark,
        view_fqn=config.upstream_churn_view_fqn,
        synthetic_builder=lambda spk: build_sample_gold_churn_view_df(spk, num_users=240),
        allow_synthetic=allow_synthetic,
    )
    raw_affinity_df = _load_or_seed_gold_view(
        spark=spark,
        view_fqn=config.upstream_affinity_view_fqn,
        synthetic_builder=lambda spk: build_sample_gold_affinity_view_df(spk, num_products=160),
        allow_synthetic=allow_synthetic,
    )

    active_fe = fe_client if fe_client is not None else _try_get_fe_client()
    churn_features_df = _prepare_and_sync_domain(
        spark=spark,
        raw_df=raw_churn_df,
        feature_transformer=engineer_customer_churn_features,
        label_extractor=extract_customer_churn_labels,
        feature_table_fqn=config.churn_feature_table_fqn,
        label_table_fqn=config.churn_label_table_fqn,
        primary_key=COL_USER_ID,
        description="Label-free Customer Churn XGBoost Feature Table keyed by (user_id, snapshot_date).",
        fe_client=active_fe,
    )
    affinity_features_df = _prepare_and_sync_domain(
        spark=spark,
        raw_df=raw_affinity_df,
        feature_transformer=engineer_product_affinity_features,
        label_extractor=extract_product_affinity_labels,
        feature_table_fqn=config.affinity_feature_table_fqn,
        label_table_fqn=config.affinity_label_table_fqn,
        primary_key=COL_PRODUCT_ID,
        description="Label-free Product Affinity XGBoost Feature Table keyed by (product_id, snapshot_date).",
        fe_client=active_fe,
    )
    return churn_features_df, affinity_features_df


@click.command(
    name="prepare-uc-features",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--catalog", default="data_engineering_dev", show_default=True, help="Target Unity Catalog name.")
@click.option("--gold-schema", default="gold", show_default=True, help="Upstream Kimball Gold schema name.")
@click.option("--feature-schema", default="ml_features", show_default=True, help="Unity Catalog Feature Store schema.")
@click.option("--model-schema", default="ml_models", show_default=True, help="Unity Catalog Model Registry schema.")
@click.option(
    "--predictions-schema",
    default="ml_predictions",
    show_default=True,
    help="Unity Catalog batch predictions schema.",
)
@click.option(
    "--trace-schema", default="ml_traces", show_default=True, help="Unity Catalog OpenTelemetry trace schema."
)
@click.option(
    "--churn-experiment-path",
    default="/Shared/mlflow_xgboost_customer_churn_uc",
    show_default=True,
    help="Databricks workspace MLflow experiment path for Customer Churn.",
)
@click.option(
    "--affinity-experiment-path",
    default="/Shared/mlflow_xgboost_product_affinity_uc",
    show_default=True,
    help="Databricks workspace MLflow experiment path for Product Affinity.",
)
@click.option(
    "--trace-warehouse-id",
    default=None,
    envvar="MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    help="Optional Databricks SQL Warehouse ID for UC trace storage queries.",
)
@click.option(
    "--migrate-uc-traces/--no-migrate-uc-traces",
    default=False,
    show_default=True,
    help="Run one-off legacy experiment trace migration into Unity Catalog OpenTelemetry tables.",
)
@click.option(
    "--allow-synthetic/--no-allow-synthetic",
    default=False,
    show_default=True,
    help="Permit deterministic synthetic seeding when upstream Gold views are absent (dev/test only).",
)
def cli(
    catalog: str,
    gold_schema: str,
    feature_schema: str,
    model_schema: str,
    predictions_schema: str,
    trace_schema: str,
    churn_experiment_path: str,
    affinity_experiment_path: str,
    trace_warehouse_id: str | None,
    migrate_uc_traces: bool,
    allow_synthetic: bool,
) -> MLProjectConfig:
    """Click CLI command preparing Unity Catalog ML Feature Tables, label spines, and optional UC trace migration."""
    config = MLProjectConfig(
        catalog=catalog,
        upstream_gold_schema=gold_schema,
        feature_schema=feature_schema,
        model_schema=model_schema,
        predictions_schema=predictions_schema,
        trace_schema=trace_schema,
        churn_experiment_path=churn_experiment_path,
        affinity_experiment_path=affinity_experiment_path,
        trace_sql_warehouse_id=trace_warehouse_id or None,
    )
    spark = SparkSession.builder.getOrCreate()
    run_feature_preparation_pipeline(
        spark=spark,
        config=config,
        migrate_uc_traces=migrate_uc_traces,
        allow_synthetic=allow_synthetic,
    )
    return config


def main(argv: Sequence[str] | None = None) -> MLProjectConfig:
    """Programmatic and Databricks entrypoint invoking the Click CLI with `standalone_mode=False`."""
    result: MLProjectConfig = cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    return result


if __name__ == "__main__":
    main()
