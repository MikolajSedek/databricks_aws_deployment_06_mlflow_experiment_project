# Databricks notebook source
"""Task 3: Batch Score `@Champion` Customer Churn & Product Affinity XGBoost Models.

Loads the `@Champion` models from Unity Catalog Model Registry (failing fast if absent unless
`--allow-challenger-fallback` is explicitly set), routes batch scoring trace spans to Unity
Catalog trace tables (`ml_traces`), scores the Unity Catalog Feature Tables in Spark via
`mlflow.pyfunc.spark_udf`, records immutable `model_version` and `scoring_date` lineage, attaches
column comments, and writes governed Delta prediction tables via idempotent `replaceWhere` (or
`append`) without destructive `overwriteSchema`:
- `ml_predictions.pred_customer_churn_xgboost`
- `ml_predictions.pred_product_affinity_xgboost`
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import click
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession

from src.python.modules.evaluation.uc_tracing import configure_uc_experiment_tracing
from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
)
from src.python.modules.features.synthetic_gold_views import (
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)
from src.python.modules.inference.batch_scoring import (
    configure_silent_spark_udf_logging,
    resolve_scoring_date,
    resolve_scoring_model_uri,
    score_affinity_with_spark_udf,
    score_churn_with_spark_udf,
    write_predictions_delta_table,
)
from src.python.modules.utils.config import MLProjectConfig


def _load_feature_table_for_scoring(
    spark: SparkSession,
    table_fqn: str,
    domain: str,
    allow_synthetic: bool,
) -> DataFrame:
    """Load a Unity Catalog Feature Table for batch scoring or fail fast unless `allow_synthetic=True`."""
    if spark.catalog.tableExists(table_fqn):
        return spark.read.table(table_fqn)
    if not allow_synthetic:
        raise FileNotFoundError(
            f"Required feature table '{table_fqn}' does not exist for batch scoring. "
            "Run prepare_uc_features first or pass --allow-synthetic for local/dev testing."
        )
    if domain == "churn":
        return build_sample_gold_churn_view_df(spark, num_users=80).transform(engineer_customer_churn_features)
    return build_sample_gold_affinity_view_df(spark, num_products=60).transform(engineer_product_affinity_features)


def _resolve_domain_model_target(
    client: MlflowClient,
    uc_model_name: str,
    preferred_alias: str,
    fallback_alias: str | None,
    uri_override: str | None,
) -> tuple[str, str, str]:
    """Return `(model_uri, resolved_alias, resolved_version)` from an override or strict UC alias lookup."""
    if uri_override is not None:
        return uri_override, preferred_alias, "override"
    return resolve_scoring_model_uri(
        client=client,
        uc_model_name=uc_model_name,
        preferred_alias=preferred_alias,
        fallback_alias=fallback_alias,
    )


def _score_and_persist_domain(
    spark: SparkSession,
    config: MLProjectConfig,
    features_df: DataFrame,
    experiment_path: str,
    trace_table_prefix: str,
    model_uri: str,
    model_alias: str,
    model_version: str,
    scorer_fn: Callable[..., DataFrame],
    predictions_table_fqn: str,
    scoring_date: str,
    write_mode: Literal["replace_where", "append"],
    persist_tables: bool,
) -> DataFrame:
    """Route scoring traces to UC, invoke the domain Spark UDF scorer, and optionally persist to Delta."""
    configure_uc_experiment_tracing(
        experiment_name=experiment_path,
        catalog_name=config.catalog,
        schema_name=config.trace_schema,
        table_prefix=trace_table_prefix,
        sql_warehouse_id=config.trace_sql_warehouse_id,
    )
    preds_df: DataFrame = scorer_fn(
        spark=spark,
        features_df=features_df,
        model_uri=model_uri,
        model_alias=model_alias,
        model_version=model_version,
        scoring_date=scoring_date,
    )
    if persist_tables:
        write_predictions_delta_table(
            spark=spark,
            predictions_df=preds_df,
            table_fqn=predictions_table_fqn,
            scoring_date=scoring_date,
            write_mode=write_mode,
        )
    return preds_df


def run_batch_scoring_pipeline(
    spark: SparkSession,
    config: MLProjectConfig,
    model_alias: str = "Champion",
    churn_model_uri_override: str | None = None,
    affinity_model_uri_override: str | None = None,
    persist_tables: bool = True,
    allow_synthetic: bool = False,
    allow_challenger_fallback: bool = False,
    scoring_date: str | None = None,
    write_mode: Literal["replace_where", "append"] = "replace_where",
) -> tuple[DataFrame, DataFrame]:
    """Score both Customer Churn and Product Affinity feature sets using governed MLflow models."""
    configure_silent_spark_udf_logging()
    resolved_date = resolve_scoring_date(scoring_date)

    # Validate feature table availability upfront before resolving or loading model artifacts.
    churn_features_df = _load_feature_table_for_scoring(spark, config.churn_feature_table_fqn, "churn", allow_synthetic)
    affinity_features_df = _load_feature_table_for_scoring(
        spark, config.affinity_feature_table_fqn, "affinity", allow_synthetic
    )

    fallback_alias = "Challenger" if allow_challenger_fallback else None
    client = MlflowClient()
    churn_uri, churn_alias, churn_version = _resolve_domain_model_target(
        client, config.churn_model_fqn, model_alias, fallback_alias, churn_model_uri_override
    )
    affinity_uri, affinity_alias, affinity_version = _resolve_domain_model_target(
        client, config.affinity_model_fqn, model_alias, fallback_alias, affinity_model_uri_override
    )

    churn_preds_df = _score_and_persist_domain(
        spark=spark,
        config=config,
        features_df=churn_features_df,
        experiment_path=config.churn_experiment_path,
        trace_table_prefix=config.churn_trace_table_prefix,
        model_uri=churn_uri,
        model_alias=churn_alias,
        model_version=churn_version,
        scorer_fn=score_churn_with_spark_udf,
        predictions_table_fqn=config.churn_predictions_table_fqn,
        scoring_date=resolved_date,
        write_mode=write_mode,
        persist_tables=persist_tables,
    )
    affinity_preds_df = _score_and_persist_domain(
        spark=spark,
        config=config,
        features_df=affinity_features_df,
        experiment_path=config.affinity_experiment_path,
        trace_table_prefix=config.affinity_trace_table_prefix,
        model_uri=affinity_uri,
        model_alias=affinity_alias,
        model_version=affinity_version,
        scorer_fn=score_affinity_with_spark_udf,
        predictions_table_fqn=config.affinity_predictions_table_fqn,
        scoring_date=resolved_date,
        write_mode=write_mode,
        persist_tables=persist_tables,
    )

    if hasattr(mlflow, "flush_trace_async_logging"):
        mlflow.flush_trace_async_logging()

    return churn_preds_df, affinity_preds_df


@click.command(
    name="batch-score-predictions",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--catalog", default="data_engineering_dev", show_default=True, help="Target Unity Catalog name.")
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
    help="Databricks workspace MLflow experiment path for Customer Churn trace spans.",
)
@click.option(
    "--affinity-experiment-path",
    default="/Shared/mlflow_xgboost_product_affinity_uc",
    show_default=True,
    help="Databricks workspace MLflow experiment path for Product Affinity trace spans.",
)
@click.option(
    "--trace-warehouse-id",
    default=None,
    envvar="MLFLOW_TRACING_SQL_WAREHOUSE_ID",
    help="Optional Databricks SQL Warehouse ID for UC trace storage queries.",
)
@click.option("--alias", default="Champion", show_default=True, help="Model Registry alias to score.")
@click.option(
    "--scoring-date",
    default=None,
    help="Optional ISO YYYY-MM-DD scoring date (defaults to UTC today) for replaceWhere partitioning.",
)
@click.option(
    "--write-mode",
    type=click.Choice(["replace_where", "append"]),
    default="replace_where",
    show_default=True,
    help="Historical Delta write mode preserving prior scoring_date slices.",
)
@click.option(
    "--allow-synthetic/--no-allow-synthetic",
    default=False,
    show_default=True,
    help="Permit synthetic features when UC Feature Tables are absent (dev/test only).",
)
@click.option(
    "--allow-challenger-fallback/--no-allow-challenger-fallback",
    default=False,
    show_default=True,
    help="Explicitly allow falling back to @Challenger when @Champion is absent (dev/test only).",
)
def cli(
    catalog: str,
    feature_schema: str,
    model_schema: str,
    predictions_schema: str,
    trace_schema: str,
    churn_experiment_path: str,
    affinity_experiment_path: str,
    trace_warehouse_id: str | None,
    alias: str,
    scoring_date: str | None,
    write_mode: Literal["replace_where", "append"],
    allow_synthetic: bool,
    allow_challenger_fallback: bool,
) -> tuple[DataFrame, DataFrame]:
    """Click CLI command executing distributed Spark batch inference with @Champion models."""
    config = MLProjectConfig(
        catalog=catalog,
        feature_schema=feature_schema,
        model_schema=model_schema,
        predictions_schema=predictions_schema,
        trace_schema=trace_schema,
        churn_experiment_path=churn_experiment_path,
        affinity_experiment_path=affinity_experiment_path,
        trace_sql_warehouse_id=trace_warehouse_id or None,
    )
    mlflow.set_registry_uri("databricks-uc")
    spark = SparkSession.builder.getOrCreate()
    return run_batch_scoring_pipeline(
        spark=spark,
        config=config,
        model_alias=alias,
        persist_tables=True,
        allow_synthetic=allow_synthetic,
        allow_challenger_fallback=allow_challenger_fallback,
        scoring_date=scoring_date,
        write_mode=write_mode,
    )


def main(argv: Sequence[str] | None = None) -> tuple[DataFrame, DataFrame]:
    """Programmatic and Databricks entrypoint invoking the Click CLI with `standalone_mode=False`."""
    result: Any = cli.main(
        args=list(argv) if argv is not None else None,
        standalone_mode=False,
    )
    return result  # type: ignore[no-any-return]


if __name__ == "__main__":
    main()
