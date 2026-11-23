"""Batch inference pipelines for `@Champion` XGBoost models on PySpark DataFrames.

Follows Palantir & Mikołaj Sędek Functional PySpark principles:
- Pure `Column -> Column` builders for business tiering (`churn_action_tier`, `affinity_tier`).
- Single-projection `.withColumns({...})` and explicit `.select()` output contracts.
- Historical prediction auditability via `scoring_date` (`DATE`) and idempotent Delta `replaceWhere`
  or `append` writes (avoiding destructive `overwriteSchema=true`).
- Supports both `mlflow.pyfunc.spark_udf` distributed batch scoring and Databricks
  `FeatureEngineeringClient.score_batch` automatic feature lookup scoring.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Final, Literal, cast

import mlflow
import pyspark.sql.functions as F
from mlflow.exceptions import MlflowException
from pyspark.sql import Column, DataFrame, SparkSession

from src.python.modules.utils.config import AFFINITY_FEATURE_COLUMNS, CHURN_FEATURE_COLUMNS
from src.python.modules.utils.metadata import attach_column_descriptions

_MISSING_ALIAS_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}
)


class _SparkUdfBenignWarningFilter(logging.Filter):
    """Filter out benign `env_manager='local'` and Databricks Serverless file endpoint probe warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for benign `env_manager='local'` and `resolving file endpoints` messages."""
        msg = record.getMessage()
        if 'Calling `spark_udf()` with `env_manager="local"`' in msg:
            return False
        return "resolving file endpoints" not in msg


def configure_silent_spark_udf_logging() -> None:
    """Disable experimental SDK file-endpoint probing and filter benign `spark_udf(env_manager='local')` warnings."""
    try:
        from databricks.sdk.config import Config

        Config.experimental_files_ext_enable_endpoints_api = False
    except Exception:
        pass

    benign_filter = _SparkUdfBenignWarningFilter()
    for logger_name in ("mlflow.pyfunc", "databricks.sdk.mixins.files"):
        target_logger = logging.getLogger(logger_name)
        if not any(isinstance(f, _SparkUdfBenignWarningFilter) for f in target_logger.filters):
            target_logger.addFilter(benign_filter)


def resolve_scoring_date(scoring_date: str | None = None) -> str:
    """Validate and return an ISO `YYYY-MM-DD` scoring date string (defaulting to UTC today)."""
    if scoring_date is None:
        return datetime.now(timezone.utc).date().isoformat()
    return date.fromisoformat(scoring_date).isoformat()


def resolve_scoring_model_uri(
    client: Any,
    uc_model_name: str,
    preferred_alias: str = "Champion",
    fallback_alias: str | None = None,
) -> tuple[str, str, str]:
    """Resolve an active Unity Catalog model URI and immutable version for batch scoring.

    Does NOT fall back to `@Challenger` unless `fallback_alias` is explicitly provided, preventing
    unvalidated models from silently entering production batch scoring.

    Returns:
        Tuple of `(model_uri, resolved_alias, resolved_version)`.

    Raises:
        LookupError: When neither `preferred_alias` nor an explicitly configured `fallback_alias`
            exists on `uc_model_name`.
    """
    candidate_aliases = (preferred_alias, fallback_alias) if fallback_alias else (preferred_alias,)
    for candidate_alias in candidate_aliases:
        try:
            mv = client.get_model_version_by_alias(uc_model_name, candidate_alias)
            return f"models:/{uc_model_name}@{candidate_alias}", candidate_alias, str(mv.version)
        except Exception as exc:
            if isinstance(exc, MlflowException) and str(getattr(exc, "error_code", "")) in _MISSING_ALIAS_ERROR_CODES:
                continue
            raise

    aliases_desc = ", ".join(f"@{a}" for a in candidate_aliases)
    raise LookupError(
        f"No registered model version found for '{uc_model_name}' under alias(es): {aliases_desc}. "
        "Train and promote a @Champion model before running batch inference."
    )


def _churn_action_tier_expr(prob_col: Column, threshold: float = 0.50) -> Column:
    """Classify predicted churn probability into an actionable retention tier."""
    return (
        F.when(prob_col >= F.lit(0.75), F.lit("HIGH_RISK_RETENTION"))
        .when(prob_col >= F.lit(threshold), F.lit("WATCHLIST"))
        .otherwise(F.lit("HEALTHY"))
    )


def _affinity_tier_expr(score_col: Column) -> Column:
    """Classify predicted product affinity score into a merchandising tier."""
    return (
        F.when(score_col >= F.lit(65.0), F.lit("HERO_PRODUCT"))
        .when(score_col >= F.lit(35.0), F.lit("CORE_ASSORTMENT"))
        .otherwise(F.lit("LONG_TAIL"))
    )


def format_churn_predictions_df(
    scored_df: DataFrame,
    model_uri: str,
    model_alias: str = "Champion",
    model_version: str = "unknown",
    raw_prediction_col: str = "raw_prediction",
    decision_threshold: float = 0.50,
    scoring_date: str | None = None,
) -> DataFrame:
    """Project a scored Customer Churn DataFrame into the governed Gold prediction contract.

    Handles both labeled evaluation sets (with `is_churn_candidate`) and unlabeled production
    scoring inputs (populating `is_churn_candidate` as typed `NULL`), and stamps `scoring_date`
    for historical auditability.
    """
    resolved_date = resolve_scoring_date(scoring_date)
    prob_col = F.col(raw_prediction_col).cast("double")
    actual_label_col = (
        F.col("is_churn_candidate").cast("int")
        if "is_churn_candidate" in scored_df.columns
        else F.lit(None).cast("int")
    )
    contract_cols = [
        "sk_user",
        "user_id",
        "is_churn_candidate",
        "predicted_churn_label",
        "predicted_churn_probability",
        "churn_action_tier",
        "model_uri",
        "model_alias",
        "model_version",
        "scoring_date",
        "scored_at_utc",
    ]
    projected_df = scored_df.withColumns(
        {
            "is_churn_candidate": actual_label_col,
            "predicted_churn_probability": F.round(prob_col, 6),
            "predicted_churn_label": F.when(prob_col >= F.lit(decision_threshold), F.lit(1))
            .otherwise(F.lit(0))
            .cast("int"),
            "churn_action_tier": _churn_action_tier_expr(prob_col, decision_threshold),
            "model_uri": F.lit(model_uri),
            "model_alias": F.lit(model_alias),
            "model_version": F.lit(model_version),
            "scoring_date": F.to_date(F.lit(resolved_date)),
            "scored_at_utc": F.current_timestamp(),
        }
    ).select(*contract_cols)
    return attach_column_descriptions(projected_df)


def format_affinity_predictions_df(
    scored_df: DataFrame,
    model_uri: str,
    model_alias: str = "Champion",
    model_version: str = "unknown",
    raw_prediction_col: str = "raw_prediction",
    scoring_date: str | None = None,
) -> DataFrame:
    """Project a scored Product Affinity DataFrame into the governed Gold prediction contract.

    Handles both labeled evaluation sets (with `affinity_index_score`) and unlabeled production
    scoring inputs (populating `affinity_index_score` and `affinity_residual` as typed `NULL`),
    and stamps `scoring_date` for historical auditability.
    """
    resolved_date = resolve_scoring_date(scoring_date)
    pred_score = F.col(raw_prediction_col).cast("double")
    has_actual = "affinity_index_score" in scored_df.columns
    actual_score = F.col("affinity_index_score").cast("double") if has_actual else F.lit(None).cast("double")
    residual_col = F.round(actual_score - pred_score, 4) if has_actual else F.lit(None).cast("double")
    contract_cols = [
        "sk_product",
        "product_id",
        "affinity_index_score",
        "predicted_affinity_score",
        "affinity_residual",
        "affinity_tier",
        "model_uri",
        "model_alias",
        "model_version",
        "scoring_date",
        "scored_at_utc",
    ]
    projected_df = scored_df.withColumns(
        {
            "affinity_index_score": actual_score,
            "predicted_affinity_score": F.round(pred_score, 4),
            "affinity_residual": residual_col,
            "affinity_tier": _affinity_tier_expr(pred_score),
            "model_uri": F.lit(model_uri),
            "model_alias": F.lit(model_alias),
            "model_version": F.lit(model_version),
            "scoring_date": F.to_date(F.lit(resolved_date)),
            "scored_at_utc": F.current_timestamp(),
        }
    ).select(*contract_cols)
    return attach_column_descriptions(projected_df)


def _apply_spark_udf_scoring(
    spark: SparkSession,
    features_df: DataFrame,
    model_uri: str,
    feature_cols: tuple[str, ...],
    env_manager: str = "local",
) -> DataFrame:
    """Invoke `mlflow.pyfunc.spark_udf` across `feature_cols` and attach `raw_prediction`."""
    configure_silent_spark_udf_logging()
    predict_udf = mlflow.pyfunc.spark_udf(
        spark,
        model_uri=model_uri,
        result_type="double",
        env_manager=env_manager,
    )
    feature_struct = F.struct(*[F.col(col_name).cast("double").alias(col_name) for col_name in feature_cols])
    return features_df.withColumns({"raw_prediction": predict_udf(feature_struct)})


def score_churn_with_spark_udf(
    spark: SparkSession,
    features_df: DataFrame,
    model_uri: str,
    feature_cols: tuple[str, ...] = CHURN_FEATURE_COLUMNS,
    model_alias: str = "Champion",
    model_version: str = "unknown",
    env_manager: str = "local",
    decision_threshold: float = 0.50,
    scoring_date: str | None = None,
) -> DataFrame:
    """Score customer churn features using `mlflow.pyfunc.spark_udf` and format predictions."""
    resolved_date = resolve_scoring_date(scoring_date)
    with mlflow.start_span(name="score_churn_with_spark_udf", span_type="CHAIN") as span:
        span.set_inputs(
            {
                "model_uri": model_uri,
                "model_alias": model_alias,
                "model_version": model_version,
                "n_features": len(feature_cols),
                "scoring_date": resolved_date,
                "decision_threshold": decision_threshold,
            }
        )
        scored_df = _apply_spark_udf_scoring(
            spark=spark,
            features_df=features_df,
            model_uri=model_uri,
            feature_cols=feature_cols,
            env_manager=env_manager,
        )
        formatted = format_churn_predictions_df(
            scored_df=scored_df,
            model_uri=model_uri,
            model_alias=model_alias,
            model_version=model_version,
            raw_prediction_col="raw_prediction",
            decision_threshold=decision_threshold,
            scoring_date=resolved_date,
        )
        span.set_outputs({"scoring_date": resolved_date, "output_columns": len(formatted.columns)})
        return formatted


def score_affinity_with_spark_udf(
    spark: SparkSession,
    features_df: DataFrame,
    model_uri: str,
    feature_cols: tuple[str, ...] = AFFINITY_FEATURE_COLUMNS,
    model_alias: str = "Champion",
    model_version: str = "unknown",
    env_manager: str = "local",
    scoring_date: str | None = None,
) -> DataFrame:
    """Score product affinity features using `mlflow.pyfunc.spark_udf` and format predictions."""
    resolved_date = resolve_scoring_date(scoring_date)
    with mlflow.start_span(name="score_affinity_with_spark_udf", span_type="CHAIN") as span:
        span.set_inputs(
            {
                "model_uri": model_uri,
                "model_alias": model_alias,
                "model_version": model_version,
                "n_features": len(feature_cols),
                "scoring_date": resolved_date,
            }
        )
        scored_df = _apply_spark_udf_scoring(
            spark=spark,
            features_df=features_df,
            model_uri=model_uri,
            feature_cols=feature_cols,
            env_manager=env_manager,
        )
        formatted = format_affinity_predictions_df(
            scored_df=scored_df,
            model_uri=model_uri,
            model_alias=model_alias,
            model_version=model_version,
            raw_prediction_col="raw_prediction",
            scoring_date=resolved_date,
        )
        span.set_outputs({"scoring_date": resolved_date, "output_columns": len(formatted.columns)})
        return formatted


def write_predictions_delta_table(
    spark: SparkSession,
    predictions_df: DataFrame,
    table_fqn: str,
    scoring_date: str | None = None,
    write_mode: Literal["replace_where", "append"] = "replace_where",
) -> str:
    """Persist batch predictions to Delta preserving historical `scoring_date` slices.

    - When the target table does not exist yet, creates it cleanly without `overwriteSchema`.
    - When `write_mode="replace_where"` (default), atomically replaces only the current
      `scoring_date` slice via Delta `replaceWhere`, making daily re-runs idempotent while
      retaining full historical prediction audit trails.
    - When `write_mode="append"`, appends the new scored rows directly.
    """
    resolved_date = resolve_scoring_date(scoring_date)
    if write_mode not in ("replace_where", "append"):
        raise ValueError(f"Unsupported prediction write_mode '{write_mode}'. Expected 'replace_where' or 'append'.")

    writer = predictions_df.write.format("delta")
    if not spark.catalog.tableExists(table_fqn):
        writer.mode("overwrite").saveAsTable(table_fqn)
    elif write_mode == "replace_where":
        writer.mode("overwrite").option("replaceWhere", f"scoring_date = DATE '{resolved_date}'").saveAsTable(table_fqn)
    else:
        writer.mode("append").saveAsTable(table_fqn)
    return resolved_date


def score_with_feature_store(
    fe_client: Any,
    keys_df: DataFrame,
    model_uri: str,
    result_type: str = "double",
) -> DataFrame:
    """Score entity primary keys via Databricks `FeatureEngineeringClient.score_batch`."""
    return cast(
        DataFrame,
        fe_client.score_batch(
            model_uri=model_uri,
            df=keys_df,
            result_type=result_type,
        ),
    )
