"""MLflow 3 holdout evaluation (`mlflow.models.evaluate`), validation gating, and UC alias governance.

Implements apples-to-apples candidate vs. `@Champion` evaluation on the same holdout dataset using
`mlflow.models.evaluate()` and `mlflow.validate_evaluation_results()` with `MetricThreshold`
(`threshold` + `min_absolute_change`), lightweight scalar-only tracing via `mlflow.start_span()`,
and deterministic alias assignment (`@Champion`, `@Challenger`, `@PreviousChampion`) in Unity Catalog.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import EvaluationResult, MetricThreshold, make_metric
from mlflow.models.evaluation.validation import ModelValidationFailedException
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# MLflow error codes that genuinely represent an absent model/alias/run rather than a transient API failure.
_MISSING_RESOURCE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}
)
_PREDICTION_COL: Final[str] = "__mlflow_eval_prediction__"


def _is_missing_registry_entity_error(exc: Exception) -> bool:
    """Return True if `exc` indicates a missing MLflow model, alias, or run rather than a transient failure."""
    if isinstance(exc, MlflowException):
        return str(getattr(exc, "error_code", "")) in _MISSING_RESOURCE_ERROR_CODES
    return False


@dataclass(frozen=True)
class PromotionDecision:
    """Immutable record of a Unity Catalog Champion / Challenger alias promotion decision."""

    model_name: str
    candidate_version: str
    assigned_alias: str
    candidate_metric: float
    previous_champion_version: str | None
    previous_champion_metric: float | None
    promoted_to_champion: bool
    champion_re_evaluated: bool = False


def _build_classification_eval_metrics() -> list[Any]:
    """Create MLflow 3 `EvaluationMetric` callables for continuous-probability binary classification."""

    def _roc_auc(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        y_true = eval_df["target"].astype(int)
        # MLflow runs a 1-row smoke test on custom metrics before full evaluation; guard single-class slices.
        if y_true.nunique() < 2:
            return 0.5
        return round(float(roc_auc_score(y_true, eval_df["prediction"].astype(float))), 6)

    def _pr_auc(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        y_true = eval_df["target"].astype(int)
        if y_true.nunique() < 2:
            return round(float(y_true.mean()), 6)
        return round(float(average_precision_score(y_true, eval_df["prediction"].astype(float))), 6)

    def _brier_score(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        probs = eval_df["prediction"].astype(float).clip(0.0, 1.0)
        return round(float(brier_score_loss(eval_df["target"].astype(int), probs)), 6)

    def _f1(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        y_pred = (eval_df["prediction"].astype(float) >= 0.5).astype(int)
        return round(float(f1_score(eval_df["target"].astype(int), y_pred, zero_division=0)), 6)

    def _precision(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        y_pred = (eval_df["prediction"].astype(float) >= 0.5).astype(int)
        return round(float(precision_score(eval_df["target"].astype(int), y_pred, zero_division=0)), 6)

    def _recall(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        y_pred = (eval_df["prediction"].astype(float) >= 0.5).astype(int)
        return round(float(recall_score(eval_df["target"].astype(int), y_pred, zero_division=0)), 6)

    def _log_loss(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        return round(
            float(log_loss(eval_df["target"].astype(int), eval_df["prediction"].astype(float), labels=[0, 1])),
            6,
        )

    return [
        make_metric(eval_fn=_roc_auc, greater_is_better=True, name="holdout_roc_auc"),
        make_metric(eval_fn=_pr_auc, greater_is_better=True, name="holdout_pr_auc"),
        make_metric(eval_fn=_brier_score, greater_is_better=False, name="holdout_brier_score"),
        make_metric(eval_fn=_f1, greater_is_better=True, name="holdout_f1"),
        make_metric(eval_fn=_precision, greater_is_better=True, name="holdout_precision"),
        make_metric(eval_fn=_recall, greater_is_better=True, name="holdout_recall"),
        make_metric(eval_fn=_log_loss, greater_is_better=False, name="holdout_log_loss"),
    ]


def _build_regression_eval_metrics() -> list[Any]:
    """Create MLflow 3 `EvaluationMetric` callables for continuous regression holdout evaluation."""

    def _rmse(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        return round(math.sqrt(float(mean_squared_error(eval_df["target"], eval_df["prediction"]))), 6)

    def _mae(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        return round(float(mean_absolute_error(eval_df["target"], eval_df["prediction"])), 6)

    def _r2(eval_df: pd.DataFrame, _builtin: dict[str, Any]) -> float:
        # Guard against MLflow's 1-row metric smoke test where R2 is mathematically undefined.
        if len(eval_df) < 2:
            return 0.0
        return round(float(r2_score(eval_df["target"], eval_df["prediction"])), 6)

    return [
        make_metric(eval_fn=_rmse, greater_is_better=False, name="holdout_rmse"),
        make_metric(eval_fn=_mae, greater_is_better=False, name="holdout_mae"),
        make_metric(eval_fn=_r2, greater_is_better=True, name="holdout_r2"),
    ]


def evaluate_model_on_holdout(
    model_or_uri: str | Callable[[pd.DataFrame], Any],
    holdout_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    task_type: Literal["classification", "regression"],
    model_id: str | None = None,
) -> EvaluationResult:
    """Evaluate a logged MLflow model URI or callable on a holdout DataFrame using `mlflow.models.evaluate()`.

    Computes consistent domain metrics (`holdout_roc_auc`/`holdout_pr_auc`/`holdout_brier_score`/
    `holdout_f1`/`holdout_log_loss` for continuous-probability classifiers, `holdout_rmse`/
    `holdout_mae`/`holdout_r2` for regressors) and links evaluation results directly to `model_id`
    when an MLflow 3 `LoggedModel` ID is supplied.
    """
    features_df = holdout_df[list(feature_cols)].astype(float)
    if isinstance(model_or_uri, str):
        loaded = mlflow.pyfunc.load_model(model_or_uri)
        preds = loaded.predict(features_df)
    else:
        preds = model_or_uri(features_df)

    # Precompute predictions into a dedicated column so custom PyFunc probability outputs work uniformly.
    eval_data = holdout_df[[*feature_cols, target_col]].copy()
    eval_data[_PREDICTION_COL] = pd.Series(preds, index=eval_data.index, dtype="float64")

    extra_metrics = (
        _build_classification_eval_metrics() if task_type == "classification" else _build_regression_eval_metrics()
    )
    eval_kwargs: dict[str, Any] = {
        "data": eval_data,
        "targets": target_col,
        "predictions": _PREDICTION_COL,
        "extra_metrics": extra_metrics,
        # Dedicated TreeSHAP artifacts are generated in `shap_explainability.py`; disable MLflow's KernelSHAP default.
        "evaluator_config": {"log_model_explainability": False},
    }
    if model_id is not None:
        eval_kwargs["model_id"] = model_id

    eval_result: EvaluationResult = mlflow.models.evaluate(**eval_kwargs)
    return eval_result


def validate_candidate_evaluation(
    candidate_eval: EvaluationResult,
    champion_eval: EvaluationResult | None,
    metric_key: str,
    min_threshold: float,
    min_absolute_change: float = 0.0,
    higher_is_better: bool = True,
) -> bool:
    """Validate candidate vs. optional baseline `EvaluationResult` via `mlflow.validate_evaluation_results()`."""
    threshold_kwargs: dict[str, Any] = {
        "threshold": min_threshold,
        "greater_is_better": higher_is_better,
    }
    # MLflow rejects `min_absolute_change` when no baseline exists (first-ever model); attach only for comparisons.
    if champion_eval is not None and min_absolute_change > 0.0:
        threshold_kwargs["min_absolute_change"] = min_absolute_change

    thresholds = {metric_key: MetricThreshold(**threshold_kwargs)}
    try:
        mlflow.validate_evaluation_results(
            validation_thresholds=thresholds,
            candidate_result=candidate_eval,
            baseline_result=champion_eval,
        )
        return True
    except ModelValidationFailedException as exc:
        logger.info("Candidate failed MLflow validation gate on '%s': %s", metric_key, exc)
        return False


def log_feature_importance_artifact(
    feature_importance_df: pd.DataFrame,
    run_id: str | None = None,
    artifact_filename: str = "feature_importance.csv",
) -> str:
    """Write an XGBoost feature importance table to CSV and log it as an MLflow artifact."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / artifact_filename
        feature_importance_df.to_csv(csv_path, index=False)

        if run_id is not None:
            client = MlflowClient()
            client.log_artifact(run_id, str(csv_path), artifact_path="feature_importance")
        elif mlflow.active_run() is not None:
            mlflow.log_artifact(str(csv_path), artifact_path="feature_importance")

        return str(csv_path)


def _fetch_champion_version_and_metric(
    client: Any,
    uc_model_name: str,
    metric_key: str,
    champion_alias: str = "Champion",
) -> tuple[str | None, float | None]:
    """Inspect the current `@Champion` version and its logged metric if an alias exists."""
    try:
        current_champion = client.get_model_version_by_alias(uc_model_name, champion_alias)
    except Exception as exc:
        if _is_missing_registry_entity_error(exc):
            return None, None
        raise

    champ_version = str(current_champion.version)
    champ_run_id = getattr(current_champion, "run_id", None)
    if not champ_run_id:
        return champ_version, None

    try:
        champ_run = client.get_run(champ_run_id)
        raw_metric = champ_run.data.metrics.get(metric_key)
        return champ_version, (float(raw_metric) if raw_metric is not None else None)
    except Exception as exc:
        if _is_missing_registry_entity_error(exc):
            return champ_version, None
        raise


def _does_candidate_outperform(
    candidate_metric: float,
    previous_metric: float | None,
    min_threshold: float,
    higher_is_better: bool,
    min_absolute_change: float = 0.0,
) -> bool:
    """Evaluate whether a candidate metric passes the absolute quality gate and beats the champion.

    The absolute `min_threshold` gate is ALWAYS enforced first — even when no `@Champion` exists
    yet — so an underperforming baseline model can never be promoted to `@Champion`.
    """
    passes_gate = candidate_metric >= min_threshold if higher_is_better else candidate_metric <= min_threshold
    if not passes_gate:
        return False

    if previous_metric is None:
        return True

    if higher_is_better:
        return candidate_metric >= (previous_metric + min_absolute_change)
    return candidate_metric <= (previous_metric - min_absolute_change)


def _evaluate_candidate_against_champion_on_holdout(
    uc_model_name: str,
    prev_version: str | None,
    prev_metric: float | None,
    metric_key: str,
    min_threshold: float,
    min_absolute_change: float,
    higher_is_better: bool,
    candidate_eval_result: EvaluationResult | None,
    candidate_model_uri: str | None,
    holdout_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    task_type: Literal["classification", "regression"],
) -> tuple[bool, float, float | None, bool]:
    """Re-evaluate candidate and reigning `@Champion` on the same holdout and run MLflow validation gating."""
    cand_eval = candidate_eval_result
    if cand_eval is None and candidate_model_uri is not None:
        cand_eval = evaluate_model_on_holdout(
            model_or_uri=candidate_model_uri,
            holdout_df=holdout_df,
            feature_cols=feature_cols,
            target_col=target_col,
            task_type=task_type,
        )
    assert cand_eval is not None  # nosec B101
    effective_candidate_metric = float(cand_eval.metrics[metric_key])

    champ_eval: EvaluationResult | None = None
    champion_re_evaluated = False
    effective_prev_metric = prev_metric
    if prev_version is not None:
        champion_uri = f"models:/{uc_model_name}@Champion"
        try:
            champ_eval = evaluate_model_on_holdout(
                model_or_uri=champion_uri,
                holdout_df=holdout_df,
                feature_cols=feature_cols,
                target_col=target_col,
                task_type=task_type,
            )
            effective_prev_metric = float(champ_eval.metrics[metric_key])
            champion_re_evaluated = True
        except Exception as exc:
            # Older @Champion artifacts may have incompatible signatures; warn and fall back to logged metrics.
            logger.warning(
                "Unable to re-evaluate existing @Champion (%s) on current holdout; "
                "falling back to logged metric (%s): %s",
                champion_uri,
                prev_metric,
                exc,
            )

    if champ_eval is not None or effective_prev_metric is None:
        should_promote = validate_candidate_evaluation(
            candidate_eval=cand_eval,
            champion_eval=champ_eval,
            metric_key=metric_key,
            min_threshold=min_threshold,
            min_absolute_change=min_absolute_change,
            higher_is_better=higher_is_better,
        )
    else:
        should_promote = _does_candidate_outperform(
            candidate_metric=effective_candidate_metric,
            previous_metric=effective_prev_metric,
            min_threshold=min_threshold,
            higher_is_better=higher_is_better,
            min_absolute_change=min_absolute_change,
        )

    return should_promote, effective_candidate_metric, effective_prev_metric, champion_re_evaluated


def _apply_registry_alias_and_tags(
    client: Any,
    uc_model_name: str,
    candidate_version: str,
    prev_version: str | None,
    should_promote: bool,
    is_synthetic: bool,
    data_source: str,
    champion_re_evaluated: bool,
    metric_key: str,
    effective_candidate_metric: float,
) -> str:
    """Assign `@Champion`/`@PreviousChampion` or `@Challenger` alias and audit tags in Unity Catalog."""
    if should_promote:
        if prev_version is not None and prev_version != candidate_version:
            client.set_registered_model_alias(uc_model_name, "PreviousChampion", prev_version)
        client.set_registered_model_alias(uc_model_name, "Champion", candidate_version)
        assigned_alias = "Champion"
        validation_status = "approved"
    else:
        client.set_registered_model_alias(uc_model_name, "Challenger", candidate_version)
        assigned_alias = "Challenger"
        validation_status = "synthetic_blocked" if is_synthetic else "challenger_review"

    audit_tags = {
        "validation_status": validation_status,
        "data_source": data_source,
        "champion_re_evaluated": str(champion_re_evaluated).lower(),
        metric_key: f"{effective_candidate_metric:.6f}",
    }
    for tag_key, tag_val in audit_tags.items():
        client.set_model_version_tag(name=uc_model_name, version=candidate_version, key=tag_key, value=tag_val)

    return assigned_alias


def promote_champion_challenger_alias(
    client: Any,
    uc_model_name: str,
    candidate_version: str,
    candidate_metric: float,
    metric_key: str,
    min_threshold: float,
    higher_is_better: bool = True,
    data_source: str = "gold",
    min_absolute_change: float = 0.0,
    candidate_eval_result: EvaluationResult | None = None,
    candidate_model_uri: str | None = None,
    holdout_df: pd.DataFrame | None = None,
    feature_cols: tuple[str, ...] | None = None,
    target_col: str | None = None,
    task_type: Literal["classification", "regression"] | None = None,
) -> PromotionDecision:
    """Assign `@Champion` or `@Challenger` alias in Unity Catalog with optional same-holdout re-evaluation.

    Promotion precedence:
    1. Models trained on synthetic data are unconditionally blocked from `@Champion`.
    2. When holdout inputs and a candidate URI/`EvaluationResult` are provided, candidate and
       reigning `@Champion` are scored on the exact same holdout via `mlflow.models.evaluate()`.
    3. Otherwise, compares scalar holdout metrics against `min_threshold` and `min_absolute_change`.
    """
    cand_ver = str(candidate_version)
    with mlflow.start_span(name="promote_champion_challenger_alias", span_type="CHAIN") as span:
        span.set_inputs(
            {
                "uc_model_name": uc_model_name,
                "candidate_version": cand_ver,
                "candidate_metric": float(candidate_metric),
                "metric_key": metric_key,
                "min_threshold": float(min_threshold),
                "min_absolute_change": float(min_absolute_change),
                "higher_is_better": higher_is_better,
                "data_source": data_source,
                "holdout_rows": len(holdout_df) if holdout_df is not None else 0,
            }
        )

        prev_version, prev_metric = _fetch_champion_version_and_metric(client, uc_model_name, metric_key)
        is_synthetic = data_source.strip().lower() == "synthetic"
        champion_re_evaluated = False
        effective_candidate_metric = float(candidate_metric)

        has_shared_holdout = (
            holdout_df is not None
            and feature_cols is not None
            and target_col is not None
            and task_type is not None
            and (candidate_eval_result is not None or candidate_model_uri is not None)
        )

        if is_synthetic:
            should_promote = False
        elif has_shared_holdout and holdout_df is not None and feature_cols and target_col and task_type:
            (
                should_promote,
                effective_candidate_metric,
                prev_metric,
                champion_re_evaluated,
            ) = _evaluate_candidate_against_champion_on_holdout(
                uc_model_name=uc_model_name,
                prev_version=prev_version,
                prev_metric=prev_metric,
                metric_key=metric_key,
                min_threshold=min_threshold,
                min_absolute_change=min_absolute_change,
                higher_is_better=higher_is_better,
                candidate_eval_result=candidate_eval_result,
                candidate_model_uri=candidate_model_uri,
                holdout_df=holdout_df,
                feature_cols=feature_cols,
                target_col=target_col,
                task_type=task_type,
            )
        else:
            should_promote = _does_candidate_outperform(
                candidate_metric=effective_candidate_metric,
                previous_metric=prev_metric,
                min_threshold=min_threshold,
                higher_is_better=higher_is_better,
                min_absolute_change=min_absolute_change,
            )

        assigned_alias = _apply_registry_alias_and_tags(
            client=client,
            uc_model_name=uc_model_name,
            candidate_version=cand_ver,
            prev_version=prev_version,
            should_promote=should_promote,
            is_synthetic=is_synthetic,
            data_source=data_source,
            champion_re_evaluated=champion_re_evaluated,
            metric_key=metric_key,
            effective_candidate_metric=effective_candidate_metric,
        )

        decision = PromotionDecision(
            model_name=uc_model_name,
            candidate_version=cand_ver,
            assigned_alias=assigned_alias,
            candidate_metric=effective_candidate_metric,
            previous_champion_version=prev_version,
            previous_champion_metric=prev_metric,
            promoted_to_champion=should_promote,
            champion_re_evaluated=champion_re_evaluated,
        )
        span.set_outputs(
            {
                "assigned_alias": decision.assigned_alias,
                "promoted_to_champion": decision.promoted_to_champion,
                "champion_re_evaluated": decision.champion_re_evaluated,
                "candidate_metric": decision.candidate_metric,
                "previous_champion_version": decision.previous_champion_version,
                "previous_champion_metric": decision.previous_champion_metric,
            }
        )
        return decision
