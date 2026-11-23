"""Optuna hyperparameter optimization, SHAP explainability, and MLflow 3 experiment tracking for XGBoost."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas as mlflow_from_pandas
from mlflow.models import EvaluationResult, infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
from pyspark.sql import SparkSession
from xgboost import XGBClassifier, XGBModel, XGBRegressor

from src.python.modules.evaluation.model_governance import evaluate_model_on_holdout
from src.python.modules.evaluation.shap_explainability import (
    ShapExplainabilityResult,
    log_shap_explainability_artifacts,
)
from src.python.modules.training.spark_optuna import (
    compute_optimal_classification_threshold,
    execute_optuna_xgb_study,
)
from src.python.modules.training.xgboost_datasets import (
    DatasetSplits,
    compute_scale_pos_weight,
    extract_feature_importance_df,
)
from src.python.modules.utils.config import TARGET_AFFINITY_COL, TARGET_CHURN_COL


@dataclass(frozen=True)
class ModelTaskSpec:
    """Declarative specification unifying XGBoost classification and regression experiment lifecycles."""

    task_type: Literal["classification", "regression"]
    dataset_name_prefix: str
    default_target_col: str
    root_span_name: str
    hpo_span_name: str
    study_run_name: str
    final_span_name: str
    final_run_name: str
    task_tag: str
    direction: Literal["minimize", "maximize"]
    val_metric_name: str
    objective: str
    eval_metric: str
    shap_artifact_prefix: str


CHURN_TASK_SPEC: Final[ModelTaskSpec] = ModelTaskSpec(
    task_type="classification",
    dataset_name_prefix="customer_churn",
    default_target_col=TARGET_CHURN_COL,
    root_span_name="run_churn_xgb_classifier_experiment",
    hpo_span_name="optuna_churn_hpo_study",
    study_run_name="optuna_xgb_churn_study",
    final_span_name="train_final_churn_xgb_candidate",
    final_run_name="xgb_churn_final_candidate",
    task_tag="churn_classification",
    direction="maximize",
    val_metric_name="val_roc_auc",
    objective="binary:logistic",
    eval_metric="logloss",
    shap_artifact_prefix="churn_shap",
)

AFFINITY_TASK_SPEC: Final[ModelTaskSpec] = ModelTaskSpec(
    task_type="regression",
    dataset_name_prefix="product_affinity",
    default_target_col=TARGET_AFFINITY_COL,
    root_span_name="run_affinity_xgb_regressor_experiment",
    hpo_span_name="optuna_affinity_hpo_study",
    study_run_name="optuna_xgb_affinity_study",
    final_span_name="train_final_affinity_xgb_candidate",
    final_run_name="xgb_affinity_final_candidate",
    task_tag="affinity_regression",
    direction="minimize",
    val_metric_name="val_rmse",
    objective="reg:squarederror",
    eval_metric="rmse",
    shap_artifact_prefix="affinity_shap",
)


class ChurnProbabilityPyFunc(PythonModel):
    """MLflow PyFunc wrapper serving positive-class churn probabilities from a UBJSON `XGBClassifier`.

    MLflow's built-in `mlflow.xgboost` PyFunc loader invokes `XGBClassifier.predict()`, which
    returns hard binary class labels (`0`/`1`) instead of continuous probabilities. Wrapping the
    UBJSON-serialized booster in this thin PyFunc ensures `mlflow.pyfunc.spark_udf` and Model
    Serving endpoints emit calibrated positive-class probabilities (`predict_proba[:, 1]`).
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the UBJSON-serialized `XGBClassifier` booster from the logged MLflow artifact."""
        clf = XGBClassifier()
        clf.load_model(context.artifacts["xgb_model"])
        self._model = clf

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.Series:
        """Return positive-class churn probabilities (`P(churn=1)`) as a float64 Pandas Series."""
        _ = (context, params)
        proba = self._model.predict_proba(model_input)[:, 1]
        return pd.Series(proba, index=model_input.index, dtype="float64", name="predicted_churn_probability")


@dataclass(frozen=True)
class ExperimentRunOutcome:
    """Immutable outcome summary of an Optuna + MLflow 3 + SHAP XGBoost training experiment."""

    parent_run_id: str
    final_run_id: str
    best_params: dict[str, Any]
    best_val_metric: float
    holdout_metrics: dict[str, float]
    model: XGBModel
    feature_importance_df: pd.DataFrame
    model_uri: str
    registered_model_version: str | None
    logged_model_id: str | None = None
    holdout_eval_result: EvaluationResult | None = None
    shap_summary_df: pd.DataFrame | None = None
    shap_artifact_files: tuple[str, ...] = ()
    optimal_decision_threshold: float | None = None


def _log_dataset_inputs(
    splits: DatasetSplits,
    target_col: str,
    dataset_name_prefix: str,
    dataset_source: str | None,
) -> tuple[Any, Any, pd.DataFrame]:
    """Log training and holdout evaluation datasets to MLflow 3 lineage via `mlflow.log_input()`."""
    source_uri = dataset_source or f"in_memory://{dataset_name_prefix}"
    trainval_df = splits.X_trainval.copy()
    trainval_df[target_col] = splits.y_trainval.to_numpy()
    holdout_df = splits.X_test.copy()
    holdout_df[target_col] = splits.y_test.to_numpy()

    train_dataset = mlflow_from_pandas(
        trainval_df,
        source=source_uri,
        targets=target_col,
        name=f"{dataset_name_prefix}_trainval",
    )
    eval_dataset = mlflow_from_pandas(
        holdout_df,
        source=source_uri,
        targets=target_col,
        name=f"{dataset_name_prefix}_holdout",
    )
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(eval_dataset, context="evaluation")
    return train_dataset, eval_dataset, holdout_df


def _extract_logged_model_metadata(model_info: Any) -> tuple[str, str | None, str | None]:
    """Extract `(model_uri, registered_model_version, logged_model_id)` from an MLflow `ModelInfo` object."""
    version = getattr(model_info, "registered_model_version", None)
    model_id = getattr(model_info, "model_id", None)
    return (
        str(model_info.model_uri),
        str(version) if version is not None else None,
        str(model_id) if model_id is not None else None,
    )


def _log_churn_probability_pyfunc_model(
    model: XGBClassifier,
    sample_input: pd.DataFrame,
    registered_model_name: str | None,
) -> tuple[str, str | None, str | None]:
    """Serialize an `XGBClassifier` to UBJSON and wrap it in `ChurnProbabilityPyFunc` for continuous probabilities."""
    sample_proba = model.predict_proba(sample_input)[:, 1].astype("float64")
    signature = infer_signature(sample_input, sample_proba)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ubj_path = Path(tmp_dir) / "model.ubj"
        model.save_model(str(ubj_path))
        pyfunc_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ChurnProbabilityPyFunc(),
            artifacts={"xgb_model": str(ubj_path)},
            signature=signature,
            input_example=sample_input,
            registered_model_name=registered_model_name,
        )
    return _extract_logged_model_metadata(pyfunc_info)


def _log_xgboost_candidate_model(
    model: XGBModel,
    X_trainval: pd.DataFrame,
    registered_model_name: str | None,
    fe_client: Any | None = None,
    training_set: Any | None = None,
) -> tuple[str, str | None, str | None]:
    """Dispatch candidate model logging across Feature Store, probability PyFunc, or native UBJSON flavors."""
    sample_input = X_trainval.iloc[: min(5, len(X_trainval))]

    if fe_client is not None and training_set is not None:
        fe_info = fe_client.log_model(
            model=model,
            artifact_path="model",
            flavor=mlflow.xgboost,
            training_set=training_set,
            registered_model_name=registered_model_name,
            input_example=sample_input,
        )
        return _extract_logged_model_metadata(fe_info)

    if isinstance(model, XGBClassifier):
        return _log_churn_probability_pyfunc_model(model, sample_input, registered_model_name)

    sample_preds = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_preds)
    model_info = mlflow.xgboost.log_model(
        xgb_model=model,
        name="model",
        model_format="ubj",
        signature=signature,
        input_example=sample_input,
        registered_model_name=registered_model_name,
    )
    return _extract_logged_model_metadata(model_info)


def _log_holdout_metrics_with_lineage(
    holdout_metrics: dict[str, float],
    model_id: str | None,
    eval_dataset: Any,
) -> None:
    """Attach holdout evaluation metrics to the run and (when available) the MLflow 3 `LoggedModel`."""
    if model_id is not None:
        mlflow.log_metrics(holdout_metrics, model_id=model_id, dataset=eval_dataset)
    else:
        mlflow.log_metrics(holdout_metrics)


def _build_final_xgb_params(
    spec: ModelTaskSpec,
    best_params: dict[str, Any],
    pos_weight: float,
    random_state: int,
) -> dict[str, Any]:
    """Merge winning Optuna hyperparameters with fixed objective, eval metric, and class balance settings."""
    final_params: dict[str, Any] = {
        **best_params,
        "objective": spec.objective,
        "eval_metric": spec.eval_metric,
        "random_state": random_state,
    }
    if spec.task_type == "classification":
        final_params["scale_pos_weight"] = pos_weight
    return final_params


def _fit_final_candidate_model(
    spec: ModelTaskSpec,
    splits: DatasetSplits,
    final_params: dict[str, Any],
) -> tuple[XGBModel, Any, float | None]:
    """Calibrate classification decision threshold on `X_val` (leak-free), then fit the final model on `X_trainval`."""
    if spec.task_type == "classification":
        # Tune the probability decision threshold on X_val BEFORE retraining on X_trainval so X_test stays untouched.
        probe_clf = XGBClassifier(**final_params)
        probe_clf.fit(splits.X_train, splits.y_train, verbose=False)
        threshold_metrics = compute_optimal_classification_threshold(
            splits.y_val, probe_clf.predict_proba(splits.X_val)[:, 1]
        )
        optimal_threshold = threshold_metrics["optimal_f1_threshold"]
        mlflow.log_metrics({f"val_{k}": v for k, v in threshold_metrics.items()})
        mlflow.set_tag("optimal_decision_threshold", f"{optimal_threshold:.4f}")

        final_clf = XGBClassifier(**final_params)
        final_clf.fit(splits.X_trainval, splits.y_trainval, verbose=False)
        return final_clf, (lambda df: final_clf.predict_proba(df)[:, 1]), optimal_threshold

    final_reg = XGBRegressor(**final_params)
    final_reg.fit(splits.X_trainval, splits.y_trainval, verbose=False)
    return final_reg, final_reg.predict, None


def _run_final_candidate_stage(
    spec: ModelTaskSpec,
    splits: DatasetSplits,
    feature_cols: tuple[str, ...],
    target_col: str,
    parent_run_id: str,
    final_params: dict[str, Any],
    registered_model_name: str | None,
    fe_client: Any | None,
    training_set: Any | None,
    training_dataset_source: str | None,
    log_shap_artifacts: bool,
    shap_max_samples: int,
    random_state: int,
) -> tuple[
    str,
    XGBModel,
    pd.DataFrame,
    str,
    str | None,
    str | None,
    EvaluationResult,
    dict[str, float],
    ShapExplainabilityResult | None,
    float | None,
]:
    """Train, log, evaluate, and explain the final candidate model inside a dedicated MLflow 3 run."""
    with (
        mlflow.start_span(name=spec.final_span_name) as final_span,
        mlflow.start_run(run_name=spec.final_run_name) as final_run,
    ):
        mlflow.set_tags(
            {
                "pipeline_stage": "final_training",
                "optuna_parent_run_id": parent_run_id,
                "split_strategy": splits.split_strategy,
                "task": spec.task_tag,
            }
        )
        mlflow.log_params(final_params)
        _, eval_dataset, holdout_df = _log_dataset_inputs(
            splits=splits,
            target_col=target_col,
            dataset_name_prefix=spec.dataset_name_prefix,
            dataset_source=training_dataset_source,
        )

        final_model, predict_fn, optimal_threshold = _fit_final_candidate_model(spec, splits, final_params)
        fi_df = extract_feature_importance_df(final_model, feature_cols)

        # Log the candidate model FIRST so MLflow 3 returns `logged_model_id` for lineage-bound holdout metrics.
        model_uri, model_version, logged_model_id = _log_xgboost_candidate_model(
            final_model, splits.X_trainval, registered_model_name, fe_client, training_set
        )
        eval_result = evaluate_model_on_holdout(
            model_or_uri=predict_fn,
            holdout_df=holdout_df,
            feature_cols=feature_cols,
            target_col=target_col,
            task_type=spec.task_type,
            model_id=logged_model_id,
        )
        holdout_metrics = {k: float(v) for k, v in eval_result.metrics.items() if k.startswith("holdout_")}
        _log_holdout_metrics_with_lineage(holdout_metrics, logged_model_id, eval_dataset)

        shap_result: ShapExplainabilityResult | None = None
        if log_shap_artifacts:
            shap_result = log_shap_explainability_artifacts(
                model=final_model,
                X_eval=splits.X_test,
                feature_cols=feature_cols,
                run_id=None,
                artifact_prefix=spec.shap_artifact_prefix,
                max_samples=shap_max_samples,
                random_state=random_state,
            )
            mlflow.log_metric("shap_expected_value", shap_result.expected_value)
            if not shap_result.summary_df.empty:
                mlflow.set_tag("shap_top_feature", str(shap_result.summary_df.iloc[0]["feature"]))

        final_run_id = str(final_run.info.run_id)
        final_span.set_outputs(
            {
                "final_run_id": final_run_id,
                "logged_model_id": logged_model_id,
                "holdout_metrics": holdout_metrics,
                "optimal_decision_threshold": optimal_threshold,
                "shap_artifact_count": len(shap_result.artifact_files) if shap_result is not None else 0,
            }
        )
    return (
        final_run_id,
        final_model,
        fi_df,
        model_uri,
        model_version,
        logged_model_id,
        eval_result,
        holdout_metrics,
        shap_result,
        optimal_threshold,
    )


def run_xgb_experiment(
    spec: ModelTaskSpec,
    splits: DatasetSplits,
    feature_cols: tuple[str, ...],
    n_trials: int = 10,
    registered_model_name: str | None = None,
    fe_client: Any | None = None,
    training_set: Any | None = None,
    random_state: int = 42,
    spark: SparkSession | None = None,
    spark_parallelism: int = 4,
    worker_n_jobs: int = 1,
    early_stopping_rounds: int | None = 30,
    training_dataset_source: str | None = None,
    target_col: str | None = None,
    log_shap_artifacts: bool = True,
    shap_max_samples: int = 500,
) -> ExperimentRunOutcome:
    """Execute a unified Spark-Optuna HPO, MLflow 3 evaluation, and SHAP explainability experiment."""
    resolved_target = target_col or spec.default_target_col
    with mlflow.start_span(name=spec.root_span_name, span_type="CHAIN") as root_span:
        root_span.set_inputs(
            {
                "task_type": spec.task_type,
                "n_trials": n_trials,
                "n_features": len(feature_cols),
                "train_rows": len(splits.X_train),
                "val_rows": len(splits.X_val),
                "test_rows": len(splits.X_test),
                "split_strategy": splits.split_strategy,
                "spark_distributed": spark is not None,
                "early_stopping_rounds": early_stopping_rounds,
                "log_shap_artifacts": log_shap_artifacts,
                "registered_model_name": registered_model_name,
            }
        )

        pos_weight = compute_scale_pos_weight(splits.y_train) if spec.task_type == "classification" else 1.0
        parent_run_id, best_params, best_val_metric = execute_optuna_xgb_study(
            splits=splits,
            task_type=spec.task_type,
            study_run_name=spec.study_run_name,
            span_name=spec.hpo_span_name,
            task_tag=spec.task_tag,
            direction=spec.direction,
            metric_name=spec.val_metric_name,
            n_trials=n_trials,
            random_state=random_state,
            scale_pos_weight=pos_weight,
            spark=spark,
            spark_parallelism=spark_parallelism,
            early_stopping_rounds=early_stopping_rounds,
            worker_n_jobs=worker_n_jobs,
        )

        final_params = _build_final_xgb_params(spec, best_params, pos_weight, random_state)
        (
            final_run_id,
            final_model,
            fi_df,
            model_uri,
            model_version,
            logged_model_id,
            eval_result,
            holdout_metrics,
            shap_result,
            optimal_threshold,
        ) = _run_final_candidate_stage(
            spec=spec,
            splits=splits,
            feature_cols=feature_cols,
            target_col=resolved_target,
            parent_run_id=parent_run_id,
            final_params=final_params,
            registered_model_name=registered_model_name,
            fe_client=fe_client,
            training_set=training_set,
            training_dataset_source=training_dataset_source,
            log_shap_artifacts=log_shap_artifacts,
            shap_max_samples=shap_max_samples,
            random_state=random_state,
        )

        root_span.set_outputs(
            {
                "parent_run_id": parent_run_id,
                "final_run_id": final_run_id,
                "logged_model_id": logged_model_id,
                "registered_model_version": model_version,
                "holdout_metrics": holdout_metrics,
                "optimal_decision_threshold": optimal_threshold,
            }
        )

    return ExperimentRunOutcome(
        parent_run_id=parent_run_id,
        final_run_id=final_run_id,
        best_params=final_params,
        best_val_metric=best_val_metric,
        holdout_metrics=holdout_metrics,
        model=final_model,
        feature_importance_df=fi_df,
        model_uri=model_uri,
        registered_model_version=model_version,
        logged_model_id=logged_model_id,
        holdout_eval_result=eval_result,
        shap_summary_df=shap_result.summary_df if shap_result is not None else None,
        shap_artifact_files=shap_result.artifact_files if shap_result is not None else (),
        optimal_decision_threshold=optimal_threshold,
    )


def run_churn_xgb_classifier_experiment(
    splits: DatasetSplits,
    feature_cols: tuple[str, ...],
    n_trials: int = 10,
    registered_model_name: str | None = None,
    fe_client: Any | None = None,
    training_set: Any | None = None,
    random_state: int = 42,
    spark: SparkSession | None = None,
    spark_parallelism: int = 4,
    training_dataset_source: str | None = None,
    target_col: str = TARGET_CHURN_COL,
    worker_n_jobs: int = 1,
    early_stopping_rounds: int | None = 30,
    log_shap_artifacts: bool = True,
) -> ExperimentRunOutcome:
    """Execute Optuna HPO, SHAP explainability, and MLflow 3 `XGBClassifier` training for customer churn."""
    return run_xgb_experiment(
        spec=CHURN_TASK_SPEC,
        splits=splits,
        feature_cols=feature_cols,
        n_trials=n_trials,
        registered_model_name=registered_model_name,
        fe_client=fe_client,
        training_set=training_set,
        random_state=random_state,
        spark=spark,
        spark_parallelism=spark_parallelism,
        worker_n_jobs=worker_n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        training_dataset_source=training_dataset_source,
        target_col=target_col,
        log_shap_artifacts=log_shap_artifacts,
    )


def run_affinity_xgb_regressor_experiment(
    splits: DatasetSplits,
    feature_cols: tuple[str, ...],
    n_trials: int = 10,
    registered_model_name: str | None = None,
    fe_client: Any | None = None,
    training_set: Any | None = None,
    random_state: int = 42,
    spark: SparkSession | None = None,
    spark_parallelism: int = 4,
    training_dataset_source: str | None = None,
    target_col: str = TARGET_AFFINITY_COL,
    worker_n_jobs: int = 1,
    early_stopping_rounds: int | None = 30,
    log_shap_artifacts: bool = True,
) -> ExperimentRunOutcome:
    """Execute Optuna HPO, SHAP explainability, and MLflow 3 `XGBRegressor` training for product affinity."""
    return run_xgb_experiment(
        spec=AFFINITY_TASK_SPEC,
        splits=splits,
        feature_cols=feature_cols,
        n_trials=n_trials,
        registered_model_name=registered_model_name,
        fe_client=fe_client,
        training_set=training_set,
        random_state=random_state,
        spark=spark,
        spark_parallelism=spark_parallelism,
        worker_n_jobs=worker_n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        training_dataset_source=training_dataset_source,
        target_col=target_col,
        log_shap_artifacts=log_shap_artifacts,
    )
