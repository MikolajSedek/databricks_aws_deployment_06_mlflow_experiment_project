"""Spark-distributed Optuna hyperparameter optimization via `mapInPandas` and Optuna Ask-and-Tell.

Distributes parallel waves of XGBoost hyperparameter trials across Spark executor partitions
using `DataFrame.mapInPandas` (100% compatible with Databricks Serverless Spark Connect and
classic Spark clusters), applies validation early stopping (`early_stopping_rounds`), caps
per-worker threads (`worker_n_jobs`) to prevent executor CPU oversubscription, updates Optuna's
`TPESampler` Bayesian posterior via `study.tell()`, and records nested MLflow 3 child runs.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from typing import Any, Final, Literal

import mlflow
import numpy as np
import optuna
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier, XGBRegressor

from src.python.modules.training.xgboost_datasets import DatasetSplits

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRIAL_INPUT_SCHEMA: Final[StructType] = StructType(
    [
        StructField("trial_number", IntegerType(), nullable=False),
        StructField("params_json", StringType(), nullable=False),
    ]
)

TRIAL_RESULT_SCHEMA: Final[StructType] = StructType(
    [
        StructField("trial_number", IntegerType(), nullable=False),
        StructField("primary_metric", DoubleType(), nullable=False),
        StructField("metrics_json", StringType(), nullable=False),
    ]
)


def compute_binary_classification_metrics(
    y_true: pd.Series,
    y_prob: Any,
    prefix: str = "holdout",
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Brier Score, F1, Precision, Recall, and Log-Loss for binary probabilities."""
    probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (probs >= 0.5).astype(int)
    unique_classes = len({int(v) for v in y_true})
    auc = float(roc_auc_score(y_true, probs)) if unique_classes >= 2 else 0.5
    pr_auc = float(average_precision_score(y_true, probs)) if unique_classes >= 2 else float(y_true.mean())
    brier = float(brier_score_loss(y_true, probs))
    return {
        f"{prefix}_roc_auc": round(auc, 6),
        f"{prefix}_pr_auc": round(pr_auc, 6),
        f"{prefix}_brier_score": round(brier, 6),
        f"{prefix}_f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{prefix}_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{prefix}_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{prefix}_log_loss": round(float(log_loss(y_true, probs, labels=[0, 1])), 6),
    }


def compute_optimal_classification_threshold(
    y_true: pd.Series,
    y_prob: Any,
) -> dict[str, float]:
    """Compute F1-optimal and Youden's J probability decision thresholds on a validation split."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if len(np.unique(y_arr)) < 2:
        return {
            "optimal_f1_threshold": 0.5,
            "optimal_f1_score": 0.0,
            "youden_j_threshold": 0.5,
        }

    # Maximize F1 across precision-recall thresholds, clamping extreme bounds to [0.05, 0.95].
    precisions, recalls, pr_thresholds = precision_recall_curve(y_arr, p_arr)
    if len(pr_thresholds) > 0:
        denom = precisions[:-1] + recalls[:-1]
        safe_denom = np.where(denom > 0.0, denom, 1.0)
        f1_scores = np.where(denom > 0.0, 2.0 * precisions[:-1] * recalls[:-1] / safe_denom, 0.0)
        best_pr_idx = int(np.argmax(f1_scores))
        opt_f1_threshold = float(np.clip(pr_thresholds[best_pr_idx], 0.05, 0.95))
        opt_f1_val = float(f1_scores[best_pr_idx])
    else:
        opt_f1_threshold = 0.5
        opt_f1_val = 0.0

    # Maximize Youden's J statistic (TPR - FPR) across finite ROC thresholds.
    fpr, tpr, roc_thresholds = roc_curve(y_arr, p_arr)
    finite_mask = np.isfinite(roc_thresholds)
    if np.any(finite_mask):
        j_scores = (tpr - fpr)[finite_mask]
        youden_threshold = float(np.clip(roc_thresholds[finite_mask][int(np.argmax(j_scores))], 0.05, 0.95))
    else:
        youden_threshold = 0.5

    return {
        "optimal_f1_threshold": round(opt_f1_threshold, 4),
        "optimal_f1_score": round(opt_f1_val, 6),
        "youden_j_threshold": round(youden_threshold, 4),
    }


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: Any,
    prefix: str = "holdout",
) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 for continuous regression predictions."""
    rmse = math.sqrt(float(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.0
    return {
        f"{prefix}_rmse": round(rmse, 6),
        f"{prefix}_mae": round(mae, 6),
        f"{prefix}_r2": round(r2, 6),
    }


def suggest_xgb_trial_params(
    trial: optuna.Trial,
    task_type: Literal["classification", "regression"],
    random_state: int,
    scale_pos_weight: float = 1.0,
    worker_n_jobs: int = 1,
) -> dict[str, Any]:
    """Sample XGBoost hyperparameters from an Optuna trial with per-worker thread capping."""
    base_params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 60, 300, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        # Cap threads per trial so parallel Spark `mapInPandas` partitions do not contend for CPU cores.
        "n_jobs": max(1, int(worker_n_jobs)),
        "random_state": random_state,
    }
    if task_type == "classification":
        return {
            **base_params,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
    return {
        **base_params,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }


def evaluate_single_xgb_trial(
    params: dict[str, Any],
    task_type: Literal["classification", "regression"],
    splits: DatasetSplits,
    metric_name: str,
    early_stopping_rounds: int | None = 30,
) -> tuple[float, dict[str, float]]:
    """Train and evaluate a single XGBoost trial on `(X_train, y_train)` with validation early stopping."""
    fit_params = dict(params)
    if early_stopping_rounds is not None and early_stopping_rounds > 0 and len(splits.X_val) > 0:
        fit_params["early_stopping_rounds"] = int(early_stopping_rounds)

    if task_type == "classification":
        clf = XGBClassifier(**fit_params)
        clf.fit(splits.X_train, splits.y_train, eval_set=[(splits.X_val, splits.y_val)], verbose=False)
        val_metrics = compute_binary_classification_metrics(
            splits.y_val, clf.predict_proba(splits.X_val)[:, 1], prefix="val"
        )
        best_iter = getattr(clf, "best_iteration", None)
    else:
        reg = XGBRegressor(**fit_params)
        reg.fit(splits.X_train, splits.y_train, eval_set=[(splits.X_val, splits.y_val)], verbose=False)
        val_metrics = compute_regression_metrics(splits.y_val, reg.predict(splits.X_val), prefix="val")
        best_iter = getattr(reg, "best_iteration", None)

    # Record effective boosting round count (+1 because XGBoost `best_iteration` is 0-indexed).
    best_n_estimators = int(best_iter) + 1 if best_iter is not None else int(params.get("n_estimators", 100))
    val_metrics["best_n_estimators"] = float(best_n_estimators)
    return float(val_metrics[metric_name]), val_metrics


def evaluate_trial_partition_pdf(
    batch_pdf: pd.DataFrame,
    task_type: Literal["classification", "regression"],
    splits: DatasetSplits,
    metric_name: str,
    early_stopping_rounds: int | None = 30,
) -> pd.DataFrame:
    """Evaluate a Pandas partition of `(trial_number, params_json)` rows on a Spark executor."""
    out_records: list[dict[str, Any]] = []
    for row in batch_pdf.itertuples(index=False):
        t_num = int(str(row.trial_number))
        t_params: dict[str, Any] = json.loads(str(row.params_json))
        primary_val, metrics_dict = evaluate_single_xgb_trial(
            params=t_params,
            task_type=task_type,
            splits=splits,
            metric_name=metric_name,
            early_stopping_rounds=early_stopping_rounds,
        )
        out_records.append(
            {
                "trial_number": t_num,
                "primary_metric": float(primary_val),
                "metrics_json": json.dumps(metrics_dict),
            }
        )
    return pd.DataFrame(out_records)


def _evaluate_wave_on_spark(
    spark: SparkSession,
    wave_items: list[tuple[optuna.Trial, dict[str, Any]]],
    task_type: Literal["classification", "regression"],
    splits: DatasetSplits,
    metric_name: str,
    early_stopping_rounds: int | None = 30,
) -> dict[int, tuple[float, dict[str, float]]]:
    """Distribute a wave of Optuna candidate trials across Spark partitions via `mapInPandas`."""
    input_rows = [(int(trial.number), json.dumps(params)) for trial, params in wave_items]
    # Repartition to 1 trial per partition so executors run all trials in the wave concurrently.
    wave_df = spark.createDataFrame(input_rows, schema=TRIAL_INPUT_SCHEMA).repartition(max(1, len(input_rows)))

    def _worker_mapper(pdf_iter: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for batch_pdf in pdf_iter:
            if not batch_pdf.empty:
                yield evaluate_trial_partition_pdf(
                    batch_pdf=batch_pdf,
                    task_type=task_type,
                    splits=splits,
                    metric_name=metric_name,
                    early_stopping_rounds=early_stopping_rounds,
                )

    collected = wave_df.mapInPandas(_worker_mapper, schema=TRIAL_RESULT_SCHEMA).collect()
    return {int(r["trial_number"]): (float(r["primary_metric"]), json.loads(str(r["metrics_json"]))) for r in collected}


def _sample_optuna_wave(
    study: optuna.Study,
    wave_size: int,
    task_type: Literal["classification", "regression"],
    random_state: int,
    scale_pos_weight: float,
    worker_n_jobs: int,
) -> list[tuple[optuna.Trial, dict[str, Any]]]:
    """Ask Optuna's TPE sampler for a batch of candidate trial hyperparameter configurations."""
    wave_items: list[tuple[optuna.Trial, dict[str, Any]]] = []
    for _ in range(wave_size):
        trial = study.ask()
        params = suggest_xgb_trial_params(
            trial=trial,
            task_type=task_type,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            worker_n_jobs=worker_n_jobs,
        )
        wave_items.append((trial, params))
    return wave_items


def _record_wave_outcomes_and_tell(
    study: optuna.Study,
    wave_items: list[tuple[optuna.Trial, dict[str, Any]]],
    wave_outcomes: dict[int, tuple[float, dict[str, float]]],
    task_tag: str,
    engine_mode: str,
) -> None:
    """Log each trial in a completed wave as a nested MLflow child run and report scores to Optuna."""
    for trial, params in wave_items:
        primary_val, val_metrics = wave_outcomes[int(trial.number)]
        with mlflow.start_run(run_name=f"xgb_{task_tag}_trial_{trial.number:02d}", nested=True) as child_run:
            mlflow.set_tag("hpo_execution_engine", engine_mode)
            mlflow.log_params(params)
            mlflow.log_metrics(val_metrics)
            trial.set_user_attr("mlflow_run_id", child_run.info.run_id)
            if "best_n_estimators" in val_metrics:
                trial.set_user_attr("best_n_estimators", int(val_metrics["best_n_estimators"]))
        study.tell(trial, primary_val)


def _finalize_best_study_params(study: optuna.Study, metric_name: str) -> tuple[dict[str, Any], float]:
    """Extract the winning trial's parameters, adopting its early-stopped `n_estimators` when available."""
    best_params = dict(study.best_trial.params)
    best_stopped_trees = study.best_trial.user_attrs.get("best_n_estimators")
    if best_stopped_trees is not None:
        # Clamp to >=10 trees so final retraining on train+val never collapses to a degenerate stump on small splits.
        best_params["n_estimators"] = max(10, int(best_stopped_trees))
        mlflow.log_metric("best_early_stopping_n_estimators", float(best_params["n_estimators"]))

    best_val = float(study.best_value)
    mlflow.log_metric(f"best_{metric_name}", best_val)
    return best_params, best_val


def execute_optuna_xgb_study(
    splits: DatasetSplits,
    task_type: Literal["classification", "regression"],
    study_run_name: str,
    span_name: str,
    task_tag: str,
    direction: Literal["minimize", "maximize"],
    metric_name: str,
    n_trials: int,
    random_state: int,
    scale_pos_weight: float = 1.0,
    spark: SparkSession | None = None,
    spark_parallelism: int = 4,
    early_stopping_rounds: int | None = 30,
    worker_n_jobs: int = 1,
) -> tuple[str, dict[str, Any], float]:
    """Run an Optuna Ask-and-Tell study (distributed on Spark via `mapInPandas` when `spark` is provided)."""
    engine_mode = "spark_map_in_pandas" if spark is not None else "local_sequential"
    with (
        mlflow.start_span(name=span_name) as hpo_span,
        mlflow.start_run(run_name=study_run_name) as parent_run,
    ):
        mlflow.set_tags(
            {
                "pipeline_stage": "hpo_study",
                "model_family": "xgboost",
                "task": task_tag,
                "hpo_execution_engine": engine_mode,
            }
        )
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "metric": metric_name,
                "hpo_execution_engine": engine_mode,
                "spark_parallelism": spark_parallelism if spark is not None else 1,
                "worker_n_jobs": max(1, int(worker_n_jobs)),
                "early_stopping_rounds": early_stopping_rounds if early_stopping_rounds is not None else 0,
                "scale_pos_weight": scale_pos_weight,
            }
        )

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
        completed = 0
        wave_capacity = max(1, spark_parallelism) if spark is not None else 1

        while completed < n_trials:
            current_wave_size = min(wave_capacity, n_trials - completed)
            wave_items = _sample_optuna_wave(
                study, current_wave_size, task_type, random_state, scale_pos_weight, worker_n_jobs
            )
            if spark is not None:
                wave_outcomes = _evaluate_wave_on_spark(
                    spark, wave_items, task_type, splits, metric_name, early_stopping_rounds
                )
            else:
                wave_outcomes = {
                    int(t.number): evaluate_single_xgb_trial(p, task_type, splits, metric_name, early_stopping_rounds)
                    for t, p in wave_items
                }

            _record_wave_outcomes_and_tell(study, wave_items, wave_outcomes, task_tag, engine_mode)
            completed += current_wave_size

        best_params, best_val = _finalize_best_study_params(study, metric_name)
        parent_run_id = str(parent_run.info.run_id)
        hpo_span.set_outputs(
            {
                "parent_run_id": parent_run_id,
                "hpo_execution_engine": engine_mode,
                f"best_{metric_name}": best_val,
                "best_n_estimators": int(best_params["n_estimators"]),
            }
        )
        return parent_run_id, best_params, best_val
