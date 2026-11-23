"""XGBoost dataset splitting, leakage detection, Spark-distributed Optuna HPO, and MLflow 3 tracking."""

from src.python.modules.training.spark_optuna import (
    compute_binary_classification_metrics,
    compute_optimal_classification_threshold,
    compute_regression_metrics,
    evaluate_single_xgb_trial,
    execute_optuna_xgb_study,
    suggest_xgb_trial_params,
)
from src.python.modules.training.xgboost_datasets import (
    DatasetSplits,
    compute_scale_pos_weight,
    detect_single_feature_leakage,
    extract_feature_importance_df,
    split_classification_dataset,
    split_regression_dataset,
)
from src.python.modules.training.xgboost_experiment import (
    AFFINITY_TASK_SPEC,
    CHURN_TASK_SPEC,
    ChurnProbabilityPyFunc,
    ExperimentRunOutcome,
    ModelTaskSpec,
    run_affinity_xgb_regressor_experiment,
    run_churn_xgb_classifier_experiment,
    run_xgb_experiment,
)

__all__ = [
    "AFFINITY_TASK_SPEC",
    "CHURN_TASK_SPEC",
    "ChurnProbabilityPyFunc",
    "DatasetSplits",
    "ExperimentRunOutcome",
    "ModelTaskSpec",
    "compute_binary_classification_metrics",
    "compute_optimal_classification_threshold",
    "compute_regression_metrics",
    "compute_scale_pos_weight",
    "detect_single_feature_leakage",
    "evaluate_single_xgb_trial",
    "execute_optuna_xgb_study",
    "extract_feature_importance_df",
    "run_affinity_xgb_regressor_experiment",
    "run_churn_xgb_classifier_experiment",
    "run_xgb_experiment",
    "split_classification_dataset",
    "split_regression_dataset",
    "suggest_xgb_trial_params",
]
