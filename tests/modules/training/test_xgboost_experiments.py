"""Integration unit tests for MLflow 3 + Spark-distributed Optuna HPO, SHAP explainability, and Spark UDF scoring."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from src.python.modules.evaluation.model_governance import log_feature_importance_artifact
from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
    extract_customer_churn_labels,
    extract_product_affinity_labels,
)
from src.python.modules.features.synthetic_gold_views import (
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)
from src.python.modules.inference.batch_scoring import (
    score_affinity_with_spark_udf,
    score_churn_with_spark_udf,
)
from src.python.modules.training.spark_optuna import (
    compute_binary_classification_metrics,
    compute_optimal_classification_threshold,
    evaluate_trial_partition_pdf,
)
from src.python.modules.training.xgboost_datasets import (
    split_classification_dataset,
    split_regression_dataset,
)
from src.python.modules.training.xgboost_experiment import (
    run_affinity_xgb_regressor_experiment,
    run_churn_xgb_classifier_experiment,
)
from src.python.modules.utils.config import (
    AFFINITY_FEATURE_COLUMNS,
    CHURN_FEATURE_COLUMNS,
    COL_PRODUCT_ID,
    COL_SNAPSHOT_DATE,
    COL_USER_ID,
    TARGET_AFFINITY_COL,
    TARGET_CHURN_COL,
)


@pytest.mark.parametrize(
    ("y_true", "y_prob", "expected_auc_min", "expected_opt_f1_range"),
    [
        ([0, 0, 0, 1, 1, 1], [0.05, 0.15, 0.25, 0.75, 0.85, 0.95], 0.99, (0.05, 0.95)),
        ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], 0.5, (0.5, 0.5)),
    ],
    ids=["well_separated_binary", "single_class_degenerate"],
)
def test_binary_classification_metrics_and_threshold_calibration(
    y_true: list[int],
    y_prob: list[float],
    expected_auc_min: float,
    expected_opt_f1_range: tuple[float, float],
) -> None:
    """Verify PR-AUC, Brier score, and optimal F1 / Youden's J threshold calibration."""
    metrics = compute_binary_classification_metrics(pd.Series(y_true), pd.Series(y_prob), prefix="val")
    assert metrics["val_roc_auc"] >= expected_auc_min
    assert "val_pr_auc" in metrics
    assert "val_brier_score" in metrics
    assert 0.0 <= metrics["val_brier_score"] <= 1.0

    thresholds = compute_optimal_classification_threshold(pd.Series(y_true), pd.Series(y_prob))
    assert expected_opt_f1_range[0] <= thresholds["optimal_f1_threshold"] <= expected_opt_f1_range[1]
    assert 0.05 <= thresholds["youden_j_threshold"] <= 0.95


@pytest.mark.spark
@pytest.mark.mlflow
@pytest.mark.slow
@pytest.mark.xdist_group("spark_churn_e2e")
def test_run_churn_xgb_classifier_experiment_and_spark_udf_scoring(
    spark: SparkSession,
    mlflow_tracking_uri: str,
) -> None:
    """Verify Spark Optuna, early stopping, SHAP artifacts, `ChurnProbabilityPyFunc`, and UDF scoring."""
    assert mlflow.get_tracking_uri() == mlflow_tracking_uri

    raw_churn_df = build_sample_gold_churn_view_df(spark, num_users=100, num_snapshots=5)
    churn_fe_df = raw_churn_df.transform(engineer_customer_churn_features)
    churn_lbl_df = raw_churn_df.transform(extract_customer_churn_labels).select(
        COL_USER_ID, COL_SNAPSHOT_DATE, TARGET_CHURN_COL
    )
    churn_train_pdf = churn_fe_df.join(churn_lbl_df, on=[COL_USER_ID, COL_SNAPSHOT_DATE], how="inner").toPandas()
    churn_splits = split_classification_dataset(
        pdf=churn_train_pdf,
        feature_cols=CHURN_FEATURE_COLUMNS,
        target_col=TARGET_CHURN_COL,
        time_col=COL_SNAPSHOT_DATE,
    )

    partition_out = evaluate_trial_partition_pdf(
        batch_pdf=pd.DataFrame(
            [
                {
                    "trial_number": 0,
                    "params_json": json.dumps({"n_estimators": 40, "max_depth": 3, "n_jobs": 1, "random_state": 42}),
                }
            ]
        ),
        task_type="classification",
        splits=churn_splits,
        metric_name="val_roc_auc",
        early_stopping_rounds=10,
    )
    assert len(partition_out) == 1
    assert "primary_metric" in partition_out.columns
    parsed_metrics = json.loads(str(partition_out.iloc[0]["metrics_json"]))
    assert "best_n_estimators" in parsed_metrics
    assert "val_pr_auc" in parsed_metrics
    assert "val_brier_score" in parsed_metrics

    churn_res = run_churn_xgb_classifier_experiment(
        splits=churn_splits,
        feature_cols=CHURN_FEATURE_COLUMNS,
        n_trials=2,
        registered_model_name="test_churn_xgb_model",
        spark=spark,
        spark_parallelism=2,
        worker_n_jobs=1,
        early_stopping_rounds=15,
        training_dataset_source="uc_table://test_cat.ml_features.fe_customer_churn_features",
    )
    assert churn_res.parent_run_id
    assert churn_res.final_run_id
    assert churn_res.model_uri.startswith("models:/")
    assert "holdout_roc_auc" in churn_res.holdout_metrics
    assert "holdout_pr_auc" in churn_res.holdout_metrics
    assert "holdout_brier_score" in churn_res.holdout_metrics
    assert "min_child_weight" in churn_res.best_params
    assert "gamma" in churn_res.best_params
    assert churn_res.optimal_decision_threshold is not None
    assert 0.05 <= churn_res.optimal_decision_threshold <= 0.95
    assert churn_res.registered_model_version is not None
    assert churn_res.logged_model_id is not None
    assert churn_res.holdout_eval_result is not None
    assert churn_res.shap_summary_df is not None
    assert len(churn_res.shap_artifact_files) == 5

    # Verify MLflow 3 dataset lineage and SHAP explainability artifacts on the final run
    client = MlflowClient()
    final_run = client.get_run(churn_res.final_run_id)
    assert final_run.inputs is not None
    input_contexts = {
        tag.value
        for ds_input in final_run.inputs.dataset_inputs
        for tag in ds_input.tags
        if tag.key == "mlflow.data.context"
    }
    assert {"training", "evaluation"}.issubset(input_contexts)
    shap_artifacts = {Path(a.path).name for a in client.list_artifacts(churn_res.final_run_id, path="explainability")}
    assert "churn_shap_beeswarm.png" in shap_artifacts
    assert "churn_shap_bar.png" in shap_artifacts
    assert "churn_shap_waterfall_top_example.png" in shap_artifacts
    assert "churn_shap_feature_importance.csv" in shap_artifacts
    assert "churn_shap_summary.json" in shap_artifacts

    # Behavioral invariant: PyFunc MUST emit continuous probabilities strictly inside (0, 1), NOT 0/1 labels
    loaded_churn_pyfunc = mlflow.pyfunc.load_model(churn_res.model_uri)
    pyfunc_probs = loaded_churn_pyfunc.predict(churn_splits.X_test)
    assert all(0.0 < float(p) < 1.0 for p in pyfunc_probs), "Churn PyFunc emitted binary 0/1 labels instead of probs"

    csv_path = log_feature_importance_artifact(
        feature_importance_df=churn_res.feature_importance_df,
        run_id=churn_res.final_run_id,
    )
    assert csv_path.endswith("feature_importance.csv")

    with mlflow.start_run(run_name="active_run_fi_log"):
        active_csv = log_feature_importance_artifact(
            feature_importance_df=churn_res.feature_importance_df,
            run_id=None,
        )
        assert active_csv.endswith("feature_importance.csv")

    # Score label-free UC Feature Table directly (unlabeled batch scoring contract)
    scored_churn_df = score_churn_with_spark_udf(
        spark=spark,
        features_df=churn_fe_df.limit(10),
        model_uri=churn_res.model_uri,
        model_alias="Champion",
        model_version=str(churn_res.registered_model_version),
        scoring_date="2026-05-01",
    )
    assert "predicted_churn_probability" in scored_churn_df.columns
    assert "churn_action_tier" in scored_churn_df.columns
    assert "model_version" in scored_churn_df.columns
    assert "scoring_date" in scored_churn_df.columns
    churn_scored_rows = scored_churn_df.collect()
    assert len(churn_scored_rows) == 10
    assert all(0.0 < r["predicted_churn_probability"] < 1.0 for r in churn_scored_rows)
    assert all(r["is_churn_candidate"] is None for r in churn_scored_rows)
    assert all(r["model_version"] == str(churn_res.registered_model_version) for r in churn_scored_rows)
    assert all(str(r["scoring_date"]) == "2026-05-01" for r in churn_scored_rows)


@pytest.mark.spark
@pytest.mark.mlflow
@pytest.mark.slow
@pytest.mark.xdist_group("spark_affinity_e2e")
def test_run_affinity_xgb_regressor_experiment_and_spark_udf_scoring(
    spark: SparkSession,
    mlflow_tracking_uri: str,
) -> None:
    """Verify Spark Optuna, early stopping, SHAP artifacts, UBJSON regressor logging, FS log_model, and UDF scoring."""
    assert mlflow.get_tracking_uri() == mlflow_tracking_uri

    raw_affinity_df = build_sample_gold_affinity_view_df(spark, num_products=75, num_snapshots=5)
    affinity_fe_df = raw_affinity_df.transform(engineer_product_affinity_features)
    affinity_lbl_df = raw_affinity_df.transform(extract_product_affinity_labels).select(
        COL_PRODUCT_ID, COL_SNAPSHOT_DATE, TARGET_AFFINITY_COL
    )
    affinity_train_pdf = affinity_fe_df.join(
        affinity_lbl_df, on=[COL_PRODUCT_ID, COL_SNAPSHOT_DATE], how="inner"
    ).toPandas()
    affinity_splits = split_regression_dataset(
        pdf=affinity_train_pdf,
        feature_cols=AFFINITY_FEATURE_COLUMNS,
        target_col=TARGET_AFFINITY_COL,
        time_col=COL_SNAPSHOT_DATE,
    )
    affinity_res = run_affinity_xgb_regressor_experiment(
        splits=affinity_splits,
        feature_cols=AFFINITY_FEATURE_COLUMNS,
        n_trials=2,
        registered_model_name="test_affinity_xgb_model",
        spark=spark,
        spark_parallelism=2,
        early_stopping_rounds=15,
    )
    assert affinity_res.parent_run_id
    assert "holdout_rmse" in affinity_res.holdout_metrics
    assert "holdout_r2" in affinity_res.holdout_metrics
    assert affinity_res.shap_summary_df is not None
    assert len(affinity_res.shap_artifact_files) == 5

    client = MlflowClient()
    aff_shap_artifacts = {
        Path(a.path).name for a in client.list_artifacts(affinity_res.final_run_id, path="explainability")
    }
    assert "affinity_shap_beeswarm.png" in aff_shap_artifacts
    assert "affinity_shap_bar.png" in aff_shap_artifacts
    assert "affinity_shap_waterfall_top_example.png" in aff_shap_artifacts

    scored_affinity_df = score_affinity_with_spark_udf(
        spark=spark,
        features_df=affinity_fe_df.limit(10),
        model_uri=affinity_res.model_uri,
        model_alias="Champion",
        model_version=str(affinity_res.registered_model_version),
        scoring_date="2026-05-01",
    )
    assert "predicted_affinity_score" in scored_affinity_df.columns
    assert "affinity_tier" in scored_affinity_df.columns
    assert "model_version" in scored_affinity_df.columns
    assert "scoring_date" in scored_affinity_df.columns
    aff_scored_rows = scored_affinity_df.collect()
    assert len(aff_scored_rows) == 10
    assert all(r["affinity_index_score"] is None for r in aff_scored_rows)
    assert all(r["affinity_residual"] is None for r in aff_scored_rows)
    assert all(str(r["scoring_date"]) == "2026-05-01" for r in aff_scored_rows)

    mock_fe = MagicMock()
    mock_fe.log_model.return_value = SimpleNamespace(
        model_uri="models:/cat.ml_models.fe_logged/1",
        registered_model_version="1",
    )
    fe_res = run_affinity_xgb_regressor_experiment(
        splits=affinity_splits,
        feature_cols=AFFINITY_FEATURE_COLUMNS,
        n_trials=1,
        registered_model_name="cat.ml_models.fe_logged",
        fe_client=mock_fe,
        training_set=MagicMock(),
        spark=None,
        log_shap_artifacts=False,
    )
    assert fe_res.model_uri == "models:/cat.ml_models.fe_logged/1"
    assert fe_res.registered_model_version == "1"
    assert fe_res.shap_summary_df is None
    assert fe_res.shap_artifact_files == ()
