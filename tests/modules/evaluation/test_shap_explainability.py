"""Unit tests for TreeSHAP explainability computation and MLflow artifact logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import pytest
import shap
from mlflow.tracking import MlflowClient
from src.python.modules.evaluation.shap_explainability import (
    _extract_scalar_expected_value,
    _normalize_shap_explanation,
    compute_shap_explanation,
    log_shap_explainability_artifacts,
)
from xgboost import XGBClassifier, XGBModel, XGBRegressor


def _build_demo_model_and_data(task_type: Literal["classification", "regression"]) -> tuple[XGBModel, pd.DataFrame]:
    """Train a tiny deterministic XGBoost model for SHAP unit tests."""
    rng = np.random.default_rng(42)
    n_rows = 60
    x_df = pd.DataFrame(
        {
            "signal_a": rng.normal(loc=0.0, scale=2.0, size=n_rows),
            "signal_b": rng.uniform(0.0, 10.0, size=n_rows),
            "noise_c": rng.normal(loc=0.0, scale=0.1, size=n_rows),
        }
    )
    if task_type == "classification":
        y = ((x_df["signal_a"] + 0.25 * x_df["signal_b"]) > 1.2).astype(int)
        clf = XGBClassifier(n_estimators=15, max_depth=3, random_state=42, eval_metric="logloss")
        clf.fit(x_df, y)
        return clf, x_df

    y_reg = 4.0 * x_df["signal_a"] + 1.5 * x_df["signal_b"]
    reg = XGBRegressor(n_estimators=15, max_depth=3, random_state=42, eval_metric="rmse")
    reg.fit(x_df, y_reg)
    return reg, x_df


@pytest.mark.mlflow
@pytest.mark.parametrize("task_type", ["classification", "regression"])
@pytest.mark.parametrize("log_via_run_id", [False, True], ids=["active_run", "explicit_run_id"])
def test_log_shap_explainability_artifacts_end_to_end(
    mlflow_tracking_uri: str,
    task_type: Literal["classification", "regression"],
    log_via_run_id: bool,
) -> None:
    """Verify TreeSHAP computes ranked attributions and logs CSV, JSON, beeswarm, bar, and waterfall artifacts."""
    _ = mlflow_tracking_uri
    model, x_df = _build_demo_model_and_data(task_type)
    feature_cols = ("signal_a", "signal_b", "noise_c")
    prefix = f"{task_type}_shap"

    with mlflow.start_run(run_name=f"shap_{task_type}") as run:
        run_id = str(run.info.run_id)
        if not log_via_run_id:
            result = log_shap_explainability_artifacts(
                model=model,
                X_eval=x_df,
                feature_cols=feature_cols,
                run_id=None,
                artifact_prefix=prefix,
                max_samples=25,
                random_state=42,
            )

    if log_via_run_id:
        result = log_shap_explainability_artifacts(
            model=model,
            X_eval=x_df,
            feature_cols=feature_cols,
            run_id=run_id,
            artifact_prefix=prefix,
            max_samples=25,
            random_state=42,
        )

    assert result.sample_size == 25
    assert list(result.summary_df.columns) == ["feature", "mean_abs_shap", "mean_shap", "std_shap", "rank"]
    assert list(result.summary_df["rank"]) == [1, 2, 3]
    assert result.summary_df["mean_abs_shap"].is_monotonic_decreasing
    assert result.summary_df.iloc[0]["feature"] in {"signal_a", "signal_b"}
    assert len(result.artifact_files) == 5

    client = MlflowClient()
    logged = {Path(a.path).name for a in client.list_artifacts(run_id, path="explainability")}
    expected = {
        f"{prefix}_feature_importance.csv",
        f"{prefix}_summary.json",
        f"{prefix}_beeswarm.png",
        f"{prefix}_bar.png",
        f"{prefix}_waterfall_top_example.png",
    }
    assert expected.issubset(logged)

    summary_local = Path(client.download_artifacts(run_id, f"explainability/{prefix}_summary.json"))
    payload = json.loads(summary_local.read_text(encoding="utf-8"))
    assert payload["sample_size"] == 25
    assert payload["n_features"] == 3
    assert len(payload["top_features"]) == 3


def test_compute_shap_explanation_edge_cases() -> None:
    """Verify empty-frame guard, 3D explanation slicing, and scalar expected-value coercion."""
    model, x_df = _build_demo_model_and_data("regression")
    with pytest.raises(ValueError, match="empty evaluation DataFrame"):
        compute_shap_explanation(model, x_df.iloc[0:0], ("signal_a", "signal_b", "noise_c"))

    assert _extract_scalar_expected_value(np.array([])) == 0.0
    assert _extract_scalar_expected_value(np.array([1.0, 3.0])) == pytest.approx(2.0)

    exp_3d = shap.Explanation(
        values=np.zeros((4, 3, 2)),
        base_values=np.zeros((4, 2)),
        data=np.zeros((4, 3)),
        feature_names=["a", "b", "c"],
    )
    normalized = _normalize_shap_explanation(exp_3d)
    assert normalized.values.shape == (4, 3)
