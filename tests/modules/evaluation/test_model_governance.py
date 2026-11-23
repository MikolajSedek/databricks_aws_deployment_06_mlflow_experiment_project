"""Unit tests for MLflow 3 holdout evaluation, validation gating, and UC Champion/Challenger governance."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlflow
import mlflow.pyfunc
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.tracking import MlflowClient
from src.python.modules.evaluation.model_governance import (
    evaluate_model_on_holdout,
    promote_champion_challenger_alias,
    validate_candidate_evaluation,
)


class _LinearScorePyFunc(PythonModel):
    """Minimal MLflow PyFunc returning deterministic linear predictions for same-holdout evaluation tests."""

    def __init__(self, slope: float, intercept: float = 0.0) -> None:
        self._slope = slope
        self._intercept = intercept

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, object] | None = None,
    ) -> pd.Series:
        _ = (context, params)
        return (model_input["x1"].astype(float) * self._slope + self._intercept).rename("pred")


@pytest.mark.parametrize(
    (
        "existing_champion",
        "existing_run_metric",
        "existing_run_missing",
        "candidate_version",
        "candidate_metric",
        "metric_key",
        "min_threshold",
        "min_absolute_change",
        "higher_is_better",
        "data_source",
        "expected_promoted",
        "expected_alias",
        "expected_prev_version",
        "expected_validation_tag",
    ),
    [
        # First-ever model below min_threshold MUST NOT become @Champion
        (
            None,
            None,
            False,
            "1",
            0.55,
            "holdout_roc_auc",
            0.72,
            0.0,
            True,
            "gold",
            False,
            "Challenger",
            None,
            "challenger_review",
        ),
        # Synthetic data models MUST NEVER become @Champion even above threshold
        (
            None,
            None,
            False,
            "1",
            0.95,
            "holdout_roc_auc",
            0.72,
            0.0,
            True,
            "synthetic",
            False,
            "Challenger",
            None,
            "synthetic_blocked",
        ),
        # First-ever real model above min_threshold becomes @Champion
        (None, None, False, "1", 0.84, "holdout_roc_auc", 0.72, 0.0, True, "gold", True, "Champion", None, "approved"),
        # Candidate beats existing @Champion by >= min_absolute_change -> promoted
        (
            SimpleNamespace(version="1", run_id="run_v1"),
            0.84,
            False,
            "2",
            0.89,
            "holdout_roc_auc",
            0.72,
            0.005,
            True,
            "gold",
            True,
            "Champion",
            "1",
            "approved",
        ),
        # Candidate improves by only +0.002 (< min_absolute_change=0.005) -> rejected to @Challenger
        (
            SimpleNamespace(version="1", run_id="run_v1"),
            0.84,
            False,
            "3",
            0.842,
            "holdout_roc_auc",
            0.72,
            0.005,
            True,
            "gold",
            False,
            "Challenger",
            "1",
            "challenger_review",
        ),
        # Candidate loses to existing @Champion -> assigned @Challenger
        (
            SimpleNamespace(version="1", run_id="run_v1"),
            0.84,
            False,
            "3",
            0.78,
            "holdout_roc_auc",
            0.72,
            0.0,
            True,
            "gold",
            False,
            "Challenger",
            "1",
            "challenger_review",
        ),
        # Lower-is-better RMSE beats existing @Champion by >= delta and <= max threshold -> promoted
        (
            SimpleNamespace(version="1", run_id="run_v1"),
            4.5,
            False,
            "2",
            3.8,
            "holdout_rmse",
            5.0,
            0.1,
            False,
            "gold",
            True,
            "Champion",
            "1",
            "approved",
        ),
        # Lower-is-better RMSE exceeds max threshold -> rejected
        (
            SimpleNamespace(version="1", run_id="run_v1"),
            4.5,
            False,
            "3",
            6.2,
            "holdout_rmse",
            5.0,
            0.0,
            False,
            "gold",
            False,
            "Challenger",
            "1",
            "challenger_review",
        ),
        # Existing @Champion has run_id=None -> candidate above min_threshold promoted
        (
            SimpleNamespace(version="1", run_id=None),
            None,
            False,
            "4",
            0.80,
            "holdout_roc_auc",
            0.70,
            0.0,
            True,
            "gold",
            True,
            "Champion",
            "1",
            "approved",
        ),
        # Existing @Champion references a deleted run -> candidate above min_threshold promoted
        (
            SimpleNamespace(version="1", run_id="missing_run"),
            None,
            True,
            "5",
            0.81,
            "holdout_roc_auc",
            0.70,
            0.0,
            True,
            "gold",
            True,
            "Champion",
            "1",
            "approved",
        ),
    ],
    ids=[
        "first_below_gate_rejected",
        "synthetic_data_blocked",
        "first_above_gate_promoted",
        "beats_champion_by_delta_promoted",
        "below_min_absolute_change_rejected",
        "loses_to_champion_challenger",
        "lower_is_better_rmse_promoted",
        "lower_is_better_rmse_above_gate_rejected",
        "champion_without_run_id_promoted",
        "champion_missing_run_promoted",
    ],
)
def test_promote_champion_challenger_alias_scenarios(
    missing_alias_error: Callable[[str], MlflowException],
    existing_champion: SimpleNamespace | None,
    existing_run_metric: float | None,
    existing_run_missing: bool,
    candidate_version: str,
    candidate_metric: float,
    metric_key: str,
    min_threshold: float,
    min_absolute_change: float,
    higher_is_better: bool,
    data_source: str,
    expected_promoted: bool,
    expected_alias: str,
    expected_prev_version: str | None,
    expected_validation_tag: str,
) -> None:
    """Verify `@Champion`, `@PreviousChampion`, `@Challenger`, absolute gate, delta gate, and synthetic blocking."""
    mock_client = MagicMock()
    if existing_champion is None:
        mock_client.get_model_version_by_alias.side_effect = missing_alias_error("Alias Champion not found")
    else:
        mock_client.get_model_version_by_alias.return_value = existing_champion
        if existing_run_missing:
            mock_client.get_run.side_effect = missing_alias_error("Run not found")
        elif existing_run_metric is not None:
            mock_client.get_run.return_value = SimpleNamespace(
                data=SimpleNamespace(metrics={metric_key: existing_run_metric})
            )

    decision = promote_champion_challenger_alias(
        client=mock_client,
        uc_model_name="cat.ml_models.test_xgb",
        candidate_version=candidate_version,
        candidate_metric=candidate_metric,
        metric_key=metric_key,
        min_threshold=min_threshold,
        higher_is_better=higher_is_better,
        data_source=data_source,
        min_absolute_change=min_absolute_change,
    )
    assert decision.promoted_to_champion is expected_promoted
    assert decision.assigned_alias == expected_alias
    assert decision.previous_champion_version == expected_prev_version
    assert decision.champion_re_evaluated is False
    if existing_run_metric is not None:
        assert decision.previous_champion_metric == pytest.approx(existing_run_metric)

    mock_client.set_model_version_tag.assert_any_call(
        name="cat.ml_models.test_xgb",
        version=candidate_version,
        key="validation_status",
        value=expected_validation_tag,
    )


@pytest.mark.parametrize(
    ("failing_call", "error_msg"),
    [
        ("get_model_version_by_alias", "Databricks API 503"),
        ("get_run", "Tracking server timeout"),
    ],
    ids=["alias_lookup_api_error", "run_lookup_api_error"],
)
def test_promote_champion_challenger_propagates_transient_api_errors(
    failing_call: str,
    error_msg: str,
) -> None:
    """Verify transient non-missing API errors propagate rather than silently triggering promotion."""
    mock_client = MagicMock()
    if failing_call == "get_model_version_by_alias":
        mock_client.get_model_version_by_alias.side_effect = ConnectionError(error_msg)
    else:
        mock_client.get_model_version_by_alias.return_value = SimpleNamespace(version="1", run_id="run_v1")
        mock_client.get_run.side_effect = ConnectionError(error_msg)

    with pytest.raises(ConnectionError, match=error_msg):
        promote_champion_challenger_alias(
            client=mock_client,
            uc_model_name="cat.ml_models.churn_xgb",
            candidate_version="2",
            candidate_metric=0.84,
            metric_key="holdout_roc_auc",
            min_threshold=0.72,
        )


@pytest.mark.mlflow
def test_same_holdout_champion_re_evaluation_and_validation(
    mlflow_tracking_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify real MLflow 3 `mlflow.models.evaluate()` + `mlflow.validate_evaluation_results()` on shared holdout."""
    _ = mlflow_tracking_uri
    model_name = "test_same_holdout_reg"
    holdout_df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    sample_x = holdout_df[["x1"]].iloc[:2]
    sig = infer_signature(sample_x, (sample_x["x1"] * 2.0).to_numpy())
    client = MlflowClient()

    # Register v1 (noisy slope=1.5) and promote as initial @Champion
    with mlflow.start_run(run_name="v1_baseline"):
        v1_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=_LinearScorePyFunc(slope=1.5),
            signature=sig,
            registered_model_name=model_name,
        )
    v1_decision = promote_champion_challenger_alias(
        client=client,
        uc_model_name=model_name,
        candidate_version=str(v1_info.registered_model_version),
        candidate_metric=0.0,
        metric_key="holdout_r2",
        min_threshold=0.50,
        higher_is_better=True,
        min_absolute_change=0.005,
        candidate_model_uri=v1_info.model_uri,
        holdout_df=holdout_df,
        feature_cols=("x1",),
        target_col="target",
        task_type="regression",
    )
    assert v1_decision.promoted_to_champion is True
    assert v1_decision.assigned_alias == "Champion"
    assert v1_decision.champion_re_evaluated is False

    # Register v2 (exact slope=2.0 -> R2=1.0) -> re-evaluates v1 @Champion on same holdout & promotes v2
    with mlflow.start_run(run_name="v2_stronger"):
        v2_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=_LinearScorePyFunc(slope=2.0),
            signature=sig,
            registered_model_name=model_name,
        )
    v2_decision = promote_champion_challenger_alias(
        client=client,
        uc_model_name=model_name,
        candidate_version=str(v2_info.registered_model_version),
        candidate_metric=0.0,
        metric_key="holdout_r2",
        min_threshold=0.50,
        higher_is_better=True,
        min_absolute_change=0.005,
        candidate_model_uri=v2_info.model_uri,
        holdout_df=holdout_df,
        feature_cols=("x1",),
        target_col="target",
        task_type="regression",
    )
    assert v2_decision.champion_re_evaluated is True
    assert v2_decision.promoted_to_champion is True
    assert v2_decision.assigned_alias == "Champion"
    assert v2_decision.previous_champion_version == str(v1_info.registered_model_version)
    assert v2_decision.candidate_metric == pytest.approx(1.0)

    # Register v3 (weaker slope=1.2) -> re-evaluates v2 @Champion on same holdout & blocks promotion
    with mlflow.start_run(run_name="v3_weaker"):
        v3_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=_LinearScorePyFunc(slope=1.2),
            signature=sig,
            registered_model_name=model_name,
        )
    v3_eval = evaluate_model_on_holdout(
        model_or_uri=v3_info.model_uri,
        holdout_df=holdout_df,
        feature_cols=("x1",),
        target_col="target",
        task_type="regression",
    )
    v3_decision = promote_champion_challenger_alias(
        client=client,
        uc_model_name=model_name,
        candidate_version=str(v3_info.registered_model_version),
        candidate_metric=float(v3_eval.metrics["holdout_r2"]),
        metric_key="holdout_r2",
        min_threshold=0.50,
        higher_is_better=True,
        min_absolute_change=0.005,
        candidate_eval_result=v3_eval,
        holdout_df=holdout_df,
        feature_cols=("x1",),
        target_col="target",
        task_type="regression",
    )
    assert v3_decision.champion_re_evaluated is True
    assert v3_decision.promoted_to_champion is False
    assert v3_decision.assigned_alias == "Challenger"

    # Verify graceful fallback + warning log if loading incumbent @Champion fails
    with (
        caplog.at_level("WARNING"),
        patch(
            "src.python.modules.evaluation.model_governance.evaluate_model_on_holdout", side_effect=RuntimeError("boom")
        ),
    ):
        fallback_decision = promote_champion_challenger_alias(
            client=client,
            uc_model_name=model_name,
            candidate_version=str(v3_info.registered_model_version),
            candidate_metric=0.90,
            metric_key="holdout_r2",
            min_threshold=0.50,
            higher_is_better=True,
            candidate_eval_result=v3_eval,
            holdout_df=holdout_df,
            feature_cols=("x1",),
            target_col="target",
            task_type="regression",
        )
    assert fallback_decision.champion_re_evaluated is False
    assert "Unable to re-evaluate existing @Champion" in caplog.text

    # Direct classification validation check via validate_candidate_evaluation
    cls_df = pd.DataFrame({"x1": [0.1, 0.2, 0.8, 0.9], "y": [0, 0, 1, 1]})
    cls_eval = evaluate_model_on_holdout(
        model_or_uri=lambda df: df["x1"].to_numpy(),
        holdout_df=cls_df,
        feature_cols=("x1",),
        target_col="y",
        task_type="classification",
    )
    assert validate_candidate_evaluation(cls_eval, None, "holdout_roc_auc", min_threshold=0.90) is True
    assert validate_candidate_evaluation(cls_eval, None, "holdout_roc_auc", min_threshold=1.01) is False
