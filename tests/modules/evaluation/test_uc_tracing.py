"""Unit tests for Unity Catalog OpenTelemetry trace configuration and migration helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from src.python.modules.evaluation.uc_tracing import (
    configure_uc_experiment_tracing,
    is_databricks_tracking_uri,
    migrate_experiment_traces_to_uc,
    migrate_v1_uc_traces_to_v2,
)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("databricks", True),
        ("databricks://dev", True),
        ("sqlite:///tmp/mlflow.db", False),
        ("http://127.0.0.1:5000", False),
    ],
    ids=["databricks_bare", "databricks_profile", "sqlite_local", "http_tracking"],
)
def test_is_databricks_tracking_uri(uri: str, expected: bool) -> None:
    """Verify Databricks tracking URI detection across Databricks and local schemes."""
    assert is_databricks_tracking_uri(uri) is expected


def test_configure_uc_experiment_tracing_databricks_and_conflict_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """Verify UC experiment trace binding on Databricks and logged warning on UC bind conflicts."""
    with (
        caplog.at_level("WARNING"),
        patch("src.python.modules.evaluation.uc_tracing.is_databricks_tracking_uri", return_value=True),
        patch("src.python.modules.evaluation.uc_tracing.mlflow.set_experiment") as mock_set_exp,
    ):
        mock_set_exp.return_value = SimpleNamespace(experiment_id="dbx_exp_1")
        assert configure_uc_experiment_tracing("/Shared/dbx_exp", "cat", "ml_traces", "pfx") == "dbx_exp_1"

        mock_set_exp.side_effect = [RuntimeError("UC bind conflict"), SimpleNamespace(experiment_id="dbx_exp_2")]
        assert configure_uc_experiment_tracing("/Shared/dbx_exp2", "cat", "ml_traces", "pfx") == "dbx_exp_2"

    assert "Failed to bind Unity Catalog trace location" in caplog.text


@pytest.mark.mlflow
def test_migrate_experiment_traces_to_uc_flows(
    mlflow_tracking_uri: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify trace replay, idempotency, missing-source handling, Databricks SDK migration, and warnings."""
    _ = mlflow_tracking_uri
    # Register env key with monkeypatch (empty string is falsy) so teardown always restores/removes it.
    monkeypatch.setenv("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "")
    local_exp_id = configure_uc_experiment_tracing(
        experiment_name="/Shared/local_source_exp",
        catalog_name="test_cat",
        schema_name="ml_traces",
        table_prefix="local_src",
        sql_warehouse_id="wh_123",
    )
    assert local_exp_id

    with mlflow.start_span(name="sample_source_root", span_type="CHAIN") as r_span:
        r_span.set_inputs({"x": 1})
        with mlflow.start_span(name="sample_source_child", span_type="TOOL") as c_span:
            c_span.set_inputs({"y": 2})
            c_span.set_outputs({"z": 3})
        r_span.set_outputs({"ok": True})

    outcome_replay = migrate_experiment_traces_to_uc(
        source_experiment_name="/Shared/local_source_exp",
        target_experiment_name="/Shared/local_target_uc_exp",
        catalog_name="test_cat",
        schema_name="ml_traces",
        table_prefix="local_tgt",
    )
    assert outcome_replay.migration_method == "mlflow_trace_replay_to_uc"
    assert outcome_replay.migrated_trace_count >= 1

    outcome_idempotent = migrate_experiment_traces_to_uc(
        source_experiment_name="/Shared/local_source_exp",
        target_experiment_name="/Shared/local_target_uc_exp",
        catalog_name="test_cat",
        schema_name="ml_traces",
        table_prefix="local_tgt",
    )
    assert outcome_idempotent.migration_method == "already_migrated_to_uc"

    outcome_missing_src = migrate_experiment_traces_to_uc(
        source_experiment_name="/Shared/nonexistent_source_exp",
        target_experiment_name="/Shared/local_target_uc_exp",
        catalog_name="test_cat",
        schema_name="ml_traces",
        table_prefix="local_tgt",
    )
    assert outcome_missing_src.migration_method == "no_source_experiment"

    # Runtime error inside Databricks migration SDK logs warning and falls back cleanly to trace replay
    failing_sdk = MagicMock()
    failing_sdk.run.side_effect = RuntimeError("UC table permissions denied")
    with (
        caplog.at_level("WARNING"),
        patch("src.python.modules.evaluation.uc_tracing.importlib.import_module", return_value=failing_sdk),
    ):
        fallback_outcome = migrate_experiment_traces_to_uc(
            source_experiment_name="/Shared/local_source_exp",
            target_experiment_name="/Shared/local_target_err_fallback",
            catalog_name="test_cat",
            schema_name="ml_traces",
            table_prefix="err_tgt",
            force_remigrate=True,
        )
    assert fallback_outcome.migration_method == "mlflow_trace_replay_to_uc"
    assert "databricks.migrations.migrate_traces_to_uc failed" in caplog.text

    mock_dbx_mig = MagicMock()
    with patch("src.python.modules.evaluation.uc_tracing.importlib.import_module", return_value=mock_dbx_mig):
        sdk_outcome = migrate_experiment_traces_to_uc(
            source_experiment_name="/Shared/local_source_exp",
            target_experiment_name="/Shared/local_target_sdk_exp",
            catalog_name="test_cat",
            schema_name="ml_traces",
            table_prefix="sdk_tgt",
            start_time_ms=1000,
            force_remigrate=True,
        )
        assert sdk_outcome.migration_method == "databricks_migrations_sdk"
        mock_dbx_mig.run.assert_called_once()

        assert migrate_v1_uc_traces_to_v2("cat.old_schema", "cat.ml_traces.new_pfx") == "migrated_v1_to_v2_sql"

    with patch(
        "src.python.modules.evaluation.uc_tracing.importlib.import_module",
        side_effect=ModuleNotFoundError("no databricks.migrations"),
    ):
        assert migrate_v1_uc_traces_to_v2("cat.old_schema", "cat.ml_traces.new_pfx") == "v1_to_v2_sdk_not_installed"
