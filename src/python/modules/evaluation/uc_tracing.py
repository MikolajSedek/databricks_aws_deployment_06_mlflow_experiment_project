"""Unity Catalog OpenTelemetry trace storage setup and legacy experiment trace migration.

Implements the official Databricks MLflow 3 Unity Catalog tracing and migration patterns:
- https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
- https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-to-uc
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.entities.trace_location import UnityCatalog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceMigrationOutcome:
    """Immutable summary of a Unity Catalog trace setup and migration operation."""

    source_experiment_id: str | None
    target_experiment_id: str
    uc_trace_prefix_fqn: str
    migration_method: str
    migrated_trace_count: int
    migrated_trace_ids: tuple[str, ...]


def is_databricks_tracking_uri(tracking_uri: str | None = None) -> bool:
    """Return True if the active MLflow tracking URI points to a Databricks workspace."""
    uri = (tracking_uri if tracking_uri is not None else mlflow.get_tracking_uri()).strip().lower()
    return uri == "databricks" or uri.startswith("databricks://")


def configure_uc_experiment_tracing(
    experiment_name: str,
    catalog_name: str,
    schema_name: str,
    table_prefix: str,
    sql_warehouse_id: str | None = None,
) -> str:
    """Create or activate an MLflow experiment bound to a Unity Catalog OpenTelemetry trace location.

    When running against a Databricks tracking server, binds `experiment_name` to
    `UnityCatalog(catalog_name=catalog_name, schema_name=schema_name, table_prefix=table_prefix)`
    so all spans are persisted in `<table_prefix>_otel_spans`, `<table_prefix>_otel_logs`,
    `<table_prefix>_otel_annotations`, and `<table_prefix>_otel_metrics`. On local SQLite/file
    tracking URIs, activates the experiment without a UC location binding.
    """
    if sql_warehouse_id and not os.environ.get("MLFLOW_TRACING_SQL_WAREHOUSE_ID"):
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = sql_warehouse_id

    if is_databricks_tracking_uri():
        uc_location = UnityCatalog(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_prefix=table_prefix,
        )
        try:
            exp = mlflow.set_experiment(experiment_name=experiment_name, trace_location=uc_location)
            return str(exp.experiment_id)
        except Exception as exc:
            # An existing experiment may already be bound to a trace location; log loudly and activate it anyway.
            logger.warning(
                "Failed to bind Unity Catalog trace location %s.%s.%s for experiment '%s'; "
                "falling back to standard experiment activation: %s",
                catalog_name,
                schema_name,
                table_prefix,
                experiment_name,
                exc,
            )

    exp = mlflow.set_experiment(experiment_name=experiment_name)
    return str(exp.experiment_id)


def _replay_trace_spans(source_trace: Any, source_experiment_id: str) -> str | None:
    """Replay a legacy experiment trace's root and child spans into the active UC-bound experiment."""
    spans = list(getattr(getattr(source_trace, "data", None), "spans", []) or [])
    info = getattr(source_trace, "info", None)
    src_trace_id = str(getattr(info, "trace_id", "unknown_trace"))
    if not spans:
        return None

    root_span_data = next((s for s in spans if getattr(s, "parent_id", None) is None), spans[0])
    child_spans_data = [s for s in spans if s is not root_span_data]

    with mlflow.start_span(
        name=str(getattr(root_span_data, "name", "migrated_root_span")),
        span_type=str(getattr(root_span_data, "span_type", "CHAIN")),
    ) as root_span:
        root_span.set_inputs(getattr(root_span_data, "inputs", None) or {"migrated_from_trace_id": src_trace_id})
        root_span.set_attributes(
            {
                "migrated_from_trace_id": src_trace_id,
                "migrated_from_experiment_id": source_experiment_id,
            }
        )
        for child_data in child_spans_data:
            with mlflow.start_span(
                name=str(getattr(child_data, "name", "migrated_child_span")),
                span_type=str(getattr(child_data, "span_type", "UNKNOWN")),
            ) as child_span:
                if getattr(child_data, "inputs", None) is not None:
                    child_span.set_inputs(child_data.inputs)
                if getattr(child_data, "outputs", None) is not None:
                    child_span.set_outputs(child_data.outputs)
        root_span.set_outputs(getattr(root_span_data, "outputs", None) or {"status": "migrated_to_uc"})

    return mlflow.get_last_active_trace_id()


def _try_databricks_sdk_trace_migration(
    source_experiment_name: str,
    target_experiment_name: str,
    source_exp_id: str,
    target_exp_id: str,
    prefix_fqn: str,
    start_time_ms: int | None,
) -> TraceMigrationOutcome | None:
    """Attempt native server-side trace migration via `databricks.migrations`, returning None to fall back to replay."""
    try:
        dbx_mod = importlib.import_module("databricks.migrations.migrate_traces_to_uc")
        kwargs: dict[str, Any] = {
            "source_experiment_id": source_exp_id,
            "target_experiment_id": target_exp_id,
        }
        if start_time_ms is not None:
            kwargs["start_time_ms"] = start_time_ms
        dbx_mod.run(**kwargs)
        mlflow.set_experiment_tags(
            {
                "migrated_from_experiment_id": source_exp_id,
                "uc_trace_prefix_fqn": prefix_fqn,
            }
        )
        return TraceMigrationOutcome(
            source_experiment_id=source_exp_id,
            target_experiment_id=target_exp_id,
            uc_trace_prefix_fqn=prefix_fqn,
            migration_method="databricks_migrations_sdk",
            migrated_trace_count=0,
            migrated_trace_ids=(),
        )
    except ModuleNotFoundError:
        logger.info(
            "databricks.migrations SDK is not installed; falling back to MLflow trace replay for '%s' -> '%s'.",
            source_experiment_name,
            target_experiment_name,
        )
    except Exception as exc:
        logger.warning(
            "databricks.migrations.migrate_traces_to_uc failed for '%s' -> '%s'; falling back to trace replay: %s",
            source_experiment_name,
            target_experiment_name,
            exc,
        )
    return None


def _replay_experiment_traces_to_target(
    source_exp_id: str,
    target_exp_id: str,
    prefix_fqn: str,
    start_time_ms: int | None,
) -> TraceMigrationOutcome:
    """Query source experiment traces via MLflow API and replay each span tree into the active UC experiment."""
    filter_str = f"trace.timestamp_ms >= {start_time_ms}" if start_time_ms is not None else None
    source_traces = mlflow.search_traces(
        locations=[source_exp_id],
        filter_string=filter_str,
        return_type="list",
        include_spans=True,
    )
    migrated_ids = [
        new_id for src_trace in source_traces if (new_id := _replay_trace_spans(src_trace, source_exp_id)) is not None
    ]

    if hasattr(mlflow, "flush_trace_async_logging"):
        mlflow.flush_trace_async_logging()

    # Stamp idempotency tags on the target experiment so subsequent runs skip already-migrated history.
    mlflow.set_experiment_tags(
        {
            "migrated_from_experiment_id": source_exp_id,
            "migrated_trace_count": str(len(migrated_ids)),
            "uc_trace_prefix_fqn": prefix_fqn,
        }
    )
    return TraceMigrationOutcome(
        source_experiment_id=source_exp_id,
        target_experiment_id=target_exp_id,
        uc_trace_prefix_fqn=prefix_fqn,
        migration_method="mlflow_trace_replay_to_uc",
        migrated_trace_count=len(migrated_ids),
        migrated_trace_ids=tuple(migrated_ids),
    )


def migrate_experiment_traces_to_uc(
    source_experiment_name: str,
    target_experiment_name: str,
    catalog_name: str,
    schema_name: str,
    table_prefix: str,
    sql_warehouse_id: str | None = None,
    start_time_ms: int | None = None,
    force_remigrate: bool = False,
) -> TraceMigrationOutcome:
    """Migrate existing MLflow traces from legacy experiment storage to Unity Catalog Delta tables.

    Follows the 3-step idempotent migration protocol:
    1. Activate/bind the UC-backed target experiment and skip if no distinct legacy source exists.
    2. Short-circuit via experiment tags when the source was already migrated (unless `force_remigrate=True`).
    3. Prefer `databricks.migrations.migrate_traces_to_uc`, falling back to MLflow span replay.
    """
    prefix_fqn = f"{catalog_name}.{schema_name}.{table_prefix}"
    target_exp_id = configure_uc_experiment_tracing(
        experiment_name=target_experiment_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_prefix=table_prefix,
        sql_warehouse_id=sql_warehouse_id,
    )

    source_exp = mlflow.get_experiment_by_name(source_experiment_name)
    if source_exp is None or str(source_exp.experiment_id) == target_exp_id:
        return TraceMigrationOutcome(
            source_experiment_id=None,
            target_experiment_id=target_exp_id,
            uc_trace_prefix_fqn=prefix_fqn,
            migration_method="no_source_experiment",
            migrated_trace_count=0,
            migrated_trace_ids=(),
        )

    source_exp_id = str(source_exp.experiment_id)
    target_exp = mlflow.get_experiment(target_exp_id)
    target_tags = dict(getattr(target_exp, "tags", {}) or {})
    if not force_remigrate and target_tags.get("migrated_from_experiment_id") == source_exp_id:
        return TraceMigrationOutcome(
            source_experiment_id=source_exp_id,
            target_experiment_id=target_exp_id,
            uc_trace_prefix_fqn=prefix_fqn,
            migration_method="already_migrated_to_uc",
            migrated_trace_count=int(target_tags.get("migrated_trace_count", "0")),
            migrated_trace_ids=(),
        )

    sdk_outcome = _try_databricks_sdk_trace_migration(
        source_experiment_name=source_experiment_name,
        target_experiment_name=target_experiment_name,
        source_exp_id=source_exp_id,
        target_exp_id=target_exp_id,
        prefix_fqn=prefix_fqn,
        start_time_ms=start_time_ms,
    )
    if sdk_outcome is not None:
        return sdk_outcome

    return _replay_experiment_traces_to_target(
        source_exp_id=source_exp_id,
        target_exp_id=target_exp_id,
        prefix_fqn=prefix_fqn,
        start_time_ms=start_time_ms,
    )


def migrate_v1_uc_traces_to_v2(
    v1_source_schema: str,
    v2_destination_prefix: str,
) -> str:
    """Migrate legacy schema-linked UC trace tables (`catalog.schema`) to table-prefix format (`catalog.schema.prefix`)."""
    try:
        v1_v2_mod = importlib.import_module("databricks.migrations.v1_to_v2")
        migration = v1_v2_mod.V1ToV2SqlMigration(
            v1_source_schema=v1_source_schema,
            v2_destination_prefix=v2_destination_prefix,
        )
        migration.run()
        return "migrated_v1_to_v2_sql"
    except ModuleNotFoundError:
        logger.info("databricks.migrations.v1_to_v2 SDK is not installed; skipping v1->v2 UC trace migration.")
        return "v1_to_v2_sdk_not_installed"
