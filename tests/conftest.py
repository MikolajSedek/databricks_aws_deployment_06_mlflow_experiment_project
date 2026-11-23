"""Shared pytest fixtures including a worker-isolated PySpark SparkSession and fast isolated MLflow tracking."""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

# Ensure project root and src/ directory are on sys.path for local testing across all test modules
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
for candidate_path in (str(_SRC_ROOT), str(_REPO_ROOT)):
    if candidate_path not in sys.path:
        sys.path.insert(0, candidate_path)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


@pytest.fixture(autouse=True)
def _isolate_mlflow_tracing_by_default(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Keep MLflow tracing disabled unless the test explicitly requests `mlflow_tracking_uri`.

    Prevents non-MLflow unit tests that invoke traced helpers (`mlflow.start_span`) from
    opening the default `./mlflow.db` in the working directory, eliminating cross-worker
    SQLite lock contention under `pytest-xdist` and keeping the repo root clean.
    """
    if "mlflow_tracking_uri" not in request.fixturenames:
        mlflow.tracing.disable()
    yield


@pytest.fixture(scope="session")
def spark(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Generator[SparkSession, None, None]:
    """Lazily create a fast, worker-isolated local[1] SparkSession for unit testing and stop it on teardown."""
    warehouse_dir = tmp_path_factory.mktemp(f"spark_warehouse_{worker_id}")
    local_dir = tmp_path_factory.mktemp(f"spark_local_{worker_id}")
    session = (
        SparkSession.builder.master("local[1]")
        .appName(f"pytest-mlflow3-xgboost-{worker_id}")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.warehouse.dir", warehouse_dir.as_uri())
        .config("spark.local.dir", str(local_dir))
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture(scope="session")
def _mlflow_template_db(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
) -> Path:
    """Create one pre-migrated MLflow 3 SQLite database per xdist worker to avoid repeated Alembic setup."""
    template_dir = tmp_path_factory.mktemp(f"mlflow_template_{worker_id}")
    template_db = template_dir / "template_mlflow.db"
    template_uri = f"sqlite:///{template_db.as_posix()}"

    client = MlflowClient(tracking_uri=template_uri, registry_uri=template_uri)
    client.search_experiments()
    client.search_registered_models()
    return template_db


@pytest.fixture()
def mlflow_tracking_uri(
    tmp_path: Path,
    _mlflow_template_db: Path,
) -> Generator[str, None, None]:
    """Configure an isolated local SQLite MLflow 3 tracking & registry URI cloned from the worker template DB."""
    prev_tracking_uri = mlflow.get_tracking_uri()
    prev_registry_uri = mlflow.get_registry_uri()

    target_db = tmp_path / "mlflow_test.db"
    # Atomic hot backup from the pre-migrated template DB (~2ms vs ~2.7s fresh Alembic migration).
    with (
        closing(sqlite3.connect(_mlflow_template_db)) as src_conn,
        closing(sqlite3.connect(target_db)) as dst_conn,
    ):
        src_conn.backup(dst_conn)

    artifact_dir = (tmp_path / "mlruns").as_uri()
    tracking_uri = f"sqlite:///{target_db.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.tracing.enable()
    exp_name = f"pytest_exp_{tmp_path.name}"
    mlflow.create_experiment(exp_name, artifact_location=artifact_dir)
    mlflow.set_experiment(exp_name)
    try:
        yield tracking_uri
    finally:
        mlflow.tracing.disable()
        mlflow.set_tracking_uri(prev_tracking_uri)
        mlflow.set_registry_uri(prev_registry_uri)


@pytest.fixture()
def missing_alias_error() -> Callable[[str], MlflowException]:
    """Return a factory creating `MlflowException` with MLflow's `RESOURCE_DOES_NOT_EXIST` code."""

    def _factory(message: str = "Alias not found") -> MlflowException:
        return MlflowException(message, error_code=RESOURCE_DOES_NOT_EXIST)

    return _factory
