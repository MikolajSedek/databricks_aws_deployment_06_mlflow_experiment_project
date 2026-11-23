"""SHAP (`TreeExplainer`) model explainability and MLflow artifact logging for XGBoost models.

Computes exact TreeSHAP global and local feature attributions on a deterministic evaluation
sample, renders headless Matplotlib visualizations (beeswarm, mean |SHAP| bar chart, and
top-instance waterfall plot), and logs structured CSV/JSON summaries alongside PNG figures
to the MLflow 3 run artifact store under `explainability/`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import shap
from mlflow.tracking import MlflowClient
from xgboost import XGBModel

# Force non-interactive backend before importing pyplot so headless Serverless/CI runners never fail.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ShapExplainabilityResult:
    """Immutable summary of SHAP feature attributions and logged MLflow explainability artifacts."""

    summary_df: pd.DataFrame
    expected_value: float
    sample_size: int
    artifact_files: tuple[str, ...]


def _extract_scalar_expected_value(base_values: Any) -> float:
    """Coerce SHAP `base_values` (scalar or ndarray) into a single scalar float."""
    arr = np.asarray(base_values, dtype=float)
    if arr.size == 0:
        return 0.0
    return round(float(np.mean(arr)), 6)


def _normalize_shap_explanation(explanation: Any) -> Any:
    """Ensure a SHAP `Explanation` has 2D `(n_samples, n_features)` values for binary/regression plots."""
    values = np.asarray(explanation.values)
    # Some SHAP/XGBoost versions return 3D `(n, features, 2)` for binary models; slice positive class.
    if values.ndim == 3:
        return explanation[:, :, -1]
    return explanation


def compute_shap_explanation(
    model: XGBModel,
    X_eval: pd.DataFrame,
    feature_cols: tuple[str, ...],
    max_samples: int = 500,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame, float]:
    """Compute exact `shap.TreeExplainer` attributions and a ranked global summary DataFrame.

    Args:
        model: Fitted `XGBClassifier` or `XGBRegressor` estimator.
        X_eval: Evaluation feature DataFrame (e.g., holdout test split).
        feature_cols: Ordered canonical feature column names.
        max_samples: Maximum number of rows to sample deterministically for SHAP computation.
        random_state: Deterministic sampling seed.

    Returns:
        Tuple of `(shap_explanation, summary_df, expected_value)`.
    """
    if X_eval.empty:
        raise ValueError("Cannot compute SHAP explainability on an empty evaluation DataFrame.")

    eval_features = X_eval[list(feature_cols)].astype(float)
    if len(eval_features) > max_samples:
        eval_features = eval_features.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    else:
        eval_features = eval_features.reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    explanation = _normalize_shap_explanation(explainer(eval_features))

    shap_matrix = np.asarray(explanation.values, dtype=float)
    summary_df = (
        pd.DataFrame(
            {
                "feature": list(feature_cols),
                "mean_abs_shap": np.round(np.mean(np.abs(shap_matrix), axis=0), 6),
                "mean_shap": np.round(np.mean(shap_matrix, axis=0), 6),
                "std_shap": np.round(np.std(shap_matrix, axis=0), 6),
            }
        )
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary_df["rank"] = range(1, len(summary_df) + 1)

    expected_value = _extract_scalar_expected_value(explanation.base_values)
    return explanation, summary_df, expected_value


def _write_shap_tabular_artifacts(
    summary_df: pd.DataFrame,
    expected_value: float,
    sample_size: int,
    n_features: int,
    max_display: int,
    output_dir: Path,
    artifact_prefix: str,
) -> tuple[str, str]:
    """Write ranked SHAP CSV and JSON metadata summary files into `output_dir`."""
    csv_file = f"{artifact_prefix}_feature_importance.csv"
    json_file = f"{artifact_prefix}_summary.json"
    summary_df.to_csv(output_dir / csv_file, index=False)

    summary_payload = {
        "expected_value": expected_value,
        "sample_size": sample_size,
        "n_features": n_features,
        "top_features": summary_df.head(max_display).to_dict(orient="records"),
    }
    (output_dir / json_file).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return csv_file, json_file


def _save_shap_plots(
    explanation: Any,
    output_dir: Path,
    artifact_prefix: str,
    max_display: int = 15,
) -> tuple[str, ...]:
    """Render and save global beeswarm, bar importance, and local waterfall SHAP PNG plots."""
    beeswarm_file = f"{artifact_prefix}_beeswarm.png"
    bar_file = f"{artifact_prefix}_bar.png"
    waterfall_file = f"{artifact_prefix}_waterfall_top_example.png"

    try:
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / beeswarm_file, dpi=150, bbox_inches="tight")
        plt.close("all")

        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / bar_file, dpi=150, bbox_inches="tight")
        plt.close("all")

        # Select the evaluation row with largest total |SHAP| magnitude as an informative local waterfall example.
        shap_values = np.asarray(explanation.values, dtype=float)
        top_idx = int(np.argmax(np.sum(np.abs(shap_values), axis=1)))
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[top_idx], max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / waterfall_file, dpi=150, bbox_inches="tight")
        plt.close("all")
    finally:
        # Always close Matplotlib figures to prevent memory leaks across repeated runs or tests.
        plt.close("all")

    return (beeswarm_file, bar_file, waterfall_file)


def _upload_shap_artifact_dir(artifact_dir: Path, artifact_subdir: str, run_id: str | None) -> None:
    """Upload a directory of SHAP artifacts to MLflow via explicit `run_id` or the currently active run."""
    if run_id is not None:
        MlflowClient().log_artifacts(run_id, str(artifact_dir), artifact_path=artifact_subdir)
    elif mlflow.active_run() is not None:
        mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_subdir)


def log_shap_explainability_artifacts(
    model: XGBModel,
    X_eval: pd.DataFrame,
    feature_cols: tuple[str, ...],
    run_id: str | None = None,
    artifact_prefix: str = "shap",
    artifact_subdir: str = "explainability",
    max_samples: int = 500,
    random_state: int = 42,
    max_display: int = 15,
) -> ShapExplainabilityResult:
    """Compute TreeSHAP explanations and log CSV, JSON, and PNG plots as MLflow artifacts."""
    with mlflow.start_span(name="log_shap_explainability_artifacts", span_type="CHAIN") as span:
        span.set_inputs(
            {
                "artifact_prefix": artifact_prefix,
                "artifact_subdir": artifact_subdir,
                "eval_rows": len(X_eval),
                "n_features": len(feature_cols),
                "max_samples": max_samples,
            }
        )
        explanation, summary_df, expected_value = compute_shap_explanation(
            model=model,
            X_eval=X_eval,
            feature_cols=feature_cols,
            max_samples=max_samples,
            random_state=random_state,
        )
        sample_size = int(np.asarray(explanation.values).shape[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            csv_file, json_file = _write_shap_tabular_artifacts(
                summary_df=summary_df,
                expected_value=expected_value,
                sample_size=sample_size,
                n_features=len(feature_cols),
                max_display=max_display,
                output_dir=out_dir,
                artifact_prefix=artifact_prefix,
            )
            plot_files = _save_shap_plots(
                explanation=explanation,
                output_dir=out_dir,
                artifact_prefix=artifact_prefix,
                max_display=max_display,
            )
            artifact_files = (csv_file, json_file, *plot_files)
            _upload_shap_artifact_dir(out_dir, artifact_subdir, run_id)

        top_feature = str(summary_df.iloc[0]["feature"]) if not summary_df.empty else "none"
        span.set_outputs(
            {
                "sample_size": sample_size,
                "expected_value": expected_value,
                "top_shap_feature": top_feature,
                "artifact_count": len(artifact_files),
            }
        )

    return ShapExplainabilityResult(
        summary_df=summary_df,
        expected_value=expected_value,
        sample_size=sample_size,
        artifact_files=artifact_files,
    )
