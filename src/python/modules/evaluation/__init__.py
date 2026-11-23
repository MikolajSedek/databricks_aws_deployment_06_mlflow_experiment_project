"""MLflow 3 model evaluation, SHAP explainability, feature importance artifacts, and UC governance."""

from src.python.modules.evaluation.model_governance import (
    PromotionDecision,
    evaluate_model_on_holdout,
    log_feature_importance_artifact,
    promote_champion_challenger_alias,
    validate_candidate_evaluation,
)
from src.python.modules.evaluation.shap_explainability import (
    ShapExplainabilityResult,
    compute_shap_explanation,
    log_shap_explainability_artifacts,
)
from src.python.modules.evaluation.uc_tracing import (
    TraceMigrationOutcome,
    configure_uc_experiment_tracing,
    is_databricks_tracking_uri,
    migrate_experiment_traces_to_uc,
    migrate_v1_uc_traces_to_v2,
)

__all__ = [
    "PromotionDecision",
    "ShapExplainabilityResult",
    "TraceMigrationOutcome",
    "compute_shap_explanation",
    "configure_uc_experiment_tracing",
    "evaluate_model_on_holdout",
    "is_databricks_tracking_uri",
    "log_feature_importance_artifact",
    "log_shap_explainability_artifacts",
    "migrate_experiment_traces_to_uc",
    "migrate_v1_uc_traces_to_v2",
    "promote_champion_challenger_alias",
    "validate_candidate_evaluation",
]
