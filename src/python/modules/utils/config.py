"""Immutable domain configuration, feature contracts, and Unity Catalog identifiers for MLflow 3 & XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Primary keys, point-in-time snapshot key, and target labels from Project 05 Gold ML Feature Views
COL_USER_ID: Final[str] = "user_id"
COL_SK_USER: Final[str] = "sk_user"
COL_PRODUCT_ID: Final[str] = "product_id"
COL_SK_PRODUCT: Final[str] = "sk_product"
COL_SNAPSHOT_DATE: Final[str] = "snapshot_date"

TARGET_CHURN_COL: Final[str] = "is_churn_candidate"
TARGET_AFFINITY_COL: Final[str] = "affinity_index_score"

# Upstream deterministic formula inputs excluded from model features to prevent direct target leakage
# (Project 05 derives `is_churn_candidate` from `days_since_last_order`, `days_since_last_event`,
# `total_orders`, and `churn_risk_score`; `affinity_index_score` is derived from
# `repeat_purchase_ratio`, `cart_to_purchase_rate`, and `category_revenue_share_pct`).
LEAKY_CHURN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "days_since_last_order",
        "days_since_last_event",
        "total_orders",
        "orders_per_session_ratio",
        "churn_risk_score",
        TARGET_CHURN_COL,
    }
)
LEAKY_AFFINITY_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "repeat_purchase_ratio",
        "cart_to_purchase_rate",
        "category_revenue_share_pct",
        TARGET_AFFINITY_COL,
    }
)

# Canonical leak-free engineered feature columns for Customer Churn XGBClassifier
CHURN_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "loyalty_tier_code",
    "customer_segment_code",
    "country_bucket_code",
    "total_net_revenue_usd",
    "avg_order_value_usd",
    "total_events_count",
    "distinct_sessions_count",
    "avg_session_duration_sec",
    "page_view_count",
    "add_to_cart_count",
    "checkout_start_count",
    "purchase_event_count",
    "product_search_count",
    "cart_to_purchase_rate",
    "revenue_per_event_usd",
    "checkout_completion_ratio",
)

# Canonical leak-free engineered feature columns for Product Affinity XGBRegressor
AFFINITY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "category_code",
    "subcategory_bucket_code",
    "unit_price",
    "gross_margin_pct",
    "total_orders",
    "distinct_buyers_count",
    "total_units_sold",
    "total_net_revenue_usd",
    "total_events_count",
    "units_per_order_ratio",
    "revenue_per_buyer_usd",
    "estimated_margin_per_unit_usd",
)

# Unity Catalog schema descriptions for ML governance
ML_SCHEMA_DESCRIPTIONS: Final[dict[str, str]] = {
    "ml_features": (
        "Unity Catalog Feature Store schema hosting governed feature tables and label spines from Project 05 Gold views."
    ),
    "ml_models": (
        "Unity Catalog Model Registry schema hosting MLflow 3 LoggedModels for XGBoost @Champion and @Challenger versions."
    ),
    "ml_predictions": (
        "Governed batch prediction tables produced by distributed Spark scoring against @Champion XGBoost models."
    ),
    "ml_traces": (
        "Unity Catalog OpenTelemetry trace tables (_otel_spans, _otel_logs, _otel_annotations, _otel_metrics) for MLflow 3."
    ),
}


@dataclass(frozen=True)
class MLProjectConfig:
    """Immutable configuration for catalogs, schemas, feature/label tables, MLflow experiments, traces, and UC models."""

    catalog: str = "data_engineering_dev"
    upstream_gold_schema: str = "gold"
    feature_schema: str = "ml_features"
    model_schema: str = "ml_models"
    predictions_schema: str = "ml_predictions"
    trace_schema: str = "ml_traces"

    upstream_churn_view: str = "vw_ml_customer_churn_features"
    upstream_affinity_view: str = "vw_ml_product_affinity_features"

    churn_feature_table: str = "fe_customer_churn_features"
    affinity_feature_table: str = "fe_product_affinity_features"

    churn_label_table: str = "lbl_customer_churn"
    affinity_label_table: str = "lbl_product_affinity"

    churn_model_name: str = "xgb_customer_churn_classifier"
    affinity_model_name: str = "xgb_product_affinity_regressor"

    churn_predictions_table: str = "pred_customer_churn_xgboost"
    affinity_predictions_table: str = "pred_product_affinity_xgboost"

    churn_experiment_path: str = "/Shared/mlflow_xgboost_customer_churn_uc"
    affinity_experiment_path: str = "/Shared/mlflow_xgboost_product_affinity_uc"
    legacy_churn_experiment_path: str = "/Shared/mlflow_xgboost_customer_churn"
    legacy_affinity_experiment_path: str = "/Shared/mlflow_xgboost_product_affinity"

    churn_trace_table_prefix: str = "xgb_customer_churn"
    affinity_trace_table_prefix: str = "xgb_product_affinity"
    trace_sql_warehouse_id: str | None = None

    min_churn_roc_auc: float = 0.70
    min_affinity_r2: float = 0.60
    min_churn_roc_auc_delta: float = 0.005
    min_affinity_r2_delta: float = 0.005

    def table_fqn(self, schema_name: str, table_name: str) -> str:
        """Return dot-separated three-level Unity Catalog FQN (`catalog.schema.table`)."""
        return f"{self.catalog}.{schema_name}.{table_name}"

    @property
    def upstream_churn_view_fqn(self) -> str:
        """Return FQN of upstream Project 05 customer churn Gold view."""
        return self.table_fqn(self.upstream_gold_schema, self.upstream_churn_view)

    @property
    def upstream_affinity_view_fqn(self) -> str:
        """Return FQN of upstream Project 05 product affinity Gold view."""
        return self.table_fqn(self.upstream_gold_schema, self.upstream_affinity_view)

    @property
    def churn_feature_table_fqn(self) -> str:
        """Return FQN of Unity Catalog customer churn Feature Table (label-free)."""
        return self.table_fqn(self.feature_schema, self.churn_feature_table)

    @property
    def affinity_feature_table_fqn(self) -> str:
        """Return FQN of Unity Catalog product affinity Feature Table (label-free)."""
        return self.table_fqn(self.feature_schema, self.affinity_feature_table)

    @property
    def churn_label_table_fqn(self) -> str:
        """Return FQN of Unity Catalog customer churn label spine table."""
        return self.table_fqn(self.feature_schema, self.churn_label_table)

    @property
    def affinity_label_table_fqn(self) -> str:
        """Return FQN of Unity Catalog product affinity label spine table."""
        return self.table_fqn(self.feature_schema, self.affinity_label_table)

    @property
    def churn_model_fqn(self) -> str:
        """Return three-level Unity Catalog model name for Customer Churn XGBClassifier."""
        return self.table_fqn(self.model_schema, self.churn_model_name)

    @property
    def affinity_model_fqn(self) -> str:
        """Return three-level Unity Catalog model name for Product Affinity XGBRegressor."""
        return self.table_fqn(self.model_schema, self.affinity_model_name)

    @property
    def churn_predictions_table_fqn(self) -> str:
        """Return FQN of customer churn batch predictions Delta table."""
        return self.table_fqn(self.predictions_schema, self.churn_predictions_table)

    @property
    def affinity_predictions_table_fqn(self) -> str:
        """Return FQN of product affinity batch predictions Delta table."""
        return self.table_fqn(self.predictions_schema, self.affinity_predictions_table)

    @property
    def churn_trace_prefix_fqn(self) -> str:
        """Return 3-part Unity Catalog trace prefix (`catalog.trace_schema.churn_trace_table_prefix`)."""
        return self.table_fqn(self.trace_schema, self.churn_trace_table_prefix)

    @property
    def affinity_trace_prefix_fqn(self) -> str:
        """Return 3-part Unity Catalog trace prefix (`catalog.trace_schema.affinity_trace_table_prefix`)."""
        return self.table_fqn(self.trace_schema, self.affinity_trace_table_prefix)

    def model_alias_uri(self, model_fqn: str, alias: str = "Champion") -> str:
        """Return MLflow Unity Catalog alias URI (`models:/catalog.schema.model@Alias`)."""
        return f"models:/{model_fqn}@{alias}"


DEFAULT_ML_CONFIG: Final[MLProjectConfig] = MLProjectConfig()
