"""Feature engineering and Unity Catalog Feature Store publishing module."""

from src.python.modules.features.feature_engineering import (
    engineer_customer_churn_features,
    engineer_product_affinity_features,
    extract_customer_churn_labels,
    extract_product_affinity_labels,
    sync_label_table,
    sync_uc_feature_table,
)
from src.python.modules.features.synthetic_gold_views import (
    GOLD_AFFINITY_VIEW_SCHEMA,
    GOLD_CHURN_VIEW_SCHEMA,
    build_sample_gold_affinity_view_df,
    build_sample_gold_churn_view_df,
)

__all__ = [
    "GOLD_AFFINITY_VIEW_SCHEMA",
    "GOLD_CHURN_VIEW_SCHEMA",
    "build_sample_gold_affinity_view_df",
    "build_sample_gold_churn_view_df",
    "engineer_customer_churn_features",
    "engineer_product_affinity_features",
    "extract_customer_churn_labels",
    "extract_product_affinity_labels",
    "sync_label_table",
    "sync_uc_feature_table",
]
