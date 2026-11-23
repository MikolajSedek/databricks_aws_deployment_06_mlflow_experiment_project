"""Deterministic synthetic generators and StructType contracts for Project 05 Gold ML views.

Produces PySpark DataFrames adhering strictly to the upstream Kimball Gold view contracts
defined in Project 05 (`databricks_aws_deployment_05_data_vault_and_dwh`):
1. `gold.vw_ml_customer_churn_features` (grain: 1 row per current user `sk_user`).
2. `gold.vw_ml_product_affinity_features` (grain: 1 row per current product `sk_product`).

Synthetic feature distributions intentionally include overlapping behavioral signal and
controlled noise so no single non-leaky feature trivially separates the target, enabling
realistic unit tests and CI single-feature leakage guards.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Final, TypeAlias

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType, StructField, StructType

# Explicit row tuple aliases keep synthetic row builders readable and strictly typed.
_ChurnViewRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    int,
    float,
    float,
    int,
    int,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    int,
    date,
]
_AffinityViewRow: TypeAlias = tuple[
    str, str, str, str, str, float, float, int, int, int, float, int, float, float, float, float, date
]

GOLD_CHURN_VIEW_SCHEMA: Final[StructType] = StructType(
    [
        StructField("sk_user", StringType(), nullable=False),
        StructField("user_id", StringType(), nullable=False),
        StructField("country_code", StringType(), nullable=True),
        StructField("loyalty_tier", StringType(), nullable=True),
        StructField("calculated_customer_segment", StringType(), nullable=True),
        StructField("total_orders", IntegerType(), nullable=False),
        StructField("total_net_revenue_usd", DoubleType(), nullable=False),
        StructField("avg_order_value_usd", DoubleType(), nullable=False),
        StructField("total_events_count", IntegerType(), nullable=False),
        StructField("distinct_sessions_count", IntegerType(), nullable=False),
        StructField("avg_session_duration_sec", DoubleType(), nullable=False),
        StructField("page_view_count", IntegerType(), nullable=False),
        StructField("add_to_cart_count", IntegerType(), nullable=False),
        StructField("checkout_start_count", IntegerType(), nullable=False),
        StructField("purchase_event_count", IntegerType(), nullable=False),
        StructField("product_search_count", IntegerType(), nullable=False),
        StructField("days_since_last_order", IntegerType(), nullable=False),
        StructField("days_since_last_event", IntegerType(), nullable=False),
        StructField("cart_to_purchase_rate", DoubleType(), nullable=False),
        StructField("churn_risk_score", DoubleType(), nullable=False),
        StructField("is_churn_candidate", IntegerType(), nullable=False),
        StructField("snapshot_date", DateType(), nullable=False),
    ]
)

GOLD_AFFINITY_VIEW_SCHEMA: Final[StructType] = StructType(
    [
        StructField("sk_product", StringType(), nullable=False),
        StructField("product_id", StringType(), nullable=False),
        StructField("product_name", StringType(), nullable=False),
        StructField("category", StringType(), nullable=True),
        StructField("subcategory", StringType(), nullable=True),
        StructField("unit_price", DoubleType(), nullable=False),
        StructField("gross_margin_pct", DoubleType(), nullable=False),
        StructField("total_orders", IntegerType(), nullable=False),
        StructField("distinct_buyers_count", IntegerType(), nullable=False),
        StructField("total_units_sold", IntegerType(), nullable=False),
        StructField("total_net_revenue_usd", DoubleType(), nullable=False),
        StructField("total_events_count", IntegerType(), nullable=False),
        StructField("repeat_purchase_ratio", DoubleType(), nullable=False),
        StructField("category_revenue_share_pct", DoubleType(), nullable=False),
        StructField("cart_to_purchase_rate", DoubleType(), nullable=False),
        StructField("affinity_index_score", DoubleType(), nullable=False),
        StructField("snapshot_date", DateType(), nullable=False),
    ]
)


def _sample_churn_label_and_recency(
    rng: random.Random,
    tier_idx: int,
    seg_idx: int,
    sessions: int,
    avg_sec: float,
    cart_rate: float,
    revenue: float,
) -> tuple[int, int, int, float]:
    """Sample binary churn target, upstream recency days, and risk score from overlapping weak behavioral signals."""
    # Blend multiple weak predictors + Gaussian noise so no single feature causes trivial target leakage.
    logit = (
        1.6
        - 0.45 * tier_idx
        - 0.50 * seg_idx
        - 0.08 * sessions
        - 0.004 * avg_sec
        - 1.8 * cart_rate
        - 0.0012 * revenue
        + rng.gauss(0.0, 0.75)
    )
    prob = 1.0 / (1.0 + math.exp(-logit))
    is_churn = 1 if rng.random() < prob else 0
    days_since_order = rng.randint(35, 110) if is_churn == 1 else rng.randint(1, 35)
    days_since_event = rng.randint(15, 70) if is_churn == 1 else rng.randint(0, 20)
    risk_score = round(min(100.0, max(0.0, prob * 100.0)), 2)
    return is_churn, days_since_order, days_since_event, risk_score


def _build_churn_user_tuple(
    idx: int,
    rng: random.Random,
    tiers: tuple[str, ...],
    segments: tuple[str, ...],
    countries: tuple[str, ...],
    snapshot_dt: date,
) -> _ChurnViewRow:
    """Generate a single deterministic, realistically noisy synthetic row for `vw_ml_customer_churn_features`."""
    tier_idx = rng.randint(0, len(tiers) - 1)
    seg_idx = rng.randint(0, len(segments) - 1)
    country = countries[rng.randint(0, len(countries) - 1)]

    # Funnel counts preserve realistic clickstream ordering: views >= carts >= checkouts >= purchases.
    sessions = rng.randint(1, 20)
    views = max(1, int(sessions * rng.uniform(2.0, 5.5)))
    carts = max(1, int(views * rng.uniform(0.10, 0.35)))
    checkouts = max(0, min(carts, int(carts * rng.uniform(0.30, 0.85))))
    buys = max(0, min(checkouts, int(checkouts * rng.uniform(0.35, 0.95))))
    searches = rng.randint(1, 24)
    avg_sec = round(rng.uniform(40.0, 420.0), 2)
    aov = round(rng.uniform(25.0, 220.0), 2)
    orders = max(1, buys + rng.randint(0, 4))
    rev = round(orders * aov, 2)
    events = views + carts + checkouts + buys + searches
    cart_rate = round(buys / max(carts, 1), 4)

    is_churn, d_ord, d_evt, risk = _sample_churn_label_and_recency(
        rng, tier_idx, seg_idx, sessions, avg_sec, cart_rate, rev
    )
    return (
        f"hk_user_{idx:05d}",
        f"USR-{idx:05d}",
        country,
        tiers[tier_idx],
        segments[seg_idx],
        int(orders),
        float(rev),
        float(aov),
        int(events),
        int(sessions),
        float(avg_sec),
        int(views),
        int(carts),
        int(checkouts),
        int(buys),
        int(searches),
        int(d_ord),
        int(d_evt),
        float(cart_rate),
        float(risk),
        int(is_churn),
        snapshot_dt,
    )


def build_sample_gold_churn_view_df(
    spark: SparkSession,
    num_users: int = 120,
    num_snapshots: int = 5,
    seed: int = 42,
) -> DataFrame:
    """Create a deterministic multi-snapshot PySpark DataFrame matching `gold.vw_ml_customer_churn_features`."""
    rng = random.Random(seed)  # nosec B311
    tiers = ("BRONZE", "SILVER", "GOLD", "PLATINUM")
    segments = ("AT_RISK", "REGULAR", "HIGH_VALUE", "VIP")
    countries = ("US", "GB", "DE", "PL")
    base_date = date(2026, 1, 1)
    snapshot_count = max(1, num_snapshots)
    rows = [
        _build_churn_user_tuple(
            idx=idx,
            rng=rng,
            tiers=tiers,
            segments=segments,
            countries=countries,
            snapshot_dt=base_date + timedelta(days=7 * ((idx - 1) % snapshot_count)),
        )
        for idx in range(1, num_users + 1)
    ]
    return spark.createDataFrame(rows, schema=GOLD_CHURN_VIEW_SCHEMA)


def _build_affinity_product_tuple(
    idx: int,
    rng: random.Random,
    categories: tuple[str, ...],
    subcategories: tuple[str, ...],
    snapshot_dt: date,
) -> _AffinityViewRow:
    """Generate a single deterministic, realistically noisy synthetic row for `vw_ml_product_affinity_features`."""
    cat_idx = rng.randint(0, len(categories) - 1)
    sub_idx = rng.randint(0, len(subcategories) - 1)
    unit_price = round(rng.uniform(15.0, 320.0), 2)
    gross_margin_pct = round(rng.uniform(18.0, 68.0), 2)
    distinct_buyers = rng.randint(5, 80)
    repeat_ratio = round(rng.uniform(0.05, 0.90), 4)
    total_orders = max(distinct_buyers, int(round(distinct_buyers * (1.0 + repeat_ratio))))
    total_units = int(total_orders * rng.uniform(1.0, 2.4))
    total_rev = round(total_units * unit_price * 0.92, 2)
    total_events = int(total_orders * rng.uniform(4.0, 10.0) + rng.randint(5, 60))
    category_share_pct = round(rng.uniform(0.5, 18.0), 4)
    cart_to_purchase = round(rng.uniform(0.12, 0.85), 4)

    # Combine multiple non-leaky commercial drivers + noise so no single feature produces R2 >= 0.98.
    units_per_order = total_units / max(total_orders, 1)
    margin_per_unit = unit_price * (gross_margin_pct / 100.0)
    raw_score = (
        18.0
        + 0.35 * distinct_buyers
        + 9.5 * units_per_order
        + 0.14 * margin_per_unit
        + 0.015 * total_events
        + 2.5 * cat_idx
        + rng.gauss(0.0, 5.5)
    )
    affinity_score = round(max(0.0, min(100.0, raw_score)), 4)
    return (
        f"hk_product_{idx:05d}",
        f"PRD-{idx:05d}",
        f"Product {idx:05d}",
        categories[cat_idx],
        subcategories[sub_idx],
        float(unit_price),
        float(gross_margin_pct),
        int(total_orders),
        int(distinct_buyers),
        int(total_units),
        float(total_rev),
        int(total_events),
        float(repeat_ratio),
        float(category_share_pct),
        float(cart_to_purchase),
        float(affinity_score),
        snapshot_dt,
    )


def build_sample_gold_affinity_view_df(
    spark: SparkSession,
    num_products: int = 80,
    num_snapshots: int = 5,
    seed: int = 42,
) -> DataFrame:
    """Create a deterministic multi-snapshot PySpark DataFrame matching `gold.vw_ml_product_affinity_features`."""
    rng = random.Random(seed)  # nosec B311
    categories = ("ELECTRONICS", "APPAREL", "HOME_GARDEN", "SPORTS")
    subcategories = ("ACCESSORIES", "FOOTWEAR", "SMART_HOME", "OUTDOOR")
    base_date = date(2026, 1, 1)
    snapshot_count = max(1, num_snapshots)
    rows = [
        _build_affinity_product_tuple(
            idx=idx,
            rng=rng,
            categories=categories,
            subcategories=subcategories,
            snapshot_dt=base_date + timedelta(days=7 * ((idx - 1) % snapshot_count)),
        )
        for idx in range(1, num_products + 1)
    ]
    return spark.createDataFrame(rows, schema=GOLD_AFFINITY_VIEW_SCHEMA)
