"""Deterministic dataset splitting (out-of-time & stratified), leakage detection, and feature importance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBModel

from src.python.modules.utils.config import COL_SNAPSHOT_DATE


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable container holding Train (60%), Validation (20%), Combined Train+Val (80%), and Holdout Test (20%)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_trainval: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_trainval: pd.Series
    y_test: pd.Series
    split_strategy: str = "random"


def compute_scale_pos_weight(y_series: pd.Series) -> float:
    """Compute positive class weight (`negative_count / positive_count`) for imbalanced binary classification."""
    pos_count = int((y_series == 1).sum())
    neg_count = int((y_series == 0).sum())
    if pos_count <= 0 or neg_count <= 0:
        return 1.0
    return round(float(neg_count) / float(pos_count), 4)


def _compute_temporal_date_partitions(unique_dates: list[Any]) -> tuple[set[Any], set[Any], set[Any]]:
    """Partition sorted unique snapshot dates into non-overlapping chronological 60/20/20 train/val/test sets."""
    n_dates = len(unique_dates)
    train_end = max(1, int(round(n_dates * 0.60)))
    val_end = max(train_end + 1, int(round(n_dates * 0.80)))
    # Guarantee at least one future period remains reserved for the untouched out-of-time holdout split.
    if val_end >= n_dates:
        val_end = n_dates - 1
        train_end = max(1, val_end - 1)
    return set(unique_dates[:train_end]), set(unique_dates[train_end:val_end]), set(unique_dates[val_end:])


def _try_temporal_split(
    pdf: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    time_col: str,
    target_dtype: type[int] | type[float],
    require_both_classes: bool,
) -> DatasetSplits | None:
    """Attempt a chronological 60/20/20 out-of-time split by `time_col`, returning None if insufficient periods."""
    if time_col not in pdf.columns:
        return None

    ordered = pdf.sort_values(by=time_col, kind="mergesort").reset_index(drop=True)
    unique_dates = ordered[time_col].dropna().unique().tolist()
    if len(unique_dates) < 3:
        return None

    train_dates, val_dates, test_dates = _compute_temporal_date_partitions(unique_dates)
    train_df = ordered[ordered[time_col].isin(train_dates)]
    val_df = ordered[ordered[time_col].isin(val_dates)]
    test_df = ordered[ordered[time_col].isin(test_dates)]
    trainval_df = ordered[ordered[time_col].isin(train_dates | val_dates)]

    if train_df.empty or val_df.empty or test_df.empty:
        return None

    y_train = train_df[target_col].astype(target_dtype).reset_index(drop=True)
    y_val = val_df[target_col].astype(target_dtype).reset_index(drop=True)
    y_test = test_df[target_col].astype(target_dtype).reset_index(drop=True)
    y_trainval = trainval_df[target_col].astype(target_dtype).reset_index(drop=True)

    # Fall back to stratified splitting if any temporal slice lacks a class (would invalidate AUC/log-loss).
    if require_both_classes and any(series.nunique() < 2 for series in (y_train, y_val, y_test)):
        return None

    feat_list = list(feature_cols)
    return DatasetSplits(
        X_train=train_df[feat_list].astype(float).reset_index(drop=True),
        X_val=val_df[feat_list].astype(float).reset_index(drop=True),
        X_trainval=trainval_df[feat_list].astype(float).reset_index(drop=True),
        X_test=test_df[feat_list].astype(float).reset_index(drop=True),
        y_train=y_train,
        y_val=y_val,
        y_trainval=y_trainval,
        y_test=y_test,
        split_strategy="temporal_out_of_time",
    )


def split_classification_dataset(
    pdf: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    test_size: float = 0.20,
    val_relative_size: float = 0.25,
    random_state: int = 42,
    time_col: str | None = COL_SNAPSHOT_DATE,
) -> DatasetSplits:
    """Split binary classification data chronologically (when `>=3` snapshot periods exist) or stratified 60/20/20."""
    if time_col is not None:
        temporal = _try_temporal_split(
            pdf=pdf,
            feature_cols=feature_cols,
            target_col=target_col,
            time_col=time_col,
            target_dtype=int,
            require_both_classes=True,
        )
        if temporal is not None:
            return temporal

    X = pdf[list(feature_cols)].astype(float).reset_index(drop=True)
    y = pdf[target_col].astype(int).reset_index(drop=True)

    # Stratify only when each class has enough observations to populate outer and inner splits cleanly.
    outer_counts = y.value_counts()
    stratify_outer = y if (len(outer_counts) >= 2 and int(outer_counts.min()) >= 4) else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_outer
    )

    inner_counts = y_trainval.value_counts()
    stratify_inner = y_trainval if (len(inner_counts) >= 2 and int(inner_counts.min()) >= 2) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative_size, random_state=random_state, stratify=stratify_inner
    )
    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_trainval=X_trainval.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_trainval=y_trainval.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        split_strategy="stratified_random",
    )


def split_regression_dataset(
    pdf: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    test_size: float = 0.20,
    val_relative_size: float = 0.25,
    random_state: int = 42,
    time_col: str | None = COL_SNAPSHOT_DATE,
) -> DatasetSplits:
    """Split continuous regression data chronologically (when `>=3` snapshot periods exist) or randomly 60/20/20."""
    if time_col is not None:
        temporal = _try_temporal_split(
            pdf=pdf,
            feature_cols=feature_cols,
            target_col=target_col,
            time_col=time_col,
            target_dtype=float,
            require_both_classes=False,
        )
        if temporal is not None:
            return temporal

    X = pdf[list(feature_cols)].astype(float).reset_index(drop=True)
    y = pdf[target_col].astype(float).reset_index(drop=True)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative_size, random_state=random_state
    )
    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_trainval=X_trainval.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_trainval=y_trainval.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        split_strategy="random",
    )


def detect_single_feature_leakage(
    splits: DatasetSplits,
    feature_cols: tuple[str, ...],
    task_type: Literal["classification", "regression"],
    threshold: float = 0.98,
    random_state: int = 42,
) -> dict[str, float]:
    """Detect suspected target leakage by fitting a depth-1 decision stump on each feature individually.

    Returns a dictionary of `{feature_name: validation_score}` for any feature whose standalone
    depth-1 ROC-AUC (classification) or R2 (regression) exceeds `threshold`.
    """
    flagged: dict[str, float] = {}
    for col in feature_cols:
        x_tr = splits.X_train[[col]]
        x_va = splits.X_val[[col]]
        if task_type == "classification":
            if splits.y_val.nunique() < 2:
                continue
            stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
            stump.fit(x_tr, splits.y_train)
            score = float(roc_auc_score(splits.y_val, stump.predict_proba(x_va)[:, 1]))
        else:
            if len(splits.y_val) < 2:
                continue
            reg_stump = DecisionTreeRegressor(max_depth=1, random_state=random_state)
            reg_stump.fit(x_tr, splits.y_train)
            score = float(r2_score(splits.y_val, reg_stump.predict(x_va)))
        if score >= threshold:
            flagged[col] = round(score, 6)
    return flagged


def _scalar_score(raw_score: float | list[float]) -> float:
    """Coerce an XGBoost booster score entry (`float` or `list[float]`) to a scalar float."""
    if isinstance(raw_score, list):
        return float(raw_score[0]) if raw_score else 0.0
    return float(raw_score)


def extract_feature_importance_df(model: XGBModel, feature_cols: tuple[str, ...]) -> pd.DataFrame:
    """Extract sorted feature importance (`gain` and `weight`) from a fitted XGBoost estimator."""
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")

    rows: list[dict[str, float | str]] = []
    for idx, col_name in enumerate(feature_cols):
        f_key = f"f{idx}"
        # Support both named-column boosters (DataFrame input) and positional f0..fN keys (ndarray input).
        gain_val = _scalar_score(gain_scores.get(col_name, gain_scores.get(f_key, 0.0)))
        weight_val = _scalar_score(weight_scores.get(col_name, weight_scores.get(f_key, 0.0)))
        rows.append(
            {
                "feature": col_name,
                "importance_gain": round(gain_val, 6),
                "importance_weight": round(weight_val, 4),
            }
        )
    return pd.DataFrame(rows).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
