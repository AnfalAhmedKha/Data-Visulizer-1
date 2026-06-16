"""
model.py
========
Train, evaluate, and persist ML models.
Supports classification and regression.
Returns structured results dicts — no side effects.
"""

import logging
import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"(^id$|.*_id$|^index$|^unnamed)", re.IGNORECASE)


# ── Model registry ──────────────────────────────────────────────────────────
def get_classifiers() -> dict:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    return {
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors":   KNeighborsClassifier(n_neighbors=5),
        "SVM":                   SVC(probability=True, random_state=42),
    }


def get_regressors() -> dict:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    return {
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression":  LinearRegression(),
        "Ridge":              Ridge(alpha=1.0),
        "Lasso":              Lasso(alpha=0.1, max_iter=2000),
    }


# ── Preparation ─────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame,
                      target: str,
                      exclude: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split df into X, y.
    Auto-drops non-numeric columns that can't be used as features, and
    auto-excludes obvious identifier columns (e.g. 'order_id', 'index')
    that carry no predictive signal and only add noise / leakage risk.
    """
    exclude = set(exclude or []) | {target}

    id_like = [c for c in df.columns
               if c not in exclude
               and _ID_PATTERN.match(c)
               and df[c].nunique() == len(df)]
    if id_like:
        logger.info(f"Auto-excluding identifier-like columns: {id_like}")
        exclude |= set(id_like)

    X = df.drop(columns=[c for c in df.columns if c in exclude])
    # Keep only numeric (post-encoding)
    X = X.select_dtypes(include=np.number)
    # Fill any residual nulls
    X = X.fillna(X.median())
    y = df[target]
    logger.info(f"Features: {X.shape[1]} cols  |  Target: '{target}'  ({y.nunique()} unique values)")
    return X, y


def detect_task(y: pd.Series, threshold: int = 15) -> str:
    """Auto-detect classification vs regression."""
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    if y.nunique() <= threshold:
        return "classification"
    return "regression"


# ── Training ────────────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y: pd.Series,
           task: str = "auto",
           model_name: str = "Random Forest",
           test_size: float = 0.2,
           cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train one model and return full evaluation results.
    task: 'auto' | 'classification' | 'regression'
    """
    from sklearn.model_selection import train_test_split, cross_val_score

    if task == "auto":
        task = detect_task(y)

    models = get_classifiers() if task == "classification" else get_regressors()
    if model_name not in models:
        model_name = list(models.keys())[0]
    model = models[model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task == "classification" else None)

    logger.info(f"Training '{model_name}' ({task})  |  train={len(X_train):,}  test={len(X_test):,}")
    model.fit(X_train, y_train)

    # Cross-validation
    cv_metric = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=cv_metric, n_jobs=-1)

    # Evaluation
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, task)

    # Feature importance
    importance = {}
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(X.columns, model.feature_importances_.round(4)))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        importance = dict(zip(X.columns, np.abs(coef).round(4)))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    result = {
        "model":          model,
        "model_name":     model_name,
        "task":           task,
        "metrics":        metrics,
        "cv_scores":      cv_scores.tolist(),
        "cv_mean":        round(float(cv_scores.mean()), 4),
        "cv_std":         round(float(cv_scores.std()), 4),
        "feature_importance": importance,
        "X_test":         X_test,
        "y_test":         y_test,
        "y_pred":         y_pred,
        "feature_names":  X.columns.tolist(),
        "trained_at":     datetime.now().isoformat(),
    }
    logger.info(f"Training complete: {metrics}")
    return result


def train_compare_all(X: pd.DataFrame, y: pd.Series,
                       task: str = "auto",
                       test_size: float = 0.2) -> pd.DataFrame:
    """
    Train ALL available models and return a comparison DataFrame.
    Great for leaderboard display.
    """
    from sklearn.model_selection import train_test_split
    if task == "auto":
        task = detect_task(y)
    models = get_classifiers() if task == "classification" else get_regressors()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42,
                                                stratify=y if task == "classification" else None)
    rows = []
    for name, m in models.items():
        try:
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            met = evaluate(y_te, y_pred, task)
            met["model"] = name
            rows.append(met)
            logger.info(f"  {name}: {met}")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")
    df = pd.DataFrame(rows).set_index("model")
    return df


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, task: str) -> dict:
    if task == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "r2":   round(r2_score(y_true, y_pred), 4),
            "rmse": round(rmse, 4),
            "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        }


def confusion_matrix_fig(y_true, y_pred) -> "plt.Figure":
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", color="#f1f5f9")
    ax.set_ylabel("Actual", color="#f1f5f9")
    ax.set_title("Confusion Matrix", color="#f1f5f9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#f1f5f9")
    plt.tight_layout()
    return fig


def feature_importance_fig(importance: dict, top_n: int = 15) -> "plt.Figure":
    import matplotlib.pyplot as plt
    items = list(importance.items())[:top_n]
    features, scores = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.4)), dpi=110)
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    colors = ["#667eea"] * len(features)
    ax.barh(range(len(features)), scores, color=colors, edgecolor="white", lw=0.3)
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=9, color="#f1f5f9")
    ax.set_xlabel("Importance", color="#f1f5f9")
    ax.set_title("Feature Importance (Top Features)", color="#f1f5f9", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#f1f5f9")
    for sp in ax.spines.values(): sp.set_color("#334155")
    ax.grid(axis="x", color="#334155", lw=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── Persistence ─────────────────────────────────────────────────────────────
def save_model(model, path: str, metadata: dict = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model, "metadata": metadata or {}, "saved_at": datetime.now().isoformat()}
    joblib.dump(payload, path)
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"Model saved: {path}  ({size_kb:.1f} KB)")
    return path


def load_model(path: str):
    payload = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return payload["model"], payload.get("metadata", {})
