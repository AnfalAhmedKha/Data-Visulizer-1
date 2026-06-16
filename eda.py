"""
eda.py
======
Exploratory Data Analysis: summary stats, correlations,
and chart generation (matplotlib/seaborn, returns figures).
All chart functions return plt.Figure objects so Streamlit
can render them with st.pyplot() without side effects.
"""

import logging
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Consistent color palette
PALETTE = ["#667eea", "#764ba2", "#10b981", "#f59e0b", "#ef4444", "#3b82f6", "#8b5cf6"]
BG      = "#0f172a"
SURFACE = "#1e293b"
TEXT    = "#f1f5f9"
GRID    = "#334155"


def _dark_style(fig: plt.Figure, axes):
    """Apply consistent dark theme to a figure."""
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    return fig


# ── Summary Statistics ───────────────────────────────────────────────────────
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["skewness"]  = num.skew()
    desc["kurtosis"]  = num.kurtosis()
    desc["nulls"]     = df[num.columns].isnull().sum()
    desc["null_%"]    = (df[num.columns].isnull().mean() * 100).round(2)
    desc["unique"]    = df[num.columns].nunique()
    return desc.round(3)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    return num.corr().round(3)


def top_correlations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    corr = correlation_matrix(df)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "feature_1":   corr.columns[i],
                "feature_2":   corr.columns[j],
                "correlation": round(corr.iloc[i, j], 4),
                "abs_corr":    round(abs(corr.iloc[i, j]), 4),
            })
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).head(n)


# ── Charts ───────────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame, cols: List[str] = None,
                        bins: int = 30) -> plt.Figure:
    """Histogram + KDE grid for numeric columns."""
    num = df.select_dtypes(include=np.number)
    cols = cols or num.columns[:9].tolist()
    n = len(cols)
    if n == 0:
        return None
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2), dpi=110)
    axes_flat = list(axes.flat) if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=bins, color=PALETTE[i % len(PALETTE)],
                edgecolor="white", alpha=0.75, density=True)
        try:
            data.plot(kind="kde", ax=ax, color="white", lw=1.5)
        except Exception:
            pass
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Density")

    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    _dark_style(fig, axes_flat[:n])
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, max_cols: int = 15) -> plt.Figure:
    num = df.select_dtypes(include=np.number).iloc[:, :max_cols]
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle
    fig, ax = plt.subplots(figsize=(max(8, num.shape[1]), max(7, num.shape[1] - 1)), dpi=110)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.4,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", color=TEXT, pad=12)
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str,
                  hue: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    if hue and hue in df.columns:
        for i, (name, grp) in enumerate(df.groupby(hue)):
            ax.scatter(grp[x], grp[y], label=str(name),
                       color=PALETTE[i % len(PALETTE)], alpha=0.6, s=22, edgecolors="none")
        ax.legend(title=hue, labelcolor=TEXT, facecolor=SURFACE, edgecolor=GRID, fontsize=8)
    else:
        ax.scatter(df[x], df[y], color=PALETTE[0], alpha=0.55, s=22, edgecolors="none")
    # Trend line
    try:
        xn = pd.to_numeric(df[x], errors="coerce")
        yn = pd.to_numeric(df[y], errors="coerce")
        mask = xn.notna() & yn.notna()
        z = np.polyfit(xn[mask], yn[mask], 1)
        xr = np.linspace(xn[mask].min(), xn[mask].max(), 200)
        ax.plot(xr, np.poly1d(z)(xr), "--", color="#f59e0b", lw=1.5, label="trend")
    except Exception:
        pass
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x}  vs  {y}", fontsize=12, fontweight="bold")
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_box(df: pd.DataFrame, cols: List[str] = None) -> plt.Figure:
    num = df.select_dtypes(include=np.number)
    cols = cols or num.columns[:10].tolist()
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.2), 5), dpi=110)
    bp = ax.boxplot([df[c].dropna() for c in cols],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="#f59e0b", lw=2))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)] + "88")
        patch.set_edgecolor(PALETTE[i % len(PALETTE)])
    for elem in ["whiskers", "caps", "fliers"]:
        for line in bp[elem]:
            line.set_color(TEXT + "88")
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_title("Box Plots — Quartiles & Outliers", fontsize=12, fontweight="bold")
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str = None,
              top_n: int = 15, agg: str = "sum") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    if y and y in df.columns:
        data = df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(top_n)
    else:
        data = df[x].value_counts().head(top_n)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    bars = ax.bar(data.index.astype(str), data.values, color=colors, edgecolor="white", lw=0.4)
    ax.bar_label(bars, fmt="%.0f", padding=3, color=TEXT, fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y or "Count")
    ax.set_title(f"{agg.title()} of {y or 'Count'} by {x}", fontsize=12, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def plot_line(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    sd = df.sort_values(x)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110)
    yn = pd.to_numeric(sd[y], errors="coerce")
    ax.plot(range(len(sd)), yn, color=PALETTE[2], lw=2)
    ax.fill_between(range(len(sd)), yn, alpha=0.15, color=PALETTE[2])
    step = max(1, len(sd) // 8)
    ax.set_xticks(range(0, len(sd), step))
    ax.set_xticklabels(sd[x].iloc[::step].astype(str), rotation=40, ha="right", fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"Trend: {y} over {x}", fontsize=12, fontweight="bold")
    _dark_style(fig, ax)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG bytes (for download)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    return buf.read()
