# ============================================================
# src/eda.py
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from .config import EDA_FIG_DIR


def plot_distribution(total_traffic, fname="traffic_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(total_traffic.values, bins=100,
                color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Total Internet Traffic (CDRs)")
    axes[0].set_ylabel("Number of Areas")
    axes[0].set_title("(a) Distribution of Total Traffic")
    axes[0].axvline(total_traffic.median(), color="red", linestyle="--",
                    label=f"Median={total_traffic.median():,.0f}")
    axes[0].axvline(total_traffic.mean(), color="orange", linestyle="--",
                    label=f"Mean={total_traffic.mean():,.0f}")
    axes[0].legend()

    axes[1].hist(np.log10(total_traffic.values + 1), bins=100,
                color="coral", edgecolor="white")
    axes[1].set_xlabel("log10(Total Traffic + 1)")
    axes[1].set_ylabel("Number of Areas")
    axes[1].set_title("(b) Log-Transformed Distribution")

    plt.tight_layout()
    plt.savefig(EDA_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_areas(area_series, top3, fname="top_areas.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, sid in zip(axes, top3):
        s = area_series[sid]
        s2 = s.loc[s.index <= s.index.min() + pd.Timedelta(days=14)]
        ax.plot(s2.index, s2.values, linewidth=0.8)
        ax.set_title(f"Square {sid} — first two weeks")
        ax.set_ylabel("Traffic")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(EDA_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_required_areas(area_series, required, fname="required_areas_first_two_weeks.png"):
    fig, axes = plt.subplots(len(required), 1, figsize=(14, 7), sharex=True)
    for ax, sid in zip(axes, required):
        s = area_series[sid]
        s2 = s.loc[s.index <= s.index.min() + pd.Timedelta(days=14)]
        ax.plot(s2.index, s2.values, linewidth=0.8)
        ax.set_title(f"Square {sid} — first two weeks")
        ax.set_ylabel("Traffic")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(EDA_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_acf_pacf(series, fname="acf_pacf.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(series, lags=300, ax=axes[0], alpha=0.05,
                title="ACF")
    plot_pacf(series, lags=100, ax=axes[1], alpha=0.05,
                title="PACF")
    plt.tight_layout()
    plt.savefig(EDA_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()


def plot_seasonality(series, period=144, fname="seasonality.png"):
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(res.observed, color="black", linewidth=0.6); axes[0].set_ylabel("Observed")
    axes[1].plot(res.trend,    color="steelblue", linewidth=1.0); axes[1].set_ylabel("Trend")
    axes[2].plot(res.seasonal, color="green",  linewidth=0.6); axes[2].set_ylabel("Seasonal")
    axes[3].plot(res.resid,    color="red",    linewidth=0.4); axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    for ax in axes: ax.grid(True, alpha=0.3)
    axes[0].set_title(f"STL Decomposition (period = {period})")
    plt.tight_layout()
    plt.savefig(EDA_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    return res