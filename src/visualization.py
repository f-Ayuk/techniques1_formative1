# ============================================================
# src/visualization.py
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .config import FCST_FIG_DIR, ERR_FIG_DIR


def plot_forecast(te_series, y_pred, model_name, area_id, fname=None):
    fname = fname or f"{model_name.lower()}_area_{area_id}.png"
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(te_series.index, te_series.values,
            label="Actual", linewidth=1.2, color="black")
    ax.plot(te_series.index, y_pred,
            label=f"{model_name} prediction", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Square {area_id} — {model_name} | Dec 16–22")
    ax.set_ylabel("Internet traffic")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(FCST_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_errors(te_series, y_pred_dict, fname="worst_prediction.png"):
    """Overlay residuals of the three models on the test window."""
    fig, ax = plt.subplots(figsize=(14, 4))
    for name, yp in y_pred_dict.items():
        err = np.asarray(te_series.values) - np.asarray(yp)
        ax.plot(te_series.index, err, label=f"{name} residual",
                linewidth=0.9, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Residuals on test week")
    ax.set_ylabel("Actual − Predicted")
    ax.set_xlabel("Date")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(ERR_FIG_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)