# ============================================================
# src/experiment.py
# ============================================================
import pandas as pd


class ExperimentLog:
    """Stores one row per training experiment for later analysis."""

    def __init__(self):
        self.rows = []

    def add(self, model, area, params, metrics,
            train_time=None, exec_time=None, notes="", stage=""):
        self.rows.append({
            "model": model, "area": area, "params": str(params),
            "MAE": metrics["MAE"], "MAPE": metrics["MAPE"],
            "RMSE": metrics["RMSE"],
            "train_time_s": train_time, "exec_time_s": exec_time,
            "stage": stage, "notes": notes,
        })
        print(f"[{model:>6s}] area={area:<5} | "
                f"MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} "
                f"MAPE={metrics['MAPE']:.2f}%"
                + (f" | train={train_time:.1f}s" if train_time is not None else "")
                + (f" | exec={exec_time:.1f}s" if exec_time is not None else "")
                + (f" | {notes}" if notes else ""))

    def to_df(self):
        return pd.DataFrame(self.rows)