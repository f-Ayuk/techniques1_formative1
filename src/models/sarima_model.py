# ============================================================
# src/models/sarima_model.py
# ============================================================
import time, warnings
from itertools import product
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..evaluation import mae

warnings.filterwarnings("ignore")


class SarimaForecaster:
    """SARIMA(p,d,q)(P,D,Q)_s wrapper with a lightweight grid search."""

    def __init__(self, order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 144), trend="n"):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model_fit = None
        self.train_time_ = 0.0

    def fit(self, y_train):
        t0 = time.time()
        self.model_fit = SARIMAX(
            y_train, order=self.order,
            seasonal_order=self.seasonal_order, trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        self.train_time_ = time.time() - t0
        return self

    def predict_one_step(self, history, steps=1):
        try:
            res = SARIMAX(
                history, order=self.order,
                seasonal_order=self.seasonal_order, trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            return float(res.forecast(steps=steps).iloc[-1])
        except Exception:
            return float(history.iloc[-1])

    @staticmethod
    def grid_search(y_train, y_val,
                    p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1),
                    P_range=(0, 1), D_range=(0, 1), Q_range=(0, 1),
                    s=144, max_evals=16):
        rows = []
        combos = list(product(p_range, d_range, q_range,
                                P_range, D_range, Q_range))
        combos = sorted(combos, key=lambda c: (sum(c), c))[:max_evals]
        for (p, d, q, P, D, Q) in combos:
            try:
                res = SARIMAX(
                    y_train, order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                val_pred = res.forecast(steps=len(y_val))
                rows.append({
                    "order": (p, d, q),
                    "seasonal": (P, D, Q, s),
                    "val_MAE": mae(y_val.values, np.asarray(val_pred)),
                })
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)