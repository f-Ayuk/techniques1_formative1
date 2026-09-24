# ============================================================
# src/evaluation.py
# ============================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y, yhat):
    return float(mean_absolute_error(y, yhat))


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def mape(y, yhat, eps=1e-6):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = np.abs(y) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def evaluate(y, yhat):
    return {"MAE": mae(y, yhat), "MAPE": mape(y, yhat), "RMSE": rmse(y, yhat)}


# ---- Walk-forward evaluation loops -------------------------

def walk_forward_sarima(model, history, test):
    """Refit SARIMA each step for one-step-ahead predictions."""
    import time
    preds = []
    hist = history.copy()
    t0 = time.time()
    for t in range(len(test)):
        p = model.predict_one_step(hist, steps=1)
        preds.append(p)
        hist = pd.concat([hist, pd.Series([test.iloc[t]],
                                            index=[test.index[t]])])
    return np.asarray(preds), time.time() - t0


def walk_forward_dl(model, history_values, test_values, seq_len):
    """One-step-ahead walk-forward for LSTM/CNN."""
    import time
    preds = []
    buf = list(history_values[-seq_len:])
    t0 = time.time()
    for i in range(len(test_values)):
        x = np.asarray(buf[-seq_len:], dtype="float32")[np.newaxis, :, np.newaxis]
        preds.append(float(model.predict(x)[0]))
        buf.append(test_values[i])
    return np.asarray(preds), time.time() - t0