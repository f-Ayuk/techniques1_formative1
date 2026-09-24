# ============================================================
# src/preprocessing.py
# ============================================================
import numpy as np
import pandas as pd
from .config import SEQ_LEN


def train_val_test_split(series, train_end, val_start, val_end,
                        test_start, test_end):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    train = s[s.index <= train_end]
    val   = s[(s.index >= val_start) & (s.index <= val_end)]
    test  = s[(s.index >= test_start) & (s.index <= test_end)]
    return train, val, test


def normalise(train, val, test):
    """Z-score using train statistics only."""
    mu, sd = float(train.mean()), float(train.std())
    sd = sd if sd > 0 else 1.0
    return ((train - mu) / sd, (val - mu) / sd, (test - mu) / sd), (mu, sd)


def make_windows(series, seq_len=SEQ_LEN, horizon=1):
    """Return (X, y) for one-step-ahead windows.
    X: (N, seq_len, 1)   y: (N,)
    """
    v = series.values.astype("float32")
    X, y = [], []
    for i in range(len(v) - seq_len - horizon + 1):
        X.append(v[i:i + seq_len])
        y.append(v[i + seq_len + horizon - 1])
    X = np.asarray(X, dtype="float32")[..., np.newaxis]
    y = np.asarray(y, dtype="float32")
    return X, y