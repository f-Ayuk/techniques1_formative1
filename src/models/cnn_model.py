# ============================================================
# src/models/cnn_model.py
# ============================================================
import time, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from ..config import SEQ_LEN, RANDOM_STATE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(RANDOM_STATE)


class CNNForecaster:
    """1-D dilated causal CNN for one-step-ahead prediction."""

    def __init__(self, seq_len=SEQ_LEN, filters=32, kernel_size=3,
                n_blocks=2, dilation_base=2, dropout=0.2,
                learning_rate=1e-3, batch_size=64,
                epochs=30, patience=5):
        self.seq_len = seq_len
        self.filters = filters
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.train_time_ = 0.0

    def _build(self, input_shape):
        inp = layers.Input(shape=input_shape)
        x = inp
        for b in range(self.n_blocks):
            x = layers.Conv1D(self.filters, self.kernel_size,
                                padding="causal",
                              dilation_rate=self.dilation_base ** b,
                                activation="relu")(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        x = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(1)(x)
        m = models.Model(inp, out)
        m.compile(optimizer=optimizers.Adam(self.learning_rate),
                    loss="mse", metrics=["mae"])
        return m

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, verbose=0):
        self.model = self._build(X_tr.shape[1:])
        cbs = [callbacks.EarlyStopping(
            patience=self.patience, restore_best_weights=True,
            monitor="val_loss" if X_va is not None else "loss")]
        t0 = time.time()
        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va) if X_va is not None else None,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=cbs, verbose=verbose)
        self.train_time_ = time.time() - t0
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()