"""Shared helpers for federated learning.

Why a small neural net / logistic regression and not the Decision Tree as the *federated* model?
A decision tree has no numeric weights that can be averaged (FedAvg would be
meaningless on `tree_.feature` / `tree_.threshold`). A net or linear model has weights that
can be averaged. Default: MLP 10-16-1 (non-linear). Set FL_MODEL=logreg for the linear one.
Decision Tree / Naive Bayes are kept as centralised edge baselines for comparison.
"""
import os

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from src.features import N_FEATURES


FL_MODEL = os.getenv("FL_MODEL", "mlp")   # "mlp" (default) or "logreg"


def build_model(seed=42, kind=None):
    kind = kind or FL_MODEL
    X0, y0 = np.zeros((2, N_FEATURES)), np.array([0, 1])
    if kind == "logreg":
        m = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="constant", eta0=0.1, random_state=seed)
        m.partial_fit(X0, y0, classes=np.array([0, 1]))
        m.coef_ = np.zeros((1, N_FEATURES)); m.intercept_ = np.zeros(1)
    else:  # small non-linear net: 10 -> 16 -> 1, ~200 weights, still tiny for an edge device
        m = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                          learning_rate_init=0.01, batch_size=256, random_state=seed)
        m.partial_fit(X0, y0, classes=np.array([0, 1]))   # same seed => identical initial weights
    return m


def get_params(m):
    if hasattr(m, "coefs_"):
        return [a.copy() for a in m.coefs_] + [a.copy() for a in m.intercepts_]
    return [m.coef_.copy(), m.intercept_.copy()]


def set_params(m, params):
    if hasattr(m, "coefs_"):
        k = len(m.coefs_)
        for i in range(k):
            m.coefs_[i][...] = np.asarray(params[i], dtype=float)
            m.intercepts_[i][...] = np.asarray(params[k + i], dtype=float)
        if hasattr(m, "_optimizer"):
            del m._optimizer          # fresh Adam state each round (only weights are shared)
    else:
        m.coef_ = np.asarray(params[0], dtype=float).reshape(1, -1)
        m.intercept_ = np.asarray(params[1], dtype=float).reshape(-1)
    return m


def local_train(m, X, y, epochs=3, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        m.partial_fit(X[idx], y[idx])
    return m


def partition(X, y, num_clients, cid, seed=42, non_iid=False):
    """Split training data among clients (IID, or label-skewed when non_iid)."""
    rng = np.random.default_rng(seed)
    if non_iid:
        order = np.argsort(y + rng.normal(0, 0.7, len(y)))
    else:
        order = rng.permutation(len(X))
    idx = np.array_split(order, num_clients)[cid]
    return X[idx], y[idx]


def fedavg(param_list, sizes):
    total = float(sum(sizes))
    return [sum(p[i] * (n / total) for p, n in zip(param_list, sizes)) for i in range(len(param_list[0]))]
