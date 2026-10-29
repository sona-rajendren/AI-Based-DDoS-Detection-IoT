import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from src.model_io import attack_probability


def evaluate_model(model, X, y, types=None):
    p = attack_probability(model, X)
    pred = (p >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    tn, fp = out["confusion_matrix"][0]
    out["false_positive_rate"] = float(fp / (tn + fp)) if (tn + fp) else 0.0
    if types is not None:
        out["per_class_detection"] = {
            t: float((pred[types == t] == (0 if t == "Normal" else 1)).mean())
            for t in sorted(set(types))}
    return out, p
