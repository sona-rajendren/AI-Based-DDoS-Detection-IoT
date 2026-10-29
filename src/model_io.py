"""Save / load model bundles (model + metadata) with joblib."""
import time
import joblib
from src.features import FEATURE_NAMES


def save_bundle(model, path, name, metrics=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model, "name": name, "features": FEATURE_NAMES,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics or {}, "extra": extra or {},
    }
    joblib.dump(bundle, path)
    return bundle


def load_bundle(path):
    b = joblib.load(path)
    if isinstance(b, dict) and "model" in b:
        return b
    return {"model": b, "name": "legacy", "features": FEATURE_NAMES, "metrics": {}, "extra": {}}


def attack_probability(model, X):
    """P(attack) for each row, works for any sklearn classifier."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        classes = list(model.classes_)
        return p[:, classes.index(1)] if 1 in classes else p[:, -1]
    return model.predict(X).astype(float)
