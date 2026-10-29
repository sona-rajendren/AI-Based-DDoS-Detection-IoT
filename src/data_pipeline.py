"""Dataset loading, filtering, feature engineering and train/test split."""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATASET_PATH, SYNTHETIC_DATASET_PATH
from src.features import FEATURE_NAMES, extract_from_dataframe, required_columns
from src.synthetic import generate_dataset

TARGET_ATTACKS = ["Normal", "DDoS_UDP", "DDoS_ICMP", "DDoS_HTTP", "DDoS_TCP"]
SELECTED_FEATURES = FEATURE_NAMES   # kept for backward compatibility with Review 1 naming


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    types_train: np.ndarray
    types_test: np.ndarray
    features: list
    source: str


def _read_raw(filepath):
    from pathlib import Path
    filepath = Path(filepath)
    if filepath.exists():
        header = pd.read_csv(filepath, nrows=0).columns
        use = [c for c in header if c in required_columns()]
        return pd.read_csv(filepath, usecols=use, low_memory=False), "edge-iiotset"
    warnings.warn(
        f"Dataset not found at {filepath}. Falling back to the SYNTHETIC sample - "
        "download Edge-IIoTset for real results (see README).", stacklevel=2)
    if not SYNTHETIC_DATASET_PATH.exists():
        SYNTHETIC_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(SYNTHETIC_DATASET_PATH, index=False)
    return pd.read_csv(SYNTHETIC_DATASET_PATH, low_memory=False), "synthetic"


def load_dataset(filepath=DATASET_PATH, sample_size=100000, test_size=0.2, seed=42) -> Dataset:
    df, source = _read_raw(filepath)
    df = df[df["Attack_type"].isin(TARGET_ATTACKS)]

    # Balanced sample: 50 % Normal / 50 % attacks (split equally over the 4 DDoS types).
    # (Review-2 first run used 20 % Normal, which pushed models to over-predict "attack".)
    attacks = [t for t in TARGET_ATTACKS if t != "Normal"]
    caps = {"Normal": sample_size // 2, **{t: sample_size // (2 * len(attacks)) for t in attacks}}
    df = pd.concat(
        [g.sample(min(len(g), caps[t]), random_state=seed) for t, g in df.groupby("Attack_type")],
        ignore_index=True)

    X = extract_from_dataframe(df)
    y = (df["Attack_type"] != "Normal").astype(int).to_numpy()   # 0 normal, 1 DDoS
    types = df["Attack_type"].to_numpy()

    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(
        X, y, types, test_size=test_size, random_state=seed, stratify=types)
    return Dataset(Xtr, Xte, ytr, yte, ttr, tte, list(FEATURE_NAMES), source)


def load_and_preprocess(filepath=DATASET_PATH, sample_size=100000):
    """Review-1 compatible signature: (X_train, X_test, y_train, y_test, features)."""
    d = load_dataset(filepath, sample_size)
    return d.X_train, d.X_test, d.y_train, d.y_test, d.features
