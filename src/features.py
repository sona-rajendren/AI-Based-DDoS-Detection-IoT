"""Shared feature engineering.

The SAME 10 features are produced
  * offline, from Edge-IIoTset CSV columns   (extract_from_dataframe)
  * online,  from live packets                (added in Sprint 3)
so a model trained on the dataset can later score live traffic.

Scaling is deterministic (fixed divisors) instead of a fitted StandardScaler:
every federated client and every edge device then scales identically without
having to share statistics of its raw data.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "tcp_syn", "tcp_ack", "tcp_rst", "tcp_fin", "tcp_len",
    "udp_present", "icmp_present", "http_present",
    "mqtt_msgtype", "mqtt_len",
]
N_FEATURES = len(FEATURE_NAMES)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

_ABSENT = {"", "0", "0.0", "nan", "none", "NaN"}


def _present(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    return (~s.isin(_ABSENT) & series.notna()).astype(float).to_numpy()


def _num(series: pd.Series) -> np.ndarray:
    # values such as "16,3" (several MQTT msgs in one frame) -> first value
    s = series.astype(str).str.split(",").str[0]
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=float)


def _first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def required_columns():
    """All dataset columns we can use (for fast `usecols` loading)."""
    return {
        "tcp.connection.syn", "tcp.flags.syn", "tcp.connection.fin", "tcp.flags.fin",
        "tcp.connection.rst", "tcp.flags.rst", "tcp.flags.ack", "tcp.len",
        "udp.port", "icmp.checksum", "http.request.method", "http.request.version",
        "mqtt.msgtype", "mqtt.len", "Attack_type", "Attack_label",
    }


def normalize(raw: np.ndarray) -> np.ndarray:
    """raw (n, N_FEATURES) -> values in ~[0, 1]."""
    x = np.array(raw, dtype=float, copy=True)
    x = np.atleast_2d(x)
    x[:, IDX["tcp_len"]] = np.clip(x[:, IDX["tcp_len"]] / 1460.0, 0, 1)
    x[:, IDX["mqtt_msgtype"]] = np.clip(x[:, IDX["mqtt_msgtype"]] / 15.0, 0, 1)
    x[:, IDX["mqtt_len"]] = np.clip(x[:, IDX["mqtt_len"]] / 256.0, 0, 1)
    return x


def extract_raw_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    cols = {}

    def flag(name, cands):
        s = _first_col(df, cands)
        cols[name] = (_num(s) > 0).astype(float) if s is not None else np.zeros(n)

    flag("tcp_syn", ["tcp.connection.syn", "tcp.flags.syn"])
    flag("tcp_ack", ["tcp.flags.ack"])
    flag("tcp_rst", ["tcp.connection.rst", "tcp.flags.rst"])
    flag("tcp_fin", ["tcp.connection.fin", "tcp.flags.fin"])
    s = _first_col(df, ["tcp.len"]); cols["tcp_len"] = _num(s) if s is not None else np.zeros(n)
    s = _first_col(df, ["udp.port"]); cols["udp_present"] = _present(s) if s is not None else np.zeros(n)
    s = _first_col(df, ["icmp.checksum"]); cols["icmp_present"] = _present(s) if s is not None else np.zeros(n)
    http = np.zeros(n)
    for c in ("http.request.method", "http.request.version"):
        if c in df.columns:
            http = np.maximum(http, _present(df[c]))
    cols["http_present"] = http
    s = _first_col(df, ["mqtt.msgtype"]); cols["mqtt_msgtype"] = _num(s) if s is not None else np.zeros(n)
    s = _first_col(df, ["mqtt.len"]); cols["mqtt_len"] = _num(s) if s is not None else np.zeros(n)
    return np.column_stack([cols[f] for f in FEATURE_NAMES])


def extract_from_dataframe(df: pd.DataFrame) -> np.ndarray:
    return normalize(extract_raw_from_dataframe(df))
