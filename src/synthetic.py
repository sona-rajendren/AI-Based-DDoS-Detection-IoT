"""Synthetic traffic generator that mimics the Edge-IIoTset column layout.

Used (a) so the whole project runs without the 1 GB Kaggle download,
(b) by unit tests, (c) by the live `--simulate` mode of the sniffer.
Per-packet classes deliberately overlap (e.g. benign HTTP vs. HTTP flood) so the
numbers are realistic and not trivially 100 %. Replace with the REAL dataset for
all reported results.
"""
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, IDX, N_FEATURES

ATTACK_CLASSES = ["DDoS_UDP", "DDoS_ICMP", "DDoS_HTTP", "DDoS_TCP"]


def _row(**kw):
    v = np.zeros(N_FEATURES)
    for k, val in kw.items():
        v[IDX[k]] = val
    return v


def _normal(rng):
    r = rng.random()
    if r < 0.55:   # MQTT PUBLISH
        return _row(tcp_ack=1, tcp_len=rng.integers(20, 120), mqtt_msgtype=3, mqtt_len=rng.integers(20, 100))
    if r < 0.65:   # other MQTT (PINGREQ / SUBSCRIBE)
        return _row(tcp_ack=1, tcp_len=rng.integers(2, 30), mqtt_msgtype=rng.choice([8, 12]), mqtt_len=rng.integers(2, 20))
    if r < 0.75:   # TCP handshake
        return _row(tcp_syn=1, tcp_ack=float(rng.random() < 0.5))
    if r < 0.83:   # bare ACK / FIN
        return _row(tcp_ack=1, tcp_fin=float(rng.random() < 0.25))
    if r < 0.90:   # benign HTTP
        return _row(tcp_ack=1, tcp_len=rng.integers(200, 600), http_present=1)
    if r < 0.95:
        return _row(udp_present=1)
    return _row(icmp_present=1)


def _attack(kind, rng):
    if rng.random() < 0.06:       # attacker packets that look like normal ones
        return _normal(rng)
    if kind == "DDoS_TCP":
        return _row(tcp_syn=1, tcp_rst=float(rng.random() < 0.1))
    if kind == "DDoS_UDP":
        return _row(udp_present=1)
    if kind == "DDoS_ICMP":
        return _row(icmp_present=1)
    if kind == "DDoS_HTTP":
        if rng.random() < 0.75:
            return _row(tcp_ack=1, tcp_len=rng.integers(150, 700), http_present=1)
        return _row(tcp_syn=1)
    raise ValueError(kind)


def make_raw(kind: str, n: int, rng) -> np.ndarray:
    fn = _normal if kind == "Normal" else (lambda g: _attack(kind, g))
    return np.vstack([fn(rng) for _ in range(n)])


def raw_to_dataset_frame(raw: np.ndarray, label: str) -> pd.DataFrame:
    """Write raw vectors in the Edge-IIoTset column format."""
    g = lambda name: raw[:, IDX[name]]
    n = len(raw)
    return pd.DataFrame({
        "tcp.connection.syn": g("tcp_syn"),
        "tcp.flags.ack": g("tcp_ack"),
        "tcp.connection.rst": g("tcp_rst"),
        "tcp.connection.fin": g("tcp_fin"),
        "tcp.len": g("tcp_len"),
        "udp.port": np.where(g("udp_present") > 0, 5683, 0.0),
        "icmp.checksum": np.where(g("icmp_present") > 0, 12345, 0.0),
        "http.request.method": np.where(g("http_present") > 0, "GET", "0.0"),
        "http.request.version": np.where(g("http_present") > 0, "HTTP/1.1", "0.0"),
        "mqtt.msgtype": g("mqtt_msgtype"),
        "mqtt.len": g("mqtt_len"),
        "mqtt.topic": np.where(g("mqtt_msgtype") == 3, "Temperature_and_Humidity", "0.0"),
        "Attack_label": 0 if label == "Normal" else 1,
        "Attack_type": label,
    })


def generate_dataset(n_normal=8000, n_per_attack=3000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [raw_to_dataset_frame(make_raw("Normal", n_normal, rng), "Normal")]
    for k in ATTACK_CLASSES:
        parts.append(raw_to_dataset_frame(make_raw(k, n_per_attack, rng), k))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
