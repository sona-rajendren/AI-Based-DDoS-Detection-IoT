"""Flower client (one per edge node).  Run:  python -m src.fl.client --cid 0"""
import argparse
import flwr as fl

from src.config import FL_SERVER_ADDRESS, FL_NUM_CLIENTS, FL_LOCAL_EPOCHS
from src.data_pipeline import load_dataset
from src.fl.common import build_model, get_params, set_params, local_train, partition
from src.metrics import evaluate_model


class IoTEdgeClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, epochs=FL_LOCAL_EPOCHS, seed=0):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.epochs, self.seed = epochs, seed
        self.model = build_model()

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        local_train(self.model, self.X_train, self.y_train, self.epochs, self.seed)
        return get_params(self.model), len(self.X_train), {}   # raw data never leaves the node

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        m, _ = evaluate_model(self.model, self.X_test, self.y_test)
        return float(1.0 - m["accuracy"]), len(self.X_test), {"accuracy": m["accuracy"], "f1": m["f1"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, default=0, help="client id (0..num-clients-1)")
    ap.add_argument("--num-clients", type=int, default=FL_NUM_CLIENTS)
    ap.add_argument("--server", default=FL_SERVER_ADDRESS)
    ap.add_argument("--non-iid", action="store_true", help="label-skewed local data")
    a = ap.parse_args()

    d = load_dataset()
    X, y = partition(d.X_train, d.y_train, a.num_clients, a.cid, non_iid=a.non_iid)
    print(f"[client {a.cid}] {len(X)} local training rows (source: {d.source})")
    client = IoTEdgeClient(X, y, d.X_test, d.y_test, seed=a.cid)
    fl.client.start_client(server_address=a.server, client=client.to_client())


if __name__ == "__main__":
    main()
