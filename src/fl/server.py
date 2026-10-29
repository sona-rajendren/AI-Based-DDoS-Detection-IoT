"""Flower aggregation server (FedAvg).  Run:  python -m src.fl.server"""
import argparse
import json
import flwr as fl

from src.config import FL_NUM_ROUNDS, FL_NUM_CLIENTS, MODEL_PATH, REPORT_DIR
from src.fl.common import FL_MODEL, build_model, set_params
from src.model_io import save_bundle


def _weighted_avg(results):
    total = sum(n for n, _ in results)
    keys = results[0][1].keys() if results else []
    return {k: sum(n * m[k] for n, m in results) / total for k in keys}


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that also exports the aggregated model as models/global_model.pkl."""
    history = []

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            nd = fl.common.parameters_to_ndarrays(params)
            model = set_params(build_model(), nd)
            save_bundle(model, MODEL_PATH, "FedAvg-" + FL_MODEL, extra={"round": server_round})
            print(f"[server] round {server_round}: aggregated {len(results)} clients -> {MODEL_PATH.name}")
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.history.append({"round": server_round, "loss": loss, **(metrics or {})})
        print(f"[server] round {server_round}: eval {metrics}")
        return loss, metrics


def start_server(rounds=FL_NUM_ROUNDS, min_clients=FL_NUM_CLIENTS, address="0.0.0.0:8080"):
    strategy = SaveModelFedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=min_clients, min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=_weighted_avg,
    )
    print(f"Starting Flower FedAvg server on {address} ({rounds} rounds, waiting for {min_clients} clients)...")
    fl.server.start_server(server_address=address,
                           config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "fl_flower_history.json").write_text(json.dumps(strategy.history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=FL_NUM_ROUNDS)
    ap.add_argument("--min-clients", type=int, default=FL_NUM_CLIENTS)
    a = ap.parse_args()
    start_server(a.rounds, a.min_clients)
