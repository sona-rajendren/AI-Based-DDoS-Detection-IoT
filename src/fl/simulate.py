"""In-process FedAvg simulation (no network / Flower needed) - same maths as the
Flower server. Produces the convergence data used in the Review-2 report.
Run:  python -m src.fl.simulate [--clients 3] [--rounds 10] [--non-iid]
"""
import argparse
import json

from src.config import FL_NUM_CLIENTS, FL_NUM_ROUNDS, FL_LOCAL_EPOCHS, MODEL_PATH, REPORT_DIR
from src.data_pipeline import load_dataset
from src.fl.common import FL_MODEL, build_model, get_params, set_params, local_train, partition, fedavg
from src.metrics import evaluate_model
from src.model_io import save_bundle


def _plot(history, clients, non_iid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for k in ("accuracy", "f1", "roc_auc"):
        plt.plot([h["round"] for h in history], [h[k] for h in history], marker="o", label=k)
    plt.xlabel("FedAvg round"); plt.ylabel("score"); plt.grid(alpha=0.3); plt.legend()
    plt.title(f"Federated convergence ({clients} clients, {'non-IID' if non_iid else 'IID'})")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "fl_convergence.png", dpi=150); plt.close()


def run(clients=FL_NUM_CLIENTS, rounds=FL_NUM_ROUNDS, epochs=FL_LOCAL_EPOCHS, non_iid=False, save=True):
    d = load_dataset()
    parts = [partition(d.X_train, d.y_train, clients, c, non_iid=non_iid) for c in range(clients)]
    glob = build_model()
    history = []
    for r in range(1, rounds + 1):
        updates, sizes = [], []
        for c, (X, y) in enumerate(parts):
            local = set_params(build_model(), get_params(glob))
            local_train(local, X, y, epochs, seed=r * 100 + c)
            updates.append(get_params(local)); sizes.append(len(X))
        set_params(glob, fedavg(updates, sizes))
        m, _ = evaluate_model(glob, d.X_test, d.y_test, d.types_test)
        history.append({"round": r, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
        print(f"round {r:2d}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['roc_auc']:.4f}")
    if save:
        save_bundle(glob, MODEL_PATH, "FedAvg-" + FL_MODEL, metrics=m,
                    extra={"clients": clients, "rounds": rounds, "non_iid": non_iid, "data_source": d.source})
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / "fl_history.json").write_text(json.dumps(history, indent=2))
        _plot(history, clients, non_iid)
        print(f"saved {MODEL_PATH}")
    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=FL_NUM_CLIENTS)
    ap.add_argument("--rounds", type=int, default=FL_NUM_ROUNDS)
    ap.add_argument("--epochs", type=int, default=FL_LOCAL_EPOCHS)
    ap.add_argument("--non-iid", action="store_true")
    a = ap.parse_args()
    run(a.clients, a.rounds, a.epochs, a.non_iid)
