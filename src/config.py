"""Global configuration and paths (override with environment variables)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path(os.getenv("DATASET_PATH", BASE_DIR / "dataset" / "DNN-EdgeIIoT-dataset.csv"))
SYNTHETIC_DATASET_PATH = BASE_DIR / "dataset" / "synthetic-edgeiiot-sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "global_model.pkl"          # FedAvg global model
DT_MODEL_PATH = MODEL_DIR / "decision_tree.pkl"      # centralised reference
NB_MODEL_PATH = MODEL_DIR / "naive_bayes.pkl"        # centralised reference
REPORT_DIR = BASE_DIR / "reports"

# Federated learning
FL_SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "127.0.0.1:8080")
FL_NUM_ROUNDS = int(os.getenv("FL_NUM_ROUNDS", 5))
FL_NUM_CLIENTS = int(os.getenv("FL_NUM_CLIENTS", 3))
FL_LOCAL_EPOCHS = int(os.getenv("FL_LOCAL_EPOCHS", 3))
