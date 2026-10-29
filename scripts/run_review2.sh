#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m src.eda                                   # Sprint 1: dataset analysis
python -m src.train_baselines                       # reference models
python -m src.fl.simulate --clients 3 --rounds 10   # Sprint 2: FedAvg
python -m pytest -q tests
