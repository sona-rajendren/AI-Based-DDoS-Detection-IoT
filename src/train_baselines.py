"""Train centralised Decision Tree + Naive Bayes baselines.  Run: python -m src.train_baselines"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from src.config import DT_MODEL_PATH, NB_MODEL_PATH
from src.data_pipeline import load_dataset
from src.metrics import evaluate_model
from src.model_io import save_bundle


def main():
    d = load_dataset()
    print(f"data source: {d.source} | train {d.X_train.shape} | test {d.X_test.shape}")
    # shallow tree => microsecond inference on a Raspberry Pi / ESP32-class gateway
    dt = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42).fit(d.X_train, d.y_train)
    nb = GaussianNB().fit(d.X_train, d.y_train)
    for name, model, path in (("DecisionTree", dt, DT_MODEL_PATH), ("NaiveBayes", nb, NB_MODEL_PATH)):
        m, _ = evaluate_model(model, d.X_test, d.y_test, d.types_test)
        save_bundle(model, path, name, metrics=m, extra={"data_source": d.source})
        print(f"{name:13s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} -> {path.name}")


if __name__ == "__main__":
    main()
