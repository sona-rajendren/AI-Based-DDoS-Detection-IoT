"""Sprint 1: dataset analysis for the report.  python -m src.eda
Writes reports/class_distribution.png, reports/feature_means_by_class.png, reports/dataset_summary.md
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import REPORT_DIR
from src.data_pipeline import load_dataset


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    d = load_dataset()
    X = np.vstack([d.X_train, d.X_test]); t = np.concatenate([d.types_train, d.types_test])
    classes = ["Normal", "DDoS_TCP", "DDoS_UDP", "DDoS_ICMP", "DDoS_HTTP"]

    counts = [int((t == c).sum()) for c in classes]
    plt.figure(figsize=(6, 3.6))
    plt.bar(classes, counts, color=["#2a9d8f"] + ["#e76f51"] * 4)
    plt.ylabel("samples"); plt.title(f"Class distribution after sampling ({d.source})")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150); plt.close()

    means = np.array([X[t == c].mean(axis=0) for c in classes])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(means, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(d.features)), d.features, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for (i, j), v in np.ndenumerate(means):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)
    plt.colorbar(im); ax.set_title("Mean (normalised) feature value per traffic class")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "feature_means_by_class.png", dpi=150); plt.close()

    lines = [f"Data source: **{d.source}**", f"Total rows used: {len(t)} (train {len(d.y_train)}, test {len(d.y_test)})",
             f"Features ({len(d.features)}): {', '.join(d.features)}", "", "| Class | Samples |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in zip(classes, counts)]
    (REPORT_DIR / "dataset_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
