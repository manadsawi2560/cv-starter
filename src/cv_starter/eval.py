import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from cv_starter.data import load_dataset, make_tfds


def main(cfg, model_path):
    # ----- Data -----
    _, _, test_ds, n_classes = load_dataset()
    test_tfds = make_tfds(test_ds, cfg["img_size"], n_classes, cfg["batch_size"])

    # ----- Model & Predict -----
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(test_tfds)
    y_pred = probs.argmax(axis=1)

    # y_true (sparse labels) จาก pipeline
    y_true = []
    for _, y in test_tfds.unbatch().batch(1):
        y_true.append(int(tf.squeeze(y).numpy()))
    y_true = np.array(y_true)

    # ----- I/O dirs -----
    out_dir = cfg["out_dir"]
    metrics_dir = os.path.join(out_dir, "metrics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ----- Report (ต้องคำนวณก่อนใช้) -----
    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True, zero_division=0
    )
    with open(os.path.join(metrics_dir, "class_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ----- Confusion Matrix: count -----
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(n_classes))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (count)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)

    # ----- Confusion Matrix: normalized (per true class) -----
    cm_norm = confusion_matrix(
        y_true, y_pred, labels=range(n_classes), normalize="true"
    )
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=range(n_classes))
    disp.plot(values_format=".2f", cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix_norm.png"), dpi=150)
    plt.close("all")

    # ----- Per-class precision / recall / F1 -----
    classes = [k for k in report.keys() if k.isdigit()]  # '0'..'7'
    prec = [report[c]["precision"] for c in classes]
    rec = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    w = 0.25
    plt.figure(figsize=(10, 4))
    plt.bar(x - w, prec, width=w, label="precision")
    plt.bar(x, rec, width=w, label="recall")
    plt.bar(x + w, f1, width=w, label="f1")
    plt.xticks(x, classes)
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title("Per-class metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_class_metrics.png"), dpi=150)
    plt.close()

    # ----- Confidence histogram -----
    conf = np.max(probs, axis=1)
    plt.figure(figsize=(6, 4))
    plt.hist(conf, bins=20)
    plt.xlabel("max probability")
    plt.ylabel("count")
    plt.title("Prediction confidence (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confidence_hist.png"), dpi=150)
    plt.close()

    # ----- Print nicely to console -----
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="results/models/bloodmnist_mnv2.h5")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.model)
