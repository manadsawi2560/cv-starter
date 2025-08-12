import os
import yaml
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from cv_starter.data import load_dataset, make_tfds


def main(cfg, model_path):
    _, _, test_ds, n_classes = load_dataset()
    test_tfds = make_tfds(test_ds, cfg["img_size"], n_classes, cfg["batch_size"])

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(test_tfds)
    y_pred = probs.argmax(axis=1)

    # เอา y_true แบบ sparse จาก dataset pipeline
    y_true = []
    for _, y in test_tfds.unbatch().batch(1):
        y_true.append(int(tf.squeeze(y).numpy()))
    y_true = np.array(y_true)

    os.makedirs(os.path.join(cfg["out_dir"], "metrics"), exist_ok=True)
    os.makedirs(os.path.join(cfg["out_dir"], "plots"), exist_ok=True)

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    with open(os.path.join(cfg["out_dir"], "metrics", "class_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(cfg["out_dir"], "plots", "confusion_matrix.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()

    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="results/models/bloodmnist_mnv2.h5")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.model)
