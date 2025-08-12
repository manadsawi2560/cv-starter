import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from cv_starter.data import load_dataset, make_tfds
from cv_starter.model import build_mobilenetv2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu_and_mixed_precision(enable_mixed=True):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print("Using GPU:", tf.config.list_logical_devices("GPU"))
        if enable_mixed:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision: enabled (float16)")
    else:
        print("GPU not found → using CPU")
        if enable_mixed:
            print("Mixed precision disabled on CPU")


def _merge_histories(*histories):
    merged = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    for h in histories:
        if not h:
            continue
        for k in merged.keys():
            if k in h.history:
                merged[k].extend(h.history[k])
    return merged


def main(cfg):
    set_seed(cfg.get("seed", 42))
    configure_gpu_and_mixed_precision(cfg.get("mixed_precision", True))

    # --- data ---
    train_ds, val_ds, test_ds, n_classes = load_dataset()
    train_tfds = make_tfds(
        train_ds, cfg["img_size"], n_classes, cfg["batch_size"], aug=True
    )
    val_tfds = make_tfds(val_ds, cfg["img_size"], n_classes, cfg["batch_size"])
    test_tfds = make_tfds(test_ds, cfg["img_size"], n_classes, cfg["batch_size"])

    # class weights (ช่วยเรื่อง imbalance)
    labels = train_ds.labels.flatten()
    class_counts = np.bincount(labels, minlength=n_classes)
    total = int(class_counts.sum())
    class_weight = {
        i: float(total / (n_classes * max(1, c))) for i, c in enumerate(class_counts)
    }
    print("class_weight:", class_weight)

    # --- model ---
    model = build_mobilenetv2(cfg["img_size"], n_classes, base_trainable=False)
    loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss=loss, metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=2, factor=0.3, monitor="val_loss"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg["out_dir"], "models", "best_mnv2.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
    ]

    # --- train ---
    hist_pre = model.fit(
        train_tfds,
        validation_data=val_tfds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # --- optional fine-tune ---
    hist_ft = None
    if cfg.get("fine_tune", False):
        # หา base ของ MobileNetV2
        base = None
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.Model) and "mobilenetv2" in lyr.name.lower():
                base = lyr
                break
        if base is None and len(model.layers) >= 3:
            base = model.layers[2]  # fallback

        if base is not None:
            base.trainable = True
            cut = cfg.get("fine_tune_layers", 30)
            for layer in base.layers[:-cut]:
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg["lr"] * 0.1),
            loss=loss,
            metrics=["accuracy"],
        )

        ft_epochs = max(5, cfg["epochs"] // 2)
        hist_ft = model.fit(
            train_tfds,
            validation_data=val_tfds,
            epochs=ft_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )

    # --- outputs ---
    os.makedirs(os.path.join(cfg["out_dir"], "models"), exist_ok=True)
    os.makedirs(os.path.join(cfg["out_dir"], "metrics"), exist_ok=True)
    os.makedirs(os.path.join(cfg["out_dir"], "plots"), exist_ok=True)

    # plot learning curves (รวม pretrain + finetune)
    hist_all = _merge_histories(hist_pre, hist_ft)

    plt.figure()
    plt.plot(hist_all["accuracy"])
    plt.plot(hist_all["val_accuracy"])
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "plots", "acc_curve.png"), dpi=150)

    plt.figure()
    plt.plot(hist_all["loss"])
    plt.plot(hist_all["val_loss"])
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "plots", "loss_curve.png"), dpi=150)
    plt.close("all")

    # save model (weights ที่ดีที่สุดถูก restore แล้วจาก EarlyStopping)
    model_path = os.path.join(
        cfg["out_dir"], "models", cfg.get("model_name", "bloodmnist_mnv2.h5")
    )
    model.save(model_path)

    # evaluate test
    test_loss, test_acc = model.evaluate(test_tfds, verbose=0)
    with open(os.path.join(cfg["out_dir"], "metrics", "summary.json"), "w") as f:
        json.dump(
            {"test_loss": float(test_loss), "test_acc": float(test_acc)}, f, indent=2
        )

    print({"model_path": model_path, "test_acc": float(test_acc)})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
