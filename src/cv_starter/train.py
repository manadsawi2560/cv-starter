import os, yaml, json, random
import numpy as np
import tensorflow as tf
from cv_starter.data import load_dataset, make_tfds
from cv_starter.model import build_mobilenetv2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def configure_gpu_and_mixed_precision(enable_mixed=True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        print("Using GPU:", tf.config.list_logical_devices('GPU'))
        if enable_mixed:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision: enabled (float16)")
    else:
        print("GPU not found → using CPU")
        if enable_mixed:
            print("Mixed precision disabled on CPU")

def main(cfg):
    set_seed(cfg.get('seed', 42))
    configure_gpu_and_mixed_precision(cfg.get('mixed_precision', True))

    train_ds, val_ds, test_ds, n_classes = load_dataset()
    train_tfds = make_tfds(train_ds, cfg['img_size'], n_classes, cfg['batch_size'], aug=True)
    val_tfds   = make_tfds(val_ds,   cfg['img_size'], n_classes, cfg['batch_size'])
    test_tfds  = make_tfds(test_ds,  cfg['img_size'], n_classes, cfg['batch_size'])

    model = build_mobilenetv2(cfg['img_size'], n_classes, base_trainable=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg['lr']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_tfds, validation_data=val_tfds, epochs=cfg['epochs'])

    # optional fine-tune
    if cfg.get('fine_tune', False):
        base = None
        # หา base โดยดูจากชนิดเลเยอร์ (Model)
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.Model) and 'mobilenetv2' in lyr.name.lower():
                base = lyr; break
        if base is None:
            base = model.layers[2]  # fallback

        base.trainable = True
        cut = cfg.get('fine_tune_layers', 30)
        for l in base.layers[:-cut]:
            l.trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(cfg['lr'] * 0.1),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        ft_epochs = max(3, cfg['epochs'] // 2)
        model.fit(train_tfds, validation_data=val_tfds, epochs=ft_epochs)

    os.makedirs(os.path.join(cfg['out_dir'], 'models'), exist_ok=True)
    os.makedirs(os.path.join(cfg['out_dir'], 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(cfg['out_dir'], 'plots'), exist_ok=True)

    model_path = os.path.join(cfg['out_dir'], 'models', cfg.get('model_name', 'bloodmnist_mnv2.h5'))
    model.save(model_path)

    test_loss, test_acc = model.evaluate(test_tfds, verbose=0)
    with open(os.path.join(cfg['out_dir'], 'metrics', 'summary.json'), 'w') as f:
        json.dump({'test_loss': float(test_loss), 'test_acc': float(test_acc)}, f, indent=2)

    print({'model_path': model_path, 'test_acc': float(test_acc)})

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
