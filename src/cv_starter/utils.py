from typing import Tuple
import numpy as np


def normalize_image(x: np.ndarray) -> np.ndarray:
    """Scale uint8 image [0,255] -> float32 [0.,1.]."""
    x = x.astype("float32")
    return x / 255.0


def simple_cnn_model(input_shape: Tuple[int, int, int] | None = None):
    """
    สร้างโมเดล Keras แบบเบาๆ (import ภายในฟังก์ชัน เพื่อไม่ให้ CI ล้มถ้าไม่ได้ติดตั้ง tensorflow)
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:
        raise ImportError(
            "TensorFlow/Keras is not installed. Install extras with: pip install -e .[cv]"
        ) from e

    inputs = keras.Input(shape=input_shape or (32, 32, 3))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model
