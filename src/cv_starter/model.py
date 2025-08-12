import tensorflow as tf


def build_mobilenetv2(
    img_size: int,
    n_classes: int,
    base_trainable: bool = False,
    alpha: float = 1.0,  # width multiplier ของ MobileNetV2
    dropout: float = 0.2,
    l2: float = 0.0,  # L2 regularization ที่หัว
) -> tf.keras.Model:
    # Backbone
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        alpha=alpha,
    )
    base.trainable = base_trainable
    base._name = "mobilenetv2_backbone"  # ชื่อไว้ค้นตอน fine-tune

    # Head
    inputs = tf.keras.Input((img_size, img_size, 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)  # BN ทำงานแบบ inference ระหว่าง pretrain
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)

    kernel_reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    # logits ใช้ float32 เสมอ แม้เปิด mixed precision
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        kernel_regularizer=kernel_reg,
        dtype="float32",
        name="cls_head",
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"mnv2_{alpha:g}_{img_size}")
    return model
