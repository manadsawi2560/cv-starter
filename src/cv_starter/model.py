import tensorflow as tf

def build_mobilenetv2(img_size, n_classes, base_trainable=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = base_trainable

    inputs = tf.keras.Input((img_size, img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    dtype = 'float32'  # keep float32 for logits even if mixed precision on
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', dtype=dtype)(x)
    return tf.keras.Model(inputs, outputs)
