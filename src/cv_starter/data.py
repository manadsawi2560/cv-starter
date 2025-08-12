import tensorflow as tf
from medmnist import INFO, BloodMNIST


def load_dataset(as_rgb=True):
    info = INFO["bloodmnist"]
    n_classes = len(info["label"])
    train = BloodMNIST(split="train", download=True, as_rgb=as_rgb)
    val = BloodMNIST(split="val", download=True, as_rgb=as_rgb)
    test = BloodMNIST(split="test", download=True, as_rgb=as_rgb)
    return train, val, test, n_classes


def make_tfds(ds, img_size, n_classes, batch, aug=False, shuffle=2048):
    def preprocess(x, y):
        # y from MedMNIST is shape (1,), keep sparse labels
        x = tf.image.resize(x, (img_size, img_size))
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(tf.squeeze(y, axis=-1), tf.int32)  # scalar class id
        return x, y

    x = tf.convert_to_tensor(ds.imgs)  # uint8
    y = tf.convert_to_tensor(ds.labels)  # shape (N,1)
    d = tf.data.Dataset.from_tensor_slices((x, y))

    if aug:
        d = d.map(
            lambda a, b: (tf.image.random_flip_left_right(a), b),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    d = d.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        d = d.shuffle(shuffle)
    d = d.batch(batch).prefetch(tf.data.AUTOTUNE)
    return d
