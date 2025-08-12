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
    # รับ img_size ผ่าน default args -> ปิดช่องว่างเรื่อง closure
    def preprocess(x, y, _img_size=img_size):
        x = tf.image.resize(x, (_img_size, _img_size), method="bilinear")
        x = tf.cast(
            x, tf.float32
        )  # ทาง A: ไม่หาร 255 (ให้โมเดลทำ preprocess_input เอง)
        # กำหนด static shape ให้ชัด (ช่วยให้ Keras ตรวจถูก)
        x.set_shape([_img_size, _img_size, 3])
        y = tf.cast(tf.squeeze(y, axis=-1), tf.int32)
        return x, y

    x = tf.convert_to_tensor(ds.imgs)  # uint8, shape (N, 28, 28, 3)
    y = tf.convert_to_tensor(ds.labels)  # shape (N, 1)
    d = tf.data.Dataset.from_tensor_slices((x, y))

    if aug:

        def aug_fn(a):
            a = tf.image.random_flip_left_right(a)
            a = tf.image.random_brightness(a, max_delta=0.10)
            a = tf.image.random_contrast(a, lower=0.9, upper=1.1)
            a = tf.image.random_saturation(a, lower=0.9, upper=1.1)
            return a

        d = d.map(lambda a, b: (aug_fn(a), b), num_parallel_calls=tf.data.AUTOTUNE)

    d = d.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        d = d.shuffle(shuffle, reshuffle_each_iteration=True)

    # cache แล้วค่อย batch/prefetch จะไวขึ้น
    d = d.cache().batch(batch).prefetch(tf.data.AUTOTUNE)
    return d
