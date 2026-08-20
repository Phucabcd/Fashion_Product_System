import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "archive"
MODEL_DIR = BASE_DIR / "models_transfer"

def preprocess_transfer(x, target_size=96):
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, (-1, 28, 28, 1))
    x = tf.image.grayscale_to_rgb(x)
    x = tf.image.resize(x, [target_size, target_size])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

def build_transfer_model(input_shape=(96, 96, 3), num_classes=10):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Giai đoạn 1: freeze toàn bộ base

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def train_transfer_model(x_train, y_train, x_test, y_test):
    x_train_p = preprocess_transfer(x_train).numpy()
    x_test_p = preprocess_transfer(x_test).numpy()

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_p, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    model, base_model = build_transfer_model()

    # ===== Giai đoạn 1: train classifier, freeze base =====
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_stage1 = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # ===== Giai đoạn 2: unfreeze 20 layer cuối, fine-tune LR nhỏ =====
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history_stage2 = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    test_loss, test_acc = model.evaluate(x_test_p, y_test)
    print(f"[Transfer-MobileNetV2] Test accuracy: {test_acc:.4f}")

    # ==== Lưu riêng, không ghi đè file gốc ====
    model.save(MODEL_DIR / "mobilenetv2_transfer.keras")
    print(f"Đã lưu model: {MODEL_DIR / 'mobilenetv2_transfer.keras'}")

    combined_history = {
        "stage1": history_stage1.history,
        "stage2": history_stage2.history
    }
    joblib.dump(combined_history, MODEL_DIR / "mobilenetv2_transfer_history.joblib")
    print(f"Đã lưu history: {MODEL_DIR / 'mobilenetv2_transfer_history.joblib'}")

    return model, combined_history, test_acc


if __name__ == "__main__":
    # Load dữ liệu Fashion-MNIST theo cách bạn đang dùng ở file gốc
    # (giữ nguyên nguồn dữ liệu, chỉ import lại — không sửa gì ở đây)
    from tensorflow.keras.datasets import fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    train_transfer_model(x_train, y_train, x_test, y_test)