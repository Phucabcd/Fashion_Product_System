import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "archive"

TRAIN_CSV = DATA_DIR / "fashion-mnist_train.csv"
TEST_CSV = DATA_DIR / "fashion-mnist_test.csv"

SAVE_DIR = "outputs"

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Nếu không loại bỏ khi training model sẽ học label => nói cách khác học gian lận (data leakage)
    x_train = train_df.drop(columns=["label"]).to_numpy()   # Loại bỏ label ở x chỉ để lại pixel
    x_test = test_df.drop(columns=["label"]).to_numpy()     
    
    y_train = train_df["label"].to_numpy()                  # Kết quả trả về là label
    y_test = test_df["label"].to_numpy()                    # Lấy đúng target

    x_train = x_train.reshape(-1, 28, 28)                   # Chuyển thành ảnh dạng 2D
    x_test = x_test.reshape(-1, 28, 28)

    x_train = x_train.astype("float32") / 255.0             # Normalize 
    x_test = x_test.astype("float32") / 255.0

    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    return (x_train, y_train), (x_test, y_test)

#dataset Fashion mnist đã train và test nên không cần train_test_split
def train_baseline(x_train, y_train, x_test, y_test):
    x_train_flat = x_train.reshape(len(x_train), -1)        # Chuyển lại dạng 784 ban đầu 
    x_test_flat = x_test.reshape(len(x_test), -1)           # Vì baseline sẽ xử lí ở dạng (n_samples, n_features)

    clf = LogisticRegression(max_iter=200)                  # Giới hạn số vòng lập (mặc định của max_iter=100)
    clf.fit(x_train_flat, y_train)
    y_predict = clf.predict(x_test_flat)                    # Lưu ở X flat dễ nhầm lẫn 
    acc = accuracy_score(y_test, y_predict)

    print(f"[Baseline] Test accuracy: {acc:.4f}")
    return clf, acc
 
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)), 
                                #giữ nguyên h,w
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),    # Conv2D: tìm đặc trưng trong ảnh 
        layers.BatchNormalization(),                                    # Batch: Hổ trợ training ổn định hơn
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),                                     # MaxPooling2D: giảm kích thước giữ feature quan trọng
        layers.Dropout(0.25),                                           # Dropout: Giảm overfitting cho model

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),    # Tìm đặc trưng phức tạp hơn 
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),                                               # Flatten: làm thẳng lại (biến thành vector)
        layers.Dense(256, activation='relu'),                           # Dense: Dựa vào feature để phân loại
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',                                               # một thuật toán điều chỉnh weights để giảm lỗi.
        loss='sparse_categorical_crossentropy',                         # loss do: Model dự đoán sai bao nhiêu.
        metrics=['accuracy']                                            
    )

    model.summary()
    return model


def train_cnn(x_train, y_train, x_test, y_test):
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)            # CNN xử lí ở dạng (batch, height, width, channels)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)

    # Tách validation set từ train (KHÔNG đụng vào test set)
    # Bước này kiểm tra overfitting model 
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_cnn, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    model = build_cnn()

    # Những cơ chế của TensorFlow sẽ gọi trong quá trình training 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True), # model sẽ dừng khi Epoch tốt nhất
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    train_history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test_cnn, y_test)
    print(f"[CNN] Test accuracy: {test_acc:.4f}")

    model.save("outputs/cnn_model.keras")
    print("Đã lưu model: cnn_model.keras")

    joblib.dump(train_history.history, "outputs/cnn_history.joblib")
    print("Đã lưu history: cnn_history.joblib")

    return model, train_history, test_acc


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()

    baseline_model, baseline_acc = train_baseline(x_train, y_train, x_test, y_test)

    cnn_model, history, cnn_acc = train_cnn(x_train, y_train, x_test, y_test)

    print(f"\n[So sánh] Baseline: {baseline_acc:.4f} | CNN: {cnn_acc:.4f}")