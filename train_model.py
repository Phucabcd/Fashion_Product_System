import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "archive"

TRAIN_CSV = DATA_DIR / "fashion-mnist_train.csv"
TEST_CSV = DATA_DIR / "fashion-mnist_test.csv"

def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 1. Tách label
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    # 2. Tách pixel
    x_train = train_df.drop(columns=["label"]).to_numpy()
    x_test = test_df.drop(columns=["label"]).to_numpy()

    # 3. Reshape 784 pixel → 28 × 28
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    # 4. Normalize 0-255 → 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)

    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    print("Pixel min:", x_train.min())
    print("Pixel max:", x_train.max())

    return (x_train, y_train), (x_test, y_test)


#dataset Fashion mnist đã train và test nên không cần train_test_split
def train_baseline(x_train, y_train, x_test, y_test):
    # 28 x 28 → 784
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    print("X_train_flat:", x_train_flat.shape)
    print("X_test_flat:", x_test_flat.shape)

    clf = LogisticRegression(
        max_iter=200,
    )

    clf.fit(x_train_flat, y_train)

    acc = clf.score(x_test_flat, y_test)

    print(f"[Baseline] Test accuracy: {acc:.4f}")

    return clf, acc


def train_cnn(x_train, y_train, x_test, y_test):
    x_train_cnn = x_train[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]

    print("CNN X_train:", x_train_cnn.shape)
    print("CNN X_test:", x_test_cnn.shape)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()

    model, accuracy = train_baseline(
        x_train,
        y_train,
        x_test,
        y_test
    )

    train_cnn(
        x_train,
        y_train,
        x_test,
        y_test
    )