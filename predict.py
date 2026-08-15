import numpy as np
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load model 1 lần khi module được import (không load lại mỗi lần predict)
baseline_model = joblib.load(MODEL_DIR / "baseline_model.pkl")
cnn_model = load_model(MODEL_DIR / "cnn_model.keras")
# baseline_scaler = joblib.load(MODEL_DIR / "baseline_scaler.pkl")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Nhận ảnh grayscale bất kỳ (H, W), trả về array (28, 28) đã normalize.
    Dùng chung logic cho cả 2 model, khác nhau ở bước reshape cuối.
    """
    if image.shape != (28, 28):
        from PIL import Image
        image = np.array(Image.fromarray(image).resize((28, 28)))

    image = image.astype("float32") / 255.0
    return image


def predict_baseline(image: np.ndarray):
    img = preprocess_image(image)
    img_flat = img.reshape(1, -1)     
    # img_scaled = baseline_scaler.transform(img_flat)# (1, 784)

    pred = baseline_model.predict(img_flat)[0]              # -> 0
    proba = baseline_model.predict_proba(img_flat)[0]       # probability-> array

    return CLASS_NAMES[pred], proba


def predict_cnn(image: np.ndarray):
    img = preprocess_image(image)
    img_cnn = img.reshape(1, 28, 28, 1)                # (1, 28, 28, 1)

    proba = cnn_model.predict(img_cnn, verbose=0)[0]
    pred = np.argmax(proba)

    return CLASS_NAMES[pred], proba


def predict(image: np.ndarray, model_type: str = "cnn"):
    """
    Hàm duy nhất mà app.py sẽ gọi.
    model_type: "baseline" hoặc "cnn"
    """
    if model_type == "baseline":
        return predict_baseline(image)
    elif model_type == "cnn":
        return predict_cnn(image)
    else:
        raise ValueError("model_type phải là 'baseline' hoặc 'cnn'")