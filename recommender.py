import pickle as pkl
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image as keras_image

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"

IMAGE_FEATURES_PATH = MODELS_DIR / "Images_features.pkl"
FILENAMES_PATH = MODELS_DIR / "filenames.pkl"

STYLES_PATH = BASE_DIR / "data" / "archive" / "styles.csv"

def load_artifacts() -> tuple:
    image_features = MODELS_DIR / "Images_features.pkl"
    filenames = MODELS_DIR / "filenames.pkl"
    
    missing = [
            path.name
            for path in (image_features, filenames)
            if not path.exists()
        ]
    if missing:
            raise FileNotFoundError(
                "Thiếu file model: "
                + ", ".join(missing)
                + ". Hãy chạy `python train_model.py` trước."
            )
    
    image_features = joblib.load(image_features)
    filenames = joblib.load(filenames)
    
    return image_features, filenames


def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])
    return model


def build_neighbors(image_features):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(image_features)
    return neighbors


def extract_features_from_images(image_source, model):
    if hasattr(image_source, "read"):
        image_source.seek(0)
        img = Image.open(image_source).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")

    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    img_array = keras_image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess, verbose=0).flatten()
    return result / norm(result)


def save_uploaded_image(uploaded_file):
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def get_recommendations(uploaded_file, model, neighbors, filenames, top_k=6):
    save_path = save_uploaded_image(uploaded_file)
    input_img_features = extract_features_from_images(save_path, model)
    _, indices = neighbors.kneighbors([input_img_features])
    recommended = [filenames[idx] for idx in indices[0][1:top_k]]
    return save_path, recommended


def load_styles():
    if not STYLES_PATH.exists():
        return None
    df = pd.read_csv(STYLES_PATH, on_bad_lines="skip")
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def get_product_info(image_path, styles_df):
    if styles_df is None:
        return None
    product_id = Path(image_path).stem  # "1163.jpg" -> "1163"
    if product_id not in styles_df.index:
        return None
    row = styles_df.loc[product_id]
    return {
        "name": row.get("productDisplayName", "N/A"),
        "category": row.get("articleType", "N/A"),
        "color": row.get("baseColour", "N/A"),
        "usage": row.get("usage", "N/A"),
    }