import joblib
import numpy as np
from pathlib import Path

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "archive" / "images"
MODELS_DIR = BASE_DIR / "models"

FEATURES_PATH = MODELS_DIR / "Images_features.pkl"
FILENAMES_PATH = MODELS_DIR / "filenames.pkl"


image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = [
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
]

print(f"Dataset path: {DATASET_DIR}")
print(f"Found {len(image_paths)} images")


def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    return __import__("tensorflow").keras.models.Sequential([
        base_model,
        GlobalMaxPool2D()
    ])


def extract_features(image_path, model):
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    feature = model.predict(
        img_array,
        verbose=0
    ).flatten()

    feature = feature / np.linalg.norm(feature)

    return feature


def train():
    MODELS_DIR.mkdir(exist_ok=True)

    model = build_model()

    features = []
    filenames = []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_paths)} images")

    for i, image_path in enumerate(image_paths):

        try:
            feature = extract_features(
                image_path,
                model
            )

            features.append(feature)
            filenames.append(str(image_path))

            print(
                f"[{i + 1}/{len(image_paths)}] "
                f"{image_path.name}"
            )

        except Exception as e:
            print(f"Skip {image_path}: {e}")

    features = np.array(features)

    joblib.dump(
        features,
        FEATURES_PATH
    )

    joblib.dump(
        filenames,
        FILENAMES_PATH
    )

    print("Done!")
    print(f"Features: {FEATURES_PATH}")
    print(f"Filenames: {FILENAMES_PATH}")


if __name__ == "__main__":
    train()