# import numpy as np
# import joblib
# import tensorflow as tf
# from pathlib import Path
# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "models"

# CLASS_NAMES = [
#     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
#     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
# ]

# resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(96, 96, 3))
# emb = joblib.load(MODEL_DIR / "embedding_db.pkl")

# def prepare_single(image_28x28: np.ndarray) -> np.ndarray:
#     img = image_28x28[np.newaxis, ..., np.newaxis]         # (1, 28, 28, 1)
#     img = tf.image.resize(img, [96, 96])
#     img = tf.image.grayscale_to_rgb(img)
#     img = img.numpy() * 255.0
#     return preprocess_input(img)

# def find_similar(image_28x28: np.ndarray, top_k: int = 5):
#     """image_28x28: ảnh grayscale đã normalize [0,1], shape (28,28)"""
#     query_ready = prepare_single(image_28x28)
#     query_embedding = resnet.predict(query_ready, verbose=0)

#     sims = cosine_similarity(query_embedding, emb["embeddings"])[0]
#     top_indices = np.argsort(sims)[::-1][:top_k]

#     results = []
#     for idx in top_indices:
#         results.append({
#             "image": emb["images"][idx],
#             "label": CLASS_NAMES[emb["labels"][idx]],
#             "similarity": float(sims[idx])
#         })
#     return results