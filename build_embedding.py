# import tensorflow as tf
# import numpy as np
# import joblib 

# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from pathlib import Path
# from train_model import load_data

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "models"

# resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(96, 96, 3))

# def preprocess_resnet(in_image: np.ndarray) -> np.ndarray:
#     images = in_image[..., np.newaxis]              # (N, 28, 28, 1)
#     images = tf.image.resize(images, [96, 96])      # (N, 96, 96, 1)
#     images = tf.image.grayscale_to_rgb(images)      # (N, 96, 96, 3)
#     images = images.numpy() * 255.0

#     return preprocess_input(images)

# if __name__ == "__main__":
#     (x_train, y_train), (x_test, y_test) = load_data()
    
#     N = 2000
#     x_subset = x_train[:N]
#     y_subset = y_train[:N]
    
#     x_ready = preprocess_resnet(x_subset)
#     embeddings = resnet.predict(x_ready, batch_size=64, verbose=1)   # (N, 2048)
    
#     joblib.dump({
#         "embeddings": embeddings,
#         "images": x_subset,     # ảnh gốc 28x28 [0,1], để hiển thị lại sau này
#         "labels": y_subset
#     }, MODEL_DIR / "embedding_db.pkl")

#     print(f"Đã lưu {N} embeddings vào embedding_db.pkl")
    