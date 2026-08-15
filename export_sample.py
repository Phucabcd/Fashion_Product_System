from train_model import load_data
from PIL import Image
import numpy as np


(x_train, y_train), (x_test, y_test) = load_data()

for i in range(10):
    img_array = (x_train[i] * 255).astype(np.uint8)
    Image.fromarray(img_array).save(f"uploads/sample_{i}.png")
    print(f"Đã lưu sample_{[i]} vào uploads")

