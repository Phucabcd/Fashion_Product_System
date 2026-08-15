from train_model import load_data
from PIL import Image
import numpy as np


(x_train, y_train), (x_test, y_test) = load_data()

indices = np.random.choice(len(x_train), size=10, replace=False)

for index ,i in enumerate(indices):
    img_array = (x_train[i] * 255).astype(np.uint8)
    Image.fromarray(img_array).save(f"uploads/sample_{index}.png")
    print(f"Đã lưu sample_{index} vào uploads")

