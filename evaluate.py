# eval.py — file MỚI, tách riêng, không đụng vào train_model.py
# import joblib
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.metrics import f1_score, confusion_matrix, classification_report
# from train_model import load_data   # tái sử dụng hàm load_data() đã có

# (x_train, y_train), (x_test, y_test) = load_data()

# # --- Baseline ---
# baseline = joblib.load("outputs/baseline_model.pkl")
# x_test_flat = x_test.reshape(len(x_test), -1)
# y_pred_baseline = baseline.predict(x_test_flat)

# print("Baseline Macro F1:", f1_score(y_test, y_pred_baseline, average="macro"))
# print(confusion_matrix(y_test, y_pred_baseline))

# # --- CNN ---
# cnn = load_model("outputs/cnn_model.keras")
# x_test_cnn = x_test.reshape(-1, 28, 28, 1)
# y_pred_cnn = np.argmax(cnn.predict(x_test_cnn), axis=1)

# print("CNN Macro F1:", f1_score(y_test, y_pred_cnn, average="macro"))
# print(confusion_matrix(y_test, y_pred_cnn))