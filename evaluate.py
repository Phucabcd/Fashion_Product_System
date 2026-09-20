import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score ,f1_score, confusion_matrix, precision_score, recall_score
from train_model import load_data   
from train_transfer_model import preprocess_transfer

(x_train, y_train), (x_test, y_test) = load_data()

# --- Baseline ---
baseline = joblib.load("models/baseline_model.pkl")
x_test_flat = x_test.reshape(len(x_test), -1)
y_pred_baseline = baseline.predict(x_test_flat)

print("\nBaseline Logistic Regression Model\n")
print("Accuracy: ", accuracy_score(y_test, y_pred_baseline))
print("Baseline Macro F1:", f1_score(y_test, y_pred_baseline, average="macro"))
print("Precision score: ", precision_score(y_test, y_pred_baseline, average="macro"))
print("Recall score: ", recall_score(y_test, y_pred_baseline, average="macro"))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_baseline))

# --- CNN ---
cnn = load_model("models/cnn_model.keras")
x_test_cnn = x_test.reshape(-1, 28, 28, 1)
y_pred_cnn = np.argmax(cnn.predict(x_test_cnn), axis=1)

print("\nCNN Model\n")
print("Accuracy: ", accuracy_score(y_test, y_pred_cnn))
print("CNN Macro F1:", f1_score(y_test, y_pred_cnn, average="macro"))
print("Precision score: ", precision_score(y_test, y_pred_cnn, average="macro"))
print("Recall score: ", recall_score(y_test, y_pred_cnn, average="macro"))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_cnn))

# --- MobileNetV2 Model ---
mobilenet = load_model("models_transfer/mobilenetv2_transfer.keras")


x_test_raw = (x_test * 255.0).astype("float32")
x_test_mobilenet = preprocess_transfer(x_test_raw)
y_pred_mobilenet = np.argmax(mobilenet.predict(x_test_mobilenet), axis=1)

print("\nMobileNetV2 Model\n")
print("Accuracy: ", accuracy_score(y_test, y_pred_mobilenet))
print("MobileNetV2 Macro F1:", f1_score(y_test, y_pred_mobilenet, average="macro"))
print("Precision score: ", precision_score(y_test, y_pred_mobilenet, average="macro"))
print("Recall score: ", recall_score(y_test, y_pred_mobilenet, average="macro"))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_mobilenet))