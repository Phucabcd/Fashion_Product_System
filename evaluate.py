import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score ,f1_score, confusion_matrix, classification_report
from train_model import load_data   

(x_train, y_train), (x_test, y_test) = load_data()

# --- Baseline ---
baseline = joblib.load("models/baseline_model.pkl")
x_test_flat = x_test.reshape(len(x_test), -1)
y_pred_baseline = baseline.predict(x_test_flat)

print("Accurazy: ", accuracy_score(y_test, y_pred_baseline))
print("Baseline Macro F1:", f1_score(y_test, y_pred_baseline, average="macro"))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_baseline))

# --- CNN ---
cnn = load_model("models/cnn_model.keras")
x_test_cnn = x_test.reshape(-1, 28, 28, 1)
y_pred_cnn = np.argmax(cnn.predict(x_test_cnn), axis=1)

print("Accurazy: ", accuracy_score(y_test, y_pred_cnn))
print("CNN Macro F1:", f1_score(y_test, y_pred_cnn, average="macro"))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_cnn))