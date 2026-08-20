import joblib
import matplotlib.pyplot as plt

history = joblib.load("models/cnn_history.joblib")

print(history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history["loss"], label="train loss")
axes[0].plot(history["val_loss"], label="val loss")
axes[0].set_title("Loss theo epoch")
axes[0].set_xlabel("Epoch")
axes[0].legend()

# Accuracy
axes[1].plot(history["accuracy"], label="train acc")
axes[1].plot(history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy theo epoch")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.show()