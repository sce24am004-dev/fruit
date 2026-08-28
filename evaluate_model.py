import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model/freshness_model.keras"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_names=["fresh", "rotten"],
    shuffle=False
)

loss, accuracy = model.evaluate(test_ds, verbose=1)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probabilities = model.predict(test_ds, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Fresh", "Rotten"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
