import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion_detection_resnet50.keras"
TEST_DIR = BASE_DIR / "data" / "fer2013" / "test"
LABELS_PATH = BASE_DIR / "resnet50_emotion_classes.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels_data = json.load(f)

if isinstance(labels_data, list):
    class_names = labels_data
else:
    class_names = [
        labels_data[str(i)]
        for i in range(len(labels_data))
    ]


# Load independent test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Actual labels
y_true = np.concatenate([
    y.numpy() for _, y in test_ds
])

# Predictions
probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)


# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro"
)
weighted_f1 = f1_score(
    y_true, y_pred, average="weighted"
)


# Display results
print("\n" + "=" * 50)
print("RESNET50 — FER2013 TEST RESULTS")
print("=" * 50)

print(f"Test images:       {len(y_true)}")
print(f"Classes:           {len(class_names)}")
print(f"Test Accuracy:     {accuracy * 100:.2f}%")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nPRESENTATION SUMMARY")
print(
    f"Successfully evaluated {len(y_true)} "
    f"unseen images across {len(class_names)} classes."
)
print(f"Overall test accuracy: {accuracy * 100:.2f}%")