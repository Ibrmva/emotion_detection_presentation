
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "data" / "fer2013" / "test"
MODEL_PATH = BASE_DIR / "best_custom_cnn.keras"
LABELS_PATH = BASE_DIR / "custom_emotion_classes.json"

OUTPUT_DIR = BASE_DIR / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

required_paths = [TEST_DIR, MODEL_PATH, LABELS_PATH]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file or directory not found:\n{path}\n"
            "Check your project structure and model training output."
        )

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError("The class-label JSON must contain a non-empty list.")

print("=" * 65)
print("CUSTOM CNN — INDEPENDENT TEST EVALUATION")
print("=" * 65)
print(f"Model:   {MODEL_PATH.name}")
print(f"Dataset: {TEST_DIR}")
print(f"Classes: {class_names}")

# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel architecture:")
model.summary()

# Check expected output shape
output_shape = model.output_shape

if isinstance(output_shape, list):
    raise ValueError("This evaluation script expects a single-output model.")

if output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model outputs {output_shape[-1]} classes, "
        f"but the label file contains {len(class_names)} classes."
    )

# ============================================================
# 4. LOAD THE INDEPENDENT TEST DATASET
# ============================================================

print("\nLoading independent test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Prefetch helps keep the evaluation pipeline efficient.
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

number_of_test_images = sum(
    int(images.shape[0]) for images, _ in test_ds
)

if number_of_test_images == 0:
    raise ValueError("The test dataset contains no images.")

print(f"Test images: {number_of_test_images}")
print(f"Number of classes: {len(class_names)}")

# ============================================================
# 5. RUN PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

all_true_labels = []
all_probabilities = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)

    all_true_labels.append(labels.numpy())
    all_probabilities.append(probabilities)

y_true = np.concatenate(all_true_labels).astype(np.int32)
y_prob = np.concatenate(all_probabilities, axis=0)

# Each image is assigned to the class with the highest probability.
y_pred = np.argmax(y_prob, axis=1)

if len(y_true) != len(y_pred):
    raise RuntimeError("Prediction count does not match label count.")

if not np.isfinite(y_prob).all():
    raise ValueError("Model predictions contain NaN or infinite values.")

print(f"Predictions completed: {len(y_pred)}")

# ============================================================
# 6. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 65)
print("OVERALL TEST PERFORMANCE")
print("=" * 65)
print(f"Test accuracy:          {accuracy:.4%}")
print(f"Macro precision:        {macro_precision:.4f}")
print(f"Macro recall:           {macro_recall:.4f}")
print(f"Macro F1-score:         {macro_f1:.4f}")
print(f"Weighted F1-score:      {weighted_f1:.4f}")

print("\nClassification report:")
print(report_text)

# ============================================================
# 7. SAVE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

# Raw counts
plt.figure(figsize=(11, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
)

plt.title("Custom CNN — Test Confusion Matrix (Counts)")
plt.xlabel("Predicted emotion")
plt.ylabel("True emotion")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

counts_path = OUTPUT_DIR / "test_confusion_matrix_counts.png"
plt.savefig(counts_path, dpi=300, bbox_inches="tight")
plt.close()

# Row-normalized percentages show how often each true class
# was predicted as each possible class.
row_totals = cm.sum(axis=1, keepdims=True)

cm_percent = np.divide(
    cm.astype(float),
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
) * 100.0

plt.figure(figsize=(11, 9))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    cbar_kws={"label": "Percentage of true class (%)"},
)

plt.title("Custom CNN — Normalized Test Confusion Matrix")
plt.xlabel("Predicted emotion")
plt.ylabel("True emotion")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

normalized_path = OUTPUT_DIR / "test_confusion_matrix_normalized.png"
plt.savefig(normalized_path, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 8. PLOT PER-CLASS F1-SCORES
# ============================================================

class_f1_scores = [
    report_dict[name]["f1-score"] for name in class_names
]

plt.figure(figsize=(10, 6))

bars = plt.bar(class_names, class_f1_scores)

plt.title("Custom CNN — Test F1-score by Emotion")
plt.xlabel("Emotion")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.grid(axis="y", alpha=0.3)

for bar, score in zip(bars, class_f1_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{score:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

f1_path = OUTPUT_DIR / "test_f1_by_emotion.png"
plt.savefig(f1_path, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 9. SAVE METRICS TO JSON
# ============================================================

metrics = {
    "model": MODEL_PATH.name,
    "test_dataset": str(TEST_DIR.relative_to(BASE_DIR)),
    "image_size": list(IMG_SIZE),
    "number_of_test_images": int(len(y_true)),
    "class_names": class_names,
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
}

metrics_path = OUTPUT_DIR / "test_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

# Save a readable text report as well.
report_path = OUTPUT_DIR / "test_classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("CUSTOM CNN — INDEPENDENT TEST EVALUATION\n")
    file.write("=" * 55 + "\n")
    file.write(f"Test images: {len(y_true)}\n")
    file.write(f"Accuracy: {accuracy:.4%}\n")
    file.write(f"Macro precision: {macro_precision:.4f}\n")
    file.write(f"Macro recall: {macro_recall:.4f}\n")
    file.write(f"Macro F1-score: {macro_f1:.4f}\n")
    file.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")
    file.write(report_text)

# ============================================================
# 10. SAVE IMAGE-LEVEL PREDICTIONS
# ============================================================

# Useful for reviewing individual errors and creating examples
# for the project presentation.
predictions_path = OUTPUT_DIR / "test_predictions.csv"

import csv

with open(predictions_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
    ])

    for true_idx, pred_idx, probabilities in zip(
        y_true, y_pred, y_prob
    ):
        writer.writerow([
            class_names[int(true_idx)],
            class_names[int(pred_idx)],
            bool(true_idx == pred_idx),
            f"{float(np.max(probabilities)):.6f}",
        ])

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)

print(f"Accuracy:     {accuracy:.4%}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")

print("\nSaved results:")
for path in [
    counts_path,
    normalized_path,
    f1_path,
    metrics_path,
    report_path,
    predictions_path,
]:
    print(f"  {path.relative_to(BASE_DIR)}")

print("\nImportant:")
print("- These metrics come from the independent test set.")
print("- Do not use the test set to tune the model repeatedly.")
print("- Report the results honestly, including weak classes.")
