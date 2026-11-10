# ============================================================
# REAL-TIME FACIAL EXPRESSION RECOGNITION
# ResNet50 + FER2013
# Experiment: Mild class weighting
# ============================================================

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Restore the original training schedule for a fair comparison.
HEAD_EPOCHS = 12
FINE_TUNE_EPOCHS = 25
FINE_TUNE_LAST_LAYERS = 30

VALIDATION_SPLIT = 0.15
NUM_CLASSES = 7

DATA_DIR = Path("data/fer2013/train")
TEST_DIR = Path("data/fer2013/test")

# Use separate filenames so the original model/results are preserved.
BEST_MODEL = Path("best_resnet50_mild.keras")
DEPLOY_MODEL = Path("emotion_detection_resnet50_mild.keras")

LABELS_FILE = Path("resnet50_emotion_classes_mild.json")
HISTORY_FILE = Path("resnet50_training_history_mild.json")

CURVES_FILE = Path("resnet50_training_curves_mild.png")
REPORT_FILE = Path("resnet50_evaluation_report_mild.txt")

VAL_CM_FILE = Path("resnet50_confusion_matrix_val_mild.png")
TEST_CM_FILE = Path("resnet50_confusion_matrix_test_mild.png")

AUTOTUNE = tf.data.AUTOTUNE
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


# ============================================================
# 2. GPU CONFIGURATION
# ============================================================

def configure_gpu():
    """Enable GPU memory growth when a GPU is available."""

    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("GPU configuration warning:", error)

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", gpus)


# ============================================================
# 3. DATASET DISCOVERY
# ============================================================

def get_class_names():
    """Read and validate the seven emotion folders."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {DATA_DIR.resolve()}"
        )

    classes = sorted(
        folder.name
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, found "
            f"{len(classes)}: {classes}"
        )

    return classes


def collect_image_paths(classes):
    """Collect image paths and their numerical labels."""

    class_to_index = {
        name: index for index, name in enumerate(classes)
    }

    image_paths = []
    image_labels = []

    print("\n===== TRAINING DATA DISTRIBUTION =====")

    for class_name in classes:
        class_folder = DATA_DIR / class_name

        files = sorted(
            path for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        if not files:
            raise ValueError(f"No images found in {class_folder}")

        label = class_to_index[class_name]
        image_paths.extend(str(path) for path in files)
        image_labels.extend([label] * len(files))

        print(f"{class_name:10s}: {len(files):,}")

    return (
        np.array(image_paths),
        np.array(image_labels),
        class_to_index,
    )


# ============================================================
# 4. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(image_paths, image_labels, classes):
    """Split data while preserving each class proportion."""

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        image_labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=image_labels,
    )

    print("\n===== DATASET SPLIT =====")
    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))

    print("\n===== VALIDATION CLASS DISTRIBUTION =====")

    for index, class_name in enumerate(classes):
        count = int(np.sum(val_labels == index))
        print(f"{class_name:10s}: {count:,}")

        if count == 0:
            raise ValueError(f"Validation set is missing {class_name}")

    for index, class_name in enumerate(classes):
        if np.sum(train_labels == index) == 0:
            raise ValueError(f"Training set is missing {class_name}")

    return train_paths, val_paths, train_labels, val_labels


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def decode_and_resize_image(path, label):
    """Load an image, convert it to RGB, and resize it."""

    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    # Pixel values remain in [0, 255].
    # ResNet50 preprocessing is performed inside the model.
    return image, label


def create_tf_dataset(paths, labels, training=False):
    """Convert image paths and labels into a TensorFlow dataset."""

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        decode_and_resize_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


# ============================================================
# 6. MILD CLASS WEIGHTS
# ============================================================

def calculate_mild_class_weights(train_labels, classes):
    """
    Calculate mild class weights using the fourth root of
    inverse class frequency, then normalize and cap them.

    This keeps all images and gives minority classes somewhat
    more influence without the stronger weighting from the
    previous square-root experiment.
    """

    class_counts = np.bincount(
        train_labels,
        minlength=len(classes),
    )

    if np.any(class_counts == 0):
        raise ValueError("At least one class has no training images.")

    weights = (
        len(train_labels) / (len(classes) * class_counts)
    ) ** 0.25

    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.75, 1.50)

    class_weights = {
        index: float(weight)
        for index, weight in enumerate(weights)
    }

    print("\n===== MILD CLASS WEIGHTS =====")
    for index, class_name in enumerate(classes):
        print(
            f"{class_name:10s}: "
            f"{class_weights[index]:.3f} "
            f"({class_counts[index]:,} images)"
        )

    return class_weights


# ============================================================
# 7. LOAD ALL DATASETS
# ============================================================

def load_datasets():
    classes = get_class_names()
    print("\nEmotion classes:", classes)

    image_paths, image_labels, _ = collect_image_paths(classes)

    (
        train_paths,
        val_paths,
        train_labels,
        val_labels,
    ) = split_dataset(image_paths, image_labels, classes)

    train_dataset = create_tf_dataset(
        train_paths,
        train_labels,
        training=True,
    )

    val_dataset = create_tf_dataset(
        val_paths,
        val_labels,
        training=False,
    )

    test_dataset = None

    if TEST_DIR.exists():
        print("\n===== LOADING TEST DATASET =====")

        test_dataset = keras.utils.image_dataset_from_directory(
            TEST_DIR,
            image_size=IMG_SIZE,
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            label_mode="int",
            class_names=classes,
            shuffle=False,
        )

        print("Test images:", len(test_dataset.file_paths))
        test_dataset = test_dataset.prefetch(AUTOTUNE)

    class_weights = calculate_mild_class_weights(
        train_labels,
        classes,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        classes,
        class_weights,
    )


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================

def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04, fill_mode="nearest"),
            layers.RandomZoom(0.08, fill_mode="nearest"),
            layers.RandomTranslation(
                0.04,
                0.04,
                fill_mode="nearest",
            ),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


# ============================================================
# 9. RESNET50 MODEL ARCHITECTURE
# ============================================================

def build_model(num_classes):
    inputs = keras.Input(
        shape=(224, 224, 3),
        name="rgb_face_224",
    )

    x = build_augmentation()(inputs)
    x = keras.applications.resnet50.preprocess_input(x)

    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    # Stage 1: freeze the pretrained backbone.
    base_model.trainable = False
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="emotion_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="fer2013_resnet50",
    )

    return model, base_model


# ============================================================
# 10. TRAINING CALLBACKS
# ============================================================

def build_checkpoint():
    return keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )


def build_early_stopping():
    return keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )


def build_lr_scheduler():
    return keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )


# ============================================================
# 11. TRAINING STAGE 1: CLASSIFICATION HEAD
# ============================================================

def train_classification_head(
    model,
    train_dataset,
    val_dataset,
    class_weights,
):
    print("\n========================================")
    print("STAGE 1: TRAIN CLASSIFICATION HEAD")
    print("========================================")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            build_checkpoint(),
            build_early_stopping(),
            build_lr_scheduler(),
        ],
    )

    return history


# ============================================================
# 12. TRAINING STAGE 2: FINE-TUNING
# ============================================================

def train_fine_tuning(
    model,
    base_model,
    train_dataset,
    val_dataset,
    class_weights,
):
    print("\n========================================")
    print("STAGE 2: FINE-TUNE RESNET50")
    print("========================================")

    base_model.trainable = True

    # Fine-tune the final 30 layers, as in the original experiment.
    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            build_checkpoint(),
            build_early_stopping(),
            build_lr_scheduler(),
        ],
    )

    return history


# ============================================================
# 13. SAVE TRAINING HISTORY AND CURVES
# ============================================================

def save_training_history(histories):
    combined = {}

    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(
                float(value) for value in values
            )

    HISTORY_FILE.write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        combined.get("accuracy", []),
        label="Training",
    )
    axes[0].plot(
        combined.get("val_accuracy", []),
        label="Validation",
    )
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(
        combined.get("loss", []),
        label="Training",
    )
    axes[1].plot(
        combined.get("val_loss", []),
        label="Validation",
    )
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CURVES_FILE, dpi=150)
    plt.close()

    print("\nTraining history saved:", HISTORY_FILE)
    print("Training curves saved:", CURVES_FILE)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    dataset,
    classes,
    dataset_name,
    confusion_matrix_path,
    report_path,
):
    if dataset is None:
        print(f"{dataset_name} dataset is unavailable.")
        return

    actual_labels = []
    predicted_probabilities = []

    for images, labels in dataset:
        predictions = model(images, training=False).numpy()
        predicted_probabilities.append(predictions)
        actual_labels.append(labels.numpy())

    y_true = np.concatenate(actual_labels)
    probabilities = np.concatenate(predicted_probabilities)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    summary = (
        f"\n===== {dataset_name.upper()} RESULTS =====\n"
        f"Images evaluated: {len(y_true)}\n"
        f"Accuracy: {accuracy:.4%}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"{report}\n"
    )

    print(summary)

    with report_path.open("a", encoding="utf-8") as file:
        file.write(summary)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
    )

    normalized_cm = (
        cm.astype(float)
        / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(
        normalized_cm,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(len(classes)),
        labels=classes,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("Actual emotion")
    ax.set_title(f"{dataset_name} Confusion Matrix")

    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized_cm[row, column]
            count = cm[row, column]

            ax.text(
                column,
                row,
                f"{value:.2f}\n({count})",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()

    print("Confusion matrix saved:", confusion_matrix_path)


# ============================================================
# 15. SAVE FINAL MODEL AND CLASS LABELS
# ============================================================

def save_final_model(classes):
    if not BEST_MODEL.exists():
        raise FileNotFoundError(
            "Best checkpoint was not created. "
            "Check whether training completed at least one epoch."
        )

    best_model = keras.models.load_model(BEST_MODEL)
    best_model.save(DEPLOY_MODEL)

    LABELS_FILE.write_text(
        json.dumps(classes, indent=2),
        encoding="utf-8",
    )

    print("\nFinal model saved:", DEPLOY_MODEL.resolve())
    print("Emotion labels saved:", LABELS_FILE.resolve())

    return best_model


# ============================================================
# 16. MAIN EXECUTION PIPELINE
# ============================================================

def main():
    print("\n========================================")
    print("REAL-TIME EMOTION RECOGNITION")
    print("ResNet50 + FER2013")
    print("MILD CLASS-WEIGHTING EXPERIMENT")
    print("========================================")

    configure_gpu()

    # Remove only this experiment's checkpoint.
    # The original model and checkpoint are not touched.
    if BEST_MODEL.exists():
        BEST_MODEL.unlink()
        print("\nRemoved previous mild-experiment checkpoint.")

    (
        train_dataset,
        val_dataset,
        test_dataset,
        classes,
        class_weights,
    ) = load_datasets()

    model, base_model = build_model(num_classes=len(classes))

    print("\n===== MODEL ARCHITECTURE =====")
    model.summary()

    history_stage1 = train_classification_head(
        model,
        train_dataset,
        val_dataset,
        class_weights,
    )

    history_stage2 = train_fine_tuning(
        model,
        base_model,
        train_dataset,
        val_dataset,
        class_weights,
    )

    save_training_history([
        history_stage1,
        history_stage2,
    ])

    best_model = save_final_model(classes)

    # Clear only this experiment's report.
    if REPORT_FILE.exists():
        REPORT_FILE.unlink()

    evaluate_model(
        best_model,
        val_dataset,
        classes,
        "Validation",
        VAL_CM_FILE,
        REPORT_FILE,
    )

    # For the final experiment, evaluate on the independent test set.
    # Avoid using test results to tune further model settings.
    if test_dataset is not None:
        evaluate_model(
            best_model,
            test_dataset,
            classes,
            "Test",
            TEST_CM_FILE,
            REPORT_FILE,
        )

    print("\n========================================")
    print("TRAINING AND EVALUATION COMPLETED")
    print("========================================")
    print("Model:", DEPLOY_MODEL.resolve())
    print("Labels:", LABELS_FILE.resolve())
    print("Report:", REPORT_FILE.resolve())
    print("Curves:", CURVES_FILE.resolve())


if __name__ == "__main__":
    main()
