
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "emotion_detection_resnet50.keras"
LABELS_PATH = BASE_DIR / "resnet50_emotion_classes.json"

IMG_SIZE = (224, 224)

# ============================================================
# LOAD RESNET50 MODEL
# ============================================================

print("Loading ResNet50 emotion classification model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    classes = json.load(file)

if isinstance(classes, dict):
    classes = [classes[str(i)] for i in range(len(classes))]

print("ResNet50 loaded successfully.")
print("Classes:", classes)

# ============================================================
# FACE DETECTOR
# ============================================================

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load OpenCV face detector.")

# ============================================================
# PREDICTION
# ============================================================

def predict_frame(frame):

    if frame is None:
        return None, "Turn on your webcam to begin."

    # Gradio supplies RGB images
    rgb = np.asarray(frame).copy()

    # Convert to grayscale for face detection only
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        return rgb, "No face detected — face the camera."

    # Select the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Crop the face from the original RGB image
    face = rgb[y:y + h, x:x + w]

    # Resize to ResNet50 input dimensions
    face = cv2.resize(
        face,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # Keep RGB pixel values in [0, 255].
    # ResNet50 preprocessing is already inside the model.
    batch = face.astype(np.float32)
    batch = np.expand_dims(batch, axis=0)

    # Predict
    probabilities = model.predict(
        batch,
        verbose=0
    )[0]

    best_index = int(np.argmax(probabilities))
    label = classes[best_index]
    confidence = float(probabilities[best_index]) * 100

    # Draw bounding box
    cv2.rectangle(
        rgb,
        (x, y),
        (x + w, y + h),
        (0, 220, 100),
        2,
    )

    # Display prediction
    cv2.putText(
        rgb,
        f"{label}: {confidence:.1f}%",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 100),
        2,
        cv2.LINE_AA,
    )

    result = (
        f"**Predicted facial expression:** {label.title()}\n\n"
        f"**Model confidence:** {confidence:.1f}%\n\n"
        "This is a facial-expression classification prediction, "
        "not a reliable measurement of someone's internal emotional state."
    )

    return rgb, result


# ============================================================
# GRADIO WEB APPLICATION
# ============================================================

with gr.Blocks(
    title="Facial Expression Classification"
) as demo:

    gr.Markdown(
        """
        # Facial Expression Classification

        Real-time facial expression classification using
        a ResNet50 model trained on the FER-2013 dataset.

        **How to test:** Allow camera access, face the camera,
        and try different facial expressions.

        The model predicts seven expression categories:
        Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
        """
    )

    with gr.Row():

        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Live webcam",
        )

        annotated = gr.Image(
            type="numpy",
            label="Detected face",
        )

    result = gr.Markdown(
        "Waiting for webcam frames..."
    )

    webcam.stream(
        fn=predict_frame,
        inputs=webcam,
        outputs=[annotated, result],
        stream_every=0.4,
        time_limit=60,
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(share=True)
