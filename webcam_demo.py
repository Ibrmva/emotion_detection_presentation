import json
import html
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_detection_resnet50.keras"
LABELS_PATH = BASE_DIR / "resnet50_emotion_classes.json"

IMG_SIZE = (224, 224)
SMOOTHING_FRAMES = 5


# ============================================================
# 2. LOAD MODEL AND LABELS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    label_data = json.load(file)

if isinstance(label_data, dict):
    if all(str(i) in label_data for i in range(len(label_data))):
        class_names = [label_data[str(i)] for i in range(len(label_data))]
    else:
        class_names = list(label_data.values())
else:
    class_names = list(label_data)

if len(class_names) != 7:
    raise ValueError(
        f"Expected 7 emotion classes, but found {len(class_names)}: {class_names}"
    )

print("Loaded emotion classes:", class_names)


# ============================================================
# 3. FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


# ============================================================
# 4. STATIC CHARACTER
# ============================================================

def build_character(expression="neutral"):
    expression = expression.lower()

    styles = {
        "happy": {
            "color": "#38D9A9",
            "eyes": "M 83 103 Q 95 88 107 103 M 133 103 Q 145 88 157 103",
            "mouth": "M 99 137 Q 120 165 141 137",
            "label": "Happy",
        },
        "sad": {
            "color": "#74A9FF",
            "eyes": "M 83 105 Q 95 115 107 105 M 133 105 Q 145 115 157 105",
            "mouth": "M 99 151 Q 120 125 141 151",
            "label": "Sad",
        },
        "angry": {
            "color": "#FF6B6B",
            "eyes": "M 82 99 L 108 110 M 132 110 L 158 99",
            "mouth": "M 100 151 L 140 151",
            "label": "Angry",
        },
        "surprise": {
            "color": "#B197FC",
            "eyes": "M 95 103 C 83 85 108 85 95 103 M 145 103 C 133 85 158 85 145 103",
            "mouth": "M 120 133 C 103 133 103 160 120 160 C 137 160 137 133 120 133",
            "label": "Surprise",
        },
        "fear": {
            "color": "#9775FA",
            "eyes": "M 95 103 C 82 85 108 85 95 103 M 145 103 C 132 85 158 85 145 103",
            "mouth": "M 120 133 C 103 133 103 160 120 160 C 137 160 137 133 120 133",
            "label": "Fear",
        },
        "disgust": {
            "color": "#A9D96C",
            "eyes": "M 83 105 L 108 105 M 133 105 Q 145 115 157 105",
            "mouth": "M 100 145 Q 120 135 140 145",
            "label": "Disgust",
        },
        "neutral": {
            "color": "#63C7FF",
            "eyes": "M 83 103 L 107 103 M 133 103 L 157 103",
            "mouth": "M 100 143 L 140 143",
            "label": "Neutral",
        },
    }

    style = styles.get(expression, styles["neutral"])
    color = style["color"]
    eyes = style["eyes"]
    mouth = style["mouth"]
    label = style["label"]

    return f"""
    <div style="background:#172033;border:1px solid #334155;border-radius:16px;
                padding:16px;text-align:center;color:white;">
        <div style="font-size:13px;color:#94A3B8;margin-bottom:8px;">
            MODEL EXPRESSION VISUALIZATION
        </div>
        <svg viewBox="0 0 240 210" width="170" height="150"
             xmlns="http://www.w3.org/2000/svg">
            <line x1="120" y1="25" x2="120" y2="43" stroke="{color}"
                  stroke-width="5" stroke-linecap="round" />
            <circle cx="120" cy="19" r="9" fill="{color}" />
            <rect x="55" y="42" width="130" height="135" rx="32"
                  fill="#263449" stroke="{color}" stroke-width="5" />
            <path d="{eyes}" stroke="{color}" stroke-width="7" fill="none"
                  stroke-linecap="round" stroke-linejoin="round" />
            <path d="{mouth}" stroke="{color}" stroke-width="6" fill="none"
                  stroke-linecap="round" />
            <circle cx="75" cy="130" r="5" fill="{color}" opacity="0.45" />
            <circle cx="165" cy="130" r="5" fill="{color}" opacity="0.45" />
        </svg>
        <div style="color:{color};font-size:19px;font-weight:700;margin-top:4px;">
            {html.escape(label)}
        </div>
        <div style="color:#94A3B8;font-size:11px;margin-top:5px;">
            Visual representation of the model prediction
        </div>
    </div>
    """


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def make_score_dict(probabilities):
    return {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }


def get_recent_predictions(history):
    if not history:
        return "No predictions yet."

    predictions = [
        class_names[int(np.argmax(item))]
        for item in history
    ]
    return " → ".join(predictions)


def empty_frame_outputs(history, raw_text, smooth_text, face=None):
    return (
        raw_text,
        smooth_text,
        {},
        {},
        0,
        0,
        face,
        history,
        get_recent_predictions(history),
        build_character("neutral"),
    )


# ============================================================
# 6. MAIN FRAME PROCESSING
# ============================================================

def process_frame(frame, history):
    if history is None:
        history = []

    if frame is None:
        return empty_frame_outputs(
            history, "Waiting for camera...", "Waiting for prediction..."
        )

    # Gradio webcam frames are RGB.
    frame = np.array(frame, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
    )

    if len(faces) == 0:
        return empty_frame_outputs(
            history, "No face detected", "Waiting for face..."
        )

    # Select the largest detected face.
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    # Add a small margin around the face.
    margin_x = int(w * 0.12)
    margin_y = int(h * 0.12)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return empty_frame_outputs(
            history, "Face crop unavailable", "Waiting for prediction..."
        )

    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_input = np.expand_dims(face_resized.astype("float32"), axis=0)

    # Current-frame prediction.
    start_time = time.perf_counter()
    predictions = model(face_input, training=False).numpy()[0]
    inference_time = (time.perf_counter() - start_time) * 1000

    # This preserves the original app's behavior: model outputs are used directly.
    # Do not apply Softmax again unless the model's final layer outputs logits.
    raw_probabilities = predictions.astype(np.float32)

    raw_index = int(np.argmax(raw_probabilities))
    raw_expression = class_names[raw_index]
    raw_confidence = float(raw_probabilities[raw_index])

    # Average the latest five probability vectors.
    history = history + [raw_probabilities.tolist()]
    history = history[-SMOOTHING_FRAMES:]

    averaged_probabilities = np.mean(np.array(history), axis=0)
    smooth_index = int(np.argmax(averaged_probabilities))
    smooth_expression = class_names[smooth_index]
    smooth_confidence = float(averaged_probabilities[smooth_index])

    # Prediction stability across the recent frames.
    recent_classes = [int(np.argmax(item)) for item in history]
    stable_count = sum(1 for item in recent_classes if item == smooth_index)
    stability = (stable_count / len(recent_classes)) * 100

    raw_text = f"{raw_expression.capitalize()} ({raw_confidence * 100:.1f}%)"
    smooth_text = (
        f"{smooth_expression.capitalize()} "
        f"({smooth_confidence * 100:.1f}%)"
    )

    raw_scores = make_score_dict(raw_probabilities)
    smooth_scores = make_score_dict(averaged_probabilities)
    recent_text = get_recent_predictions(history)
    character_html = build_character(smooth_expression)

    return (
        raw_text,
        smooth_text,
        raw_scores,
        smooth_scores,
        round(stability, 1),
        round(inference_time, 2),
        face_crop,
        history,
        recent_text,
        character_html,
    )


# ============================================================
# 7. INTERACTIVE SOFTMAX DEMONSTRATION
# ============================================================

# These are illustrative logits for teaching the Softmax process.
# They are NOT taken from the live webcam prediction.
SOFTMAX_EXAMPLE_LOGITS = np.array(
    [2.5, 0.8, 0.3, 3.2, 1.5, 0.6, 1.0], dtype=np.float32
)


def softmax_view(stage):
    names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    logits = SOFTMAX_EXAMPLE_LOGITS
    exp_values = np.exp(logits)
    total = float(np.sum(exp_values))
    probabilities = exp_values / total

    stages = [
        (
            "01 / Raw Scores (Logits)",
            "The neural network produces seven raw numerical scores. These scores are not probabilities yet.",
            "A larger logit generally gives a category a stronger contribution before normalization.",
            "Raw model scores: z",
        ),
        (
            "02 / Exponentiation",
            "Softmax applies the exponential function e to each raw score.",
            "Exponentiation makes every value positive. Larger logits also receive relatively larger exponential values.",
            "Exponentiation: eᶻ",
        ),
        (
            "03 / Normalization",
            "Each exponential value is divided by the sum of all seven exponential values.",
            "The denominator is the sum of all seven exponentials. This makes the values add up to 1.",
            "Normalization: eᶻ / Σeᶻ",
        ),
        (
            "04 / Final Probability Distribution",
            "The normalized values are displayed as percentages for the seven expression categories.",
            "All seven values add up to 100%. The largest value is the model's top-scoring category, not a guarantee of correctness.",
            "Final Softmax probabilities",
        ),
    ]

    stage = max(0, min(int(stage), 3))
    title, description, explanation, formula = stages[stage]

    if stage == 0:
        values = logits
        labels = [f"{value:.2f}" for value in values]
        widths = np.abs(values) / max(float(np.max(np.abs(values))), 1e-8) * 100
        max_width = 100
    elif stage == 1:
        values = exp_values
        labels = [f"{value:.2f}" for value in values]
        widths = values / max(float(np.max(values)), 1e-8) * 100
        max_width = 100
    elif stage == 2:
        values = probabilities
        labels = [f"{value:.2f}" for value in values]
        widths = values * 100
        max_width = 100
    else:
        values = probabilities
        labels = [f"{value * 100:.1f}%" for value in values]
        widths = values * 100
        max_width = 100

    colors = [
        "#FF6B6B", "#A9D96C", "#9775FA", "#38D9A9",
        "#63C7FF", "#74A9FF", "#B197FC",
    ]

    rows = ""
    for name, label, width, color in zip(names, labels, widths, colors):
        width = max(0.0, min(float(width), max_width))
        rows += f"""
        <div class="softmax-row">
            <div class="softmax-row-label">
                <strong>{name}</strong>
                <span>{label}</span>
            </div>
            <div class="softmax-track">
                <div class="softmax-bar" style="--bar-width:{width:.2f}%;--bar-color:{color};"></div>
            </div>
        </div>
        """

    total_note = "Total probability: 100%" if stage == 3 else "Follow the transformation step by step"

    return f"""
    <style>
      .softmax-panel {{
        background: linear-gradient(145deg, #172033, #202D40);
        border: 1px solid #334155; border-radius: 16px;
        padding: 22px; color: #F8FAFC;
      }}
      .softmax-kicker {{ color:#38D9A9; font-size:11px; font-weight:800;
        letter-spacing:1.5px; }}
      .softmax-title {{ font-size:23px; font-weight:750; margin:9px 0; }}
      .softmax-description {{ color:#CBD5E1; line-height:1.65; }}
      .softmax-formula {{ background:#263449; color:#C4B5FD; padding:13px;
        border-radius:10px; text-align:center; font-weight:700;
        margin:17px 0 20px; }}
      .softmax-row {{ margin:14px 0; }}
      .softmax-row-label {{ display:flex; justify-content:space-between;
        gap:12px; font-size:14px; margin-bottom:7px; }}
      .softmax-row-label span {{ color:#CBD5E1; font-variant-numeric:tabular-nums; }}
      .softmax-track {{ height:13px; border-radius:20px; overflow:hidden;
        background:#334155; }}
      .softmax-bar {{ width:0; height:100%; border-radius:20px;
        background:var(--bar-color); animation:softmaxGrow 850ms cubic-bezier(.2,.8,.2,1) forwards; }}
      @keyframes softmaxGrow {{ from {{ width:0; }} to {{ width:var(--bar-width); }} }}
      .softmax-explanation {{ background:#263449; border-left:4px solid #38D9A9;
        border-radius:8px; padding:14px 16px; margin-top:20px; }}
      .softmax-explanation p {{ color:#E2E8F0; line-height:1.6; margin:7px 0 0; }}
      .softmax-footer {{ display:flex; justify-content:space-between; gap:10px;
        color:#94A3B8; font-size:11px; margin-top:17px; }}
      @media (prefers-reduced-motion: reduce) {{ .softmax-bar {{ animation:none; width:var(--bar-width); }} }}
    </style>
    <div class="softmax-panel">
      <div class="softmax-kicker">INTERACTIVE SOFTMAX DEMONSTRATION</div>
      <div class="softmax-title">{title}</div>
      <div class="softmax-description">{description}</div>
      <div class="softmax-formula">{formula}</div>
      {rows}
      <div class="softmax-explanation">
        <strong style="color:#38D9A9;">What does this step mean?</strong>
        <p>{explanation}</p>
      </div>
      <div class="softmax-footer">
        <span>STEP {stage + 1} OF 4</span>
        <span>{total_note}</span>
      </div>
    </div>
    """


def next_softmax_step(stage):
    stage = min(int(stage) + 1, 3)
    return stage, softmax_view(stage)


def previous_softmax_step(stage):
    stage = max(int(stage) - 1, 0)
    return stage, softmax_view(stage)


def reset_softmax_step():
    return 0, softmax_view(0)


# ============================================================
# 8. GRADIO INTERFACE
# ============================================================

custom_css = """
.gradio-container { max-width: 1450px !important; }
.main-title { text-align: center; margin-bottom: 4px; }
.subtitle { text-align: center; color: #94A3B8; margin-bottom: 20px; }
.section-title { font-size: 17px; font-weight: 700; margin-bottom: 8px; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="Real-Time Emotion Detection",
) as demo:
    gr.Markdown(
        """
        # Real-Time Facial Expression Recognition
        <div class="subtitle">
            ResNet50 | FER2013 | Seven Expression Categories
        </div>
        """
    )

    history_state = gr.State([])

    with gr.Row():
        # LEFT SIDE: WEBCAM AND FACE CROP
        with gr.Column(scale=1):
            gr.Markdown("### Live Camera")

            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Webcam",
            )

            face_output = gr.Image(label="Detected Face", height=180)

            gr.Markdown("### Prediction Results")

            raw_prediction = gr.Textbox(
                label="Current Frame Prediction",
                interactive=False,
            )
            smooth_prediction = gr.Textbox(
                label="Five-frame Averaged Prediction",
                interactive=False,
            )

        # RIGHT SIDE: PROBABILITY ANALYSIS AND CHARACTER
        with gr.Column(scale=1):
            gr.Markdown("## Expression Probability Analysis")

            gr.Markdown("### Current Frame Probabilities")
            raw_scores_output = gr.Label(
                num_top_classes=7,
                label="Current Frame — Seven Categories",
            )

            gr.Markdown("### Five-frame Averaged Probabilities")
            smooth_scores_output = gr.Label(
                num_top_classes=7,
                label="Averaged Frame — Seven Categories",
            )

            gr.Markdown("### Character Visualization")
            character_output = gr.HTML(value=build_character("neutral"))

    # SEPARATE TIME ANALYSIS SECTION
    gr.Markdown("---")
    gr.Markdown("## Time Analysis")

    with gr.Row():
        stability_output = gr.Number(
            label="Prediction Stability (%)",
            precision=1,
            interactive=False,
        )
        inference_output = gr.Number(
            label="Model Inference Time (ms)",
            precision=2,
            interactive=False,
        )

    gr.Markdown("### Recent Predictions")
    recent_output = gr.Textbox(
        interactive=False,
        label="Last Five Predictions",
    )

    gr.Markdown(
        """
        <div style="font-size:12px;color:#94A3B8;">
        Note: This application classifies visible facial expressions.
        It does not determine a person's actual internal emotional state.
        Prediction scores are model outputs and are not necessarily
        calibrated probabilities.
        </div>
        """
    )

    # INTERACTIVE SOFTMAX FOOTER
    gr.Markdown("---")
    gr.Markdown("## Explore How Softmax Works")
    gr.Markdown(
        "Use the buttons to follow the transformation from raw model scores "
        "to a probability distribution. The example values below are for "
        "learning and are separate from the live webcam prediction."
    )

    softmax_stage = gr.State(0)
    softmax_output = gr.HTML(value=softmax_view(0))

    with gr.Row():
        previous_btn = gr.Button("← Previous")
        next_btn = gr.Button("Next Step →", variant="primary")
        reset_btn = gr.Button("↻ Restart")

    next_btn.click(
        fn=next_softmax_step,
        inputs=[softmax_stage],
        outputs=[softmax_stage, softmax_output],
    )
    previous_btn.click(
        fn=previous_softmax_step,
        inputs=[softmax_stage],
        outputs=[softmax_stage, softmax_output],
    )
    reset_btn.click(
        fn=reset_softmax_step,
        inputs=[],
        outputs=[softmax_stage, softmax_output],
    )

    # LIVE WEBCAM STREAM
    webcam.stream(
        fn=process_frame,
        inputs=[webcam, history_state],
        outputs=[
            raw_prediction,
            smooth_prediction,
            raw_scores_output,
            smooth_scores_output,
            stability_output,
            inference_output,
            face_output,
            history_state,
            recent_output,
            character_output,
        ],
        stream_every=0.3,
        time_limit=3600,
    )


# ============================================================
# 9. LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue().launch(share=True)
