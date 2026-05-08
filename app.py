from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# CLASS NAMES
# =========================
class_names = [
    'Anxiety',
    'Blood_Pressure',
    'Constipation',
    'Cough',
    'Cut',
    'Disease',
    'Doctor',
    'Emergency',
    'Fever',
    'Headache',
    'Infection',
    'Nurse',
    'Operation',
    'Stomachache',
    'Treatment'
]

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
)

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({
            "success": False,
            "message": "No hand detected"
        })

    landmarks = []

    # =========================
    # EXTRACT LANDMARKS
    # =========================
    for hand_landmarks in results.multi_hand_landmarks:

        for lm in hand_landmarks.landmark:

            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])

    # =========================
    # PAD SINGLE HAND
    # =========================
    if len(results.multi_hand_landmarks) == 1:
        landmarks.extend([0] * 63)

    # =========================
    # EXACTLY 126 FEATURES
    # =========================
    landmarks = landmarks[:126]

    if len(landmarks) < 126:
        landmarks.extend([0] * (126 - len(landmarks)))

    features = np.array(landmarks).reshape(1, -1)

    prediction = model.predict(features)[0]

    label = class_names[prediction]

    return jsonify({
        "success": True,
        "prediction": label
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)