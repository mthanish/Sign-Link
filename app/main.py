import streamlit as st
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyttsx3
import os
import time

st.title("SignLink: AI Sign Language Bridge")

# ------------------ Load Model ------------------
MODEL_FILE = "data_processing/isl_model.pkl"

if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) == 0:
    st.error("Model file missing or empty. Make sure 'isl_model.pkl' exists and is properly saved.")
    st.stop()

try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except Exception:
    st.error("Failed to load the model. It may be corrupted.")
    st.stop()

# ------------------ Text-to-Speech ------------------
engine = pyttsx3.init()
def speak_text(text):
    import threading
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

# ------------------ Mediapipe Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

st.write("Click 'Start Webcam' to begin real-time sign detection.")

# ------------------ Webcam Stream ------------------
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.stop()

    for _ in range(1000):  # runs for ~1000 frames
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        pred_class = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_list), min(y_list)
                max_x, max_y = max(x_list), max(y_list)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    norm_x = (lm.x - min_x) / (max_x - min_x)
                    norm_y = (lm.y - min_y) / (max_y - min_y)
                    landmarks.extend([norm_x, norm_y])

                if len(landmarks) == model.n_features_in_:
                    pred_class = model.predict([np.asarray(landmarks)])[0]
                    speak_text(pred_class)
                else:
                    pred_class = "Shape mismatch"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if pred_class:
            result_placeholder.text(f"🖐️ Detected Sign: {pred_class}")

        time.sleep(0.03)  # ~30 FPS

    cap.release()
    cv2.destroyAllWindows()
