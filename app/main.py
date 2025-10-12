import streamlit as st
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyttsx3
import os
import time

st.title("SignLink: AI Sign Language Bridge")

# ------------------ Load Model Safely ------------------
MODEL_FILE = "data_processing/isl_model.pkl"

if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) == 0:
    st.error("Model file missing or empty. Make sure 'isl_model.pkl' exists and is properly saved.")
    st.stop()

try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except (EOFError, pickle.UnpicklingError):
    st.error("Failed to load the model. The file may be corrupted.")
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

st.write("Click 'Start Webcam' to detect signs.")

# ------------------ Webcam Stream ------------------
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.stop()

    for _ in range(1000):  # run for ~1000 frames
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        pred_class = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Flatten landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                # Ensure correct input shape
                landmarks_array = np.asarray(landmarks)
                if landmarks_array.shape[0] != model.n_features_in_:
                    pred_class = "Error: shape mismatch"
                else:
                    pred_class = model.predict([landmarks_array])[0]

                speak_text(pred_class)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update Streamlit display
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if pred_class:
            result_placeholder.text(f"Detected Sign: {pred_class}")

        time.sleep(0.03)  # ~30 FPS

    cap.release()
    cv2.destroyAllWindows()
