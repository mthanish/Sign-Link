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
# Initialize engine outside the function to avoid re-initialization
try:
    engine = pyttsx3.init()
except ImportError:
    st.error("pyttsx3 driver not found. Please ensure it's installed correctly.")
    st.stop()
except RuntimeError:
    st.error("pyttsx3 driver failed to initialize.")
    st.stop()

def speak_text(text):
    import threading
    def run_speech():
        # Check if the engine is busy, wait if it is
        if engine._inLoop:
            engine.endLoop()
        engine.say(text)
        engine.runAndWait()
    # Use a daemon thread to not block the main app
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
    
    # <-- CHANGED: Initialize a variable to store the last spoken letter
    last_spoken_letter = ""
    # <-- CHANGED: Initialize a timestamp to control speech rate
    last_speech_time = 0

    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.stop()
    
    # Use a while loop for continuous streaming until stopped
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to read from webcam. Stopping.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        pred_class = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # (Your landmark processing code remains the same)
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_list), min(y_list)
                max_x, max_y = max(x_list), max(y_list)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    norm_x = (lm.x - min_x) / (max_x - min_x) if (max_x - min_x) != 0 else 0
                    norm_y = (lm.y - min_y) / (max_y - min_y) if (max_y - min_y) != 0 else 0
                    landmarks.extend([norm_x, norm_y])

                if len(landmarks) == model.n_features_in_:
                    prediction = model.predict([np.asarray(landmarks)])
                    pred_class = prediction[0]
                    
                    current_time = time.time()
                    
                    # <-- CHANGED: Check if the prediction is new and enough time has passed
                    if pred_class != last_spoken_letter and (current_time - last_speech_time > 1.5): # 1.5-second cooldown
                        speak_text(pred_class)
                        last_spoken_letter = pred_class # Update the last spoken letter
                        last_speech_time = current_time # Update the time of the last speech
                else:
                    pred_class = "Shape mismatch"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # <-- CHANGED: If no hand is detected, reset the last spoken letter
            last_spoken_letter = ""

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if pred_class:
            result_placeholder.text(f"🖐️ Detected Sign: {pred_class}")

        time.sleep(0.03)  # ~30 FPS

    cap.release()
    cv2.destroyAllWindows()