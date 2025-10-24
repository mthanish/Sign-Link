import streamlit as st
import numpy as np
import pickle
import cv2
import mediapipe as mp
import pyttsx3
import os
import time
from collections import Counter # <-- NEW: We need this to find the most common letter

st.title("SignLink: Word Builder")

# --- Initialize Session State (App Memory) ---
# This is the most important part. It stores data even when the script reruns.
if "current_word" not in st.session_state:
    st.session_state.current_word = ""

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
try:
    engine = pyttsx3.init()
except Exception:
    st.error("Failed to initialize pyttsx3. Please check your audio drivers.")
    st.stop()

def speak_text(text):
    import threading
    def run_speech():
        if engine._inLoop:
            engine.endLoop()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech, daemon=True).start()

# ------------------ Mediapipe Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ------------------ MAIN APP LAYOUT ------------------

# 1. Display the current word being built
st.header(f"Current Word: {st.session_state.current_word}")
st.markdown("---")

# 2. Create columns for the buttons
col1, col2, col3 = st.columns(3)

with col1:
    # 3. The "Add Letter" button
    if st.button("Add Letter (Start Cam) 📸"):
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        
        if not cap.isOpened():
            st.error("Cannot access webcam.")
            st.stop()
            
        st.info("Webcam starting... Hold your sign steady for 3 seconds.")
        detected_letters = [] # A list to store all detections
        
        # Run the webcam for ~3 seconds (90 frames)
        for _ in range(90): 
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam feed ended early.")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            pred_class = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # (Landmark processing code is the same)
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
                        
                        # Add the detected letter to our list
                        if pred_class:
                            detected_letters.append(pred_class)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if pred_class:
                result_placeholder.text(f"Detecting: {pred_class}")

            time.sleep(0.033) # ~30 FPS

        # --- After the loop finishes ---
        cap.release()
        cv2.destroyAllWindows()
        
        # Now, find the most common (most stable) letter detected
        if detected_letters:
            most_common_letter = Counter(detected_letters).most_common(1)[0][0]
            
            # Add this letter to our word in session_state
            st.session_state.current_word += most_common_letter
            
            st.success(f"Added letter: '{most_common_letter}'")
            time.sleep(1) # Pause for a second to show the success message
            
            # Rerun the script to update the "Current Word" header
            st.rerun()
        else:
            st.warning("No sign detected. Please try again.")

with col2:
    # 4. The "Speak Word" button
    if st.button("Speak Word 🗣️"):
        if st.session_state.current_word:
            speak_text(st.session_state.current_word)
        else:
            st.warning("No word to speak.")

with col3:
    # 5. The "Clear" button
    if st.button("Clear Word ❌"):
        st.session_state.current_word = ""
        # Rerun to update the header
        st.rerun()