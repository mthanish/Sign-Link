import streamlit as st
from speech_to_text.speech_to_text import listen
from text_to_speech.text_to_speech import speak_text
import threading
import cv2
import mediapipe as mp
import pickle
import numpy as np

st.title("SignLink: AI Sign Language Bridge")

MODEL_FILE = 'model/isl_model.pkl'
model = pickle.load(open(MODEL_FILE, 'rb'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

st.write("Click 'Start Webcam' to detect signs.")

if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                prediction = model.predict([np.asarray(landmarks)])
                pred_class = prediction[0]
                st.text(f"Detected Sign: {pred_class}")
                speak_text(pred_class)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("ISL Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
