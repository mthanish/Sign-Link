import cv2
import mediapipe as mp
import pickle
import numpy as np
from text_to_speech.text_to_speech import speak_text  # import TTS function

MODEL_FILE = 'model/isl_model.pkl'
model = pickle.load(open(MODEL_FILE, 'rb'))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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

            cv2.putText(frame, pred_class, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Speak detected sign
            speak_text(pred_class)

    cv2.imshow("ISL Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
