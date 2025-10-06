import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

DATASET_PATH = 'data_preprocessing/isl_dataset'
CSV_FILE = 'data_preprocessing/isl_landmarks.csv'

# Prepare CSV
with open(CSV_FILE, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = ['label'] + [f'{i}_{coord}' for i in range(21) for coord in ('x', 'y')]
    csv_writer.writerow(header)

    for label in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [label]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    csv_writer.writerow(row)

print("✅ Landmarks extracted and saved to isl_landmarks.csv")
