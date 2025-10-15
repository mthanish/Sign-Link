import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Dataset directory
DATA_DIR = "data_processing/data"
IMG_SIZE = 128

all_data = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_file in tqdm(os.listdir(dir_path), desc=f"Processing {dir_}"):
        img_path = os.path.join(dir_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]

                # Normalize to [0, 1]
                min_x, min_y = min(x_list), min(y_list)
                max_x, max_y = max(x_list), max(y_list)

                row = []
                for lm in hand_landmarks.landmark:
                    norm_x = (lm.x - min_x) / (max_x - min_x)
                    norm_y = (lm.y - min_y) / (max_y - min_y)
                    row.extend([norm_x, norm_y])

                row.append(dir_)
                all_data.append(row)

# Create column names dynamically
num_coords = len(all_data[0]) - 1
columns = [f'coord_{i+1}' for i in range(num_coords)] + ['label']

df = pd.DataFrame(all_data, columns=columns)
df.to_csv('data_processing/isl_landmarks.csv', index=False)
print("✅ CSV created successfully and saved as data_processing/isl_landmarks.csv")
