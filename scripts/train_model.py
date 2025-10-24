import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

CSV_FILE = 'data_processing/isl_landmarks.csv'
MODEL_FILE = 'data_processing/isl_model.pkl'

# Load flattened CSV
data = pd.read_csv(CSV_FILE)

# Separate features and label
X = data.drop('label', axis=1)
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save trained model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
