import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

CSV_FILE = 'data_preprocessing/isl_landmarks.csv'
MODEL_FILE = 'model/isl_model.pkl'

# Load data
data = pd.read_csv(CSV_FILE)
X = data.drop('label', axis=1)
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("✅ Training complete")
print("Accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open(MODEL_FILE, 'wb'))
print(f"✅ Model saved to {MODEL_FILE}")
