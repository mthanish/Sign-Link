import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import platform

# Check if we're on Windows
IS_WINDOWS = platform.system() == "Windows"

# Import MediaPipe if not on Windows
if not IS_WINDOWS:
    import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self, model_path=None, num_classes=5):
        # Initialize hand detection
        self.use_mediapipe = not IS_WINDOWS
        
        if self.use_mediapipe:
            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            # Initialize OpenCV-based hand detection
            # We'll use a simple skin color detection approach
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
            if self.hand_cascade.empty():
                print("Warning: Hand cascade classifier not found. Using backup method.")
                self.hand_cascade = None
        
        # Model parameters
        self.num_classes = num_classes
        self.sequence_length = 30
        self.model_path = model_path
        
        # Initialize or load model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = self._build_model()
            print("New model created")
        
        # For real-time prediction
        self.sequence = []
        self.predictions = []
        self.threshold = 0.7
        
        # Gesture labels (can be customized)
        self.labels = {
            0: 'hello',
            1: 'thank you',
            2: 'yes',
            3: 'no',
            4: 'help'
        }
    
    def _build_model(self):
        """Build LSTM model for gesture recognition"""
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 63)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        return model
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from a frame"""
        if self.use_mediapipe:
            # MediaPipe approach
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = self.hands.process(image)
            
            landmarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the image
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmark coordinates
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Flatten landmarks to a 1D array
            flattened_landmarks = np.array(landmarks).flatten() if landmarks else np.zeros(21*3)
        else:
            # OpenCV-based approach
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a binary mask for skin color
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            landmarks = []
            if contours:
                # Find the largest contour (assuming it's the hand)
                max_contour = max(contours, key=cv2.contourArea)
                
                # Get convex hull of the contour
                hull = cv2.convexHull(max_contour)
                
                # Draw the contour and hull on the frame
                cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
                
                # Get defects in the hull
                hull_indices = cv2.convexHull(max_contour, returnPoints=False)
                if len(hull_indices) > 3:  # Need at least 4 points for convexity defects
                    try:
                        defects = cv2.convexityDefects(max_contour, hull_indices)
                        if defects is not None:
                            # Extract key points from defects
                            for i in range(min(7, len(defects))):  # Limit to 7 defects
                                s, e, f, _ = defects[i, 0]
                                start = tuple(max_contour[s][0])
                                end = tuple(max_contour[e][0])
                                far = tuple(max_contour[f][0])
                                
                                # Add these points as landmarks
                                landmarks.append([start[0] / frame.shape[1], start[1] / frame.shape[0], 0])
                                landmarks.append([end[0] / frame.shape[1], end[1] / frame.shape[0], 0])
                                landmarks.append([far[0] / frame.shape[1], far[1] / frame.shape[0], 0])
                                
                                # Draw circles at these points
                                cv2.circle(frame, far, 5, [0, 0, 255], -1)
                    except:
                        pass
                
                # Get the centroid of the contour
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    landmarks.append([cx / frame.shape[1], cy / frame.shape[0], 0])
                    cv2.circle(frame, (cx, cy), 8, [255, 0, 0], -1)
            
            # Ensure we have 21 landmarks (same as MediaPipe) by padding or truncating
            while len(landmarks) < 21:
                landmarks.append([0, 0, 0])
            
            if len(landmarks) > 21:
                landmarks = landmarks[:21]
            
            # Flatten landmarks to a 1D array
            flattened_landmarks = np.array(landmarks).flatten()
        
        return frame, flattened_landmarks
    
    def preprocess_landmarks(self, landmarks):
        """Preprocess landmarks for model input"""
        # Ensure we have the right number of landmarks
        if len(landmarks) < 63:
            # Pad with zeros if we have fewer landmarks
            landmarks = np.pad(landmarks, (0, 63 - len(landmarks)))
        elif len(landmarks) > 63:
            # Truncate if we have more landmarks
            landmarks = landmarks[:63]
        
        return landmarks
    
    def predict_in_real_time(self, frame):
        """Make predictions in real-time"""
        # Extract landmarks
        frame, landmarks = self.extract_hand_landmarks(frame)
        
        # Preprocess landmarks
        landmarks = self.preprocess_landmarks(landmarks)
        
        # Update sequence
        self.sequence.append(landmarks)
        self.sequence = self.sequence[-self.sequence_length:]
        
        # Make prediction when we have enough frames
        result = None
        if len(self.sequence) == self.sequence_length:
            # Prepare input for model
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            
            # Get prediction with highest confidence
            prediction_index = np.argmax(res)
            confidence = res[prediction_index]
            
            if confidence > self.threshold:
                # If confidence is high enough, return the prediction
                result = {
                    'gesture': self.labels[prediction_index],
                    'confidence': float(confidence)
                }
                
                # Add to predictions list
                self.predictions.append(prediction_index)
                
                # Only keep last 10 predictions
                self.predictions = self.predictions[-10:]
                
                # Get most common prediction from last 10
                if len(self.predictions) > 5:
                    most_common = np.bincount(self.predictions).argmax()
                    result = {
                        'gesture': self.labels[most_common],
                        'confidence': float(confidence)
                    }
        
        return frame, result
    
    def prepare_training_data(self, data_path):
        """Prepare training data from collected sequences"""
        sequences, labels = [], []
        
        # Load data from CSV files in the data directory
        for gesture_folder in os.listdir(data_path):
            gesture_path = os.path.join(data_path, gesture_folder)
            
            if os.path.isdir(gesture_path):
                gesture_id = int(gesture_folder.split('_')[0])
                
                for sequence_folder in os.listdir(gesture_path):
                    sequence_path = os.path.join(gesture_path, sequence_folder)
                    
                    if os.path.isdir(sequence_path):
                        sequence_data = []
                        
                        # Load all frames in the sequence
                        for frame_file in sorted(os.listdir(sequence_path)):
                            if frame_file.endswith('.csv'):
                                frame_path = os.path.join(sequence_path, frame_file)
                                landmarks = pd.read_csv(frame_path).values.flatten()
                                landmarks = self.preprocess_landmarks(landmarks)
                                sequence_data.append(landmarks)
                        
                        # Ensure sequence is of correct length
                        if len(sequence_data) == self.sequence_length:
                            sequences.append(sequence_data)
                            labels.append(gesture_id)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)
        
        return X, y
    
    def train(self, data_path, epochs=50, batch_size=32):
        """Train the model on collected data"""
        # Prepare data
        X, y = self.prepare_training_data(data_path)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Create logs directory for TensorBoard
        log_dir = os.path.join('logs', 'hand_gestures')
        os.makedirs(log_dir, exist_ok=True)
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[TensorBoard(log_dir=log_dir)]
        )
        
        # Evaluate the model
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        y_true = np.argmax(y_val, axis=1)
        
        print("Validation Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=list(self.labels.values())))
        
        # Save the model
        if not self.model_path:
            self.model_path = 'hand_gesture_model.h5'
        
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return self.model
    
    def collect_data(self, output_dir, gesture_id, gesture_name, num_sequences=30, sequence_length=30):
        """Collect training data for a specific gesture"""
        # Create output directory
        gesture_dir = os.path.join(output_dir, f"{gesture_id}_{gesture_name}")
        os.makedirs(gesture_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        for sequence in range(num_sequences):
            sequence_dir = os.path.join(gesture_dir, str(sequence))
            os.makedirs(sequence_dir, exist_ok=True)
            
            print(f"Collecting sequence {sequence+1}/{num_sequences} for gesture '{gesture_name}'")
            print("Press 'q' to start recording")
            
            # Wait for user to press 'q' to start recording
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, f"Prepare for gesture: {gesture_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to start recording", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('Collect Data', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Collect frames for the sequence
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract landmarks
                _, landmarks = self.extract_hand_landmarks(frame)
                
                # Save landmarks to CSV
                landmarks_df = pd.DataFrame(landmarks.reshape(1, -1))
                landmarks_df.to_csv(os.path.join(sequence_dir, f"{frame_num}.csv"), index=False)
                
                # Display progress
                cv2.putText(frame, f"Recording: {frame_num+1}/{sequence_length}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('Collect Data', frame)
                cv2.waitKey(1)
            
            print(f"Sequence {sequence+1}/{num_sequences} collected")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection for gesture '{gesture_name}' completed")

# Example usage
if __name__ == "__main__":
    # Initialize the recognizer
    recognizer = HandGestureRecognizer()
    
    # Collect data for training (uncomment to use)
    # data_dir = "gesture_data"
    # os.makedirs(data_dir, exist_ok=True)
    # recognizer.collect_data(data_dir, gesture_id=0, gesture_name="hello", num_sequences=30)
    
    # Train the model (uncomment to use)
    # recognizer.train(data_dir, epochs=50)
    
    # Real-time prediction
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make prediction
        frame, result = recognizer.predict_in_real_time(frame)
        
        # Display result
        if result:
            gesture = result['gesture']
            confidence = result['confidence']
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()