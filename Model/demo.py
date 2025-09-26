import cv2
import numpy as np
import time
from hand_gesture_model import HandGestureRecognizer

def main():
    # Initialize the recognizer
    # If you have a trained model, specify the path: model_path='path/to/model.h5'
    recognizer = HandGestureRecognizer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # FPS calculation variables
    prev_time = 0
    fps = 0
    
    print("Starting hand gesture recognition demo...")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Make prediction
        frame, result = recognizer.predict_in_real_time(frame)
        
        # Display result
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if result:
            gesture = result['gesture']
            confidence = result['confidence']
            
            # Display gesture and confidence
            cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display visual indicator for high confidence predictions
            if confidence > 0.8:
                # Draw a green rectangle around the frame
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 3)
        else:
            # No gesture detected
            cv2.putText(frame, "No gesture detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Hand Gesture Recognition Demo', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended")

def collect_training_data():
    """Function to collect training data for model training"""
    # Initialize the recognizer
    recognizer = HandGestureRecognizer()
    
    # Define gestures to collect
    gestures = [
        {"id": 0, "name": "hello"},
        {"id": 1, "name": "thank you"},
        {"id": 2, "name": "yes"},
        {"id": 3, "name": "no"},
        {"id": 4, "name": "help"}
    ]
    
    # Create data directory
    data_dir = "gesture_data"
    
    # Collect data for each gesture
    for gesture in gestures:
        print(f"\nCollecting data for gesture: {gesture['name']}")
        input("Press Enter to start collecting data for this gesture...")
        
        recognizer.collect_data(
            output_dir=data_dir,
            gesture_id=gesture["id"],
            gesture_name=gesture["name"],
            num_sequences=30,  # Number of sequences to collect
            sequence_length=30  # Frames per sequence
        )
    
    print("\nData collection completed!")
    print(f"Data saved to '{data_dir}' directory")
    
    # Ask if user wants to train the model
    train = input("\nDo you want to train the model now? (y/n): ")
    if train.lower() == 'y':
        print("\nTraining model...")
        recognizer.train(data_dir, epochs=50)
        print("\nTraining completed!")

if __name__ == "__main__":
    print("Hand Gesture Recognition Demo")
    print("1. Run demo with existing model")
    print("2. Collect training data and train new model")
    
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        main()
    elif choice == '2':
        collect_training_data()
    else:
        print("Invalid choice. Exiting.")