# Sign-Link: Hand Gesture Recognition Model

A real-time hand gesture recognition system using MediaPipe and TensorFlow for sign language translation.

## Features

- Real-time hand landmark detection using MediaPipe
- LSTM-based gesture classification with TensorFlow
- Data collection tools for custom gesture training
- Simple API for front-end integration
- Demo application for testing

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- MediaPipe
- Flask (for API)
- Other dependencies in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo

```
python demo.py
```

This will present two options:
1. Run the demo with an existing model
2. Collect training data and train a new model

### Training Your Own Model

To train your own model:

1. Run `python demo.py` and select option 2
2. Follow the prompts to collect data for each gesture
3. The system will guide you through recording sequences for each gesture
4. After data collection, you can train the model immediately

### Using the API

Start the API server:

```
python api.py
```

API Endpoints:
- `GET /health` - Check if API is running
- `POST /initialize` - Initialize the model
- `POST /predict` - Predict gesture from image
- `GET /available-gestures` - Get list of available gestures
- `POST /train` - Train model with provided data

Example API usage:

```python
import requests
import base64
import cv2

# Initialize model
requests.post('http://localhost:5000/initialize')

# Capture image
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Convert to base64
_, buffer = cv2.imencode('.jpg', frame)
img_base64 = base64.b64encode(buffer).decode('utf-8')

# Send for prediction
response = requests.post(
    'http://localhost:5000/predict',
    json={'image': img_base64}
)

print(response.json())
```

## Project Structure

- `hand_gesture_model.py` - Core model implementation
- `demo.py` - Demo application for testing
- `api.py` - Flask API for integration
- `requirements.txt` - Project dependencies

## Integration with Front-end

The API is designed to be easily integrated with any front-end application. Use the `/predict` endpoint to send images from the front-end and receive gesture predictions.

## Customization

You can customize the model by:
1. Adding more gesture classes in the `HandGestureRecognizer` class
2. Adjusting model parameters for better performance
3. Collecting more training data for improved accuracy