from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
from hand_gesture_model import HandGestureRecognizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the recognizer
recognizer = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Hand gesture recognition API is running'
    })

@app.route('/initialize', methods=['POST'])
def initialize_model():
    """Initialize the model with optional model path"""
    global recognizer
    
    data = request.json
    model_path = data.get('model_path', None)
    num_classes = data.get('num_classes', 5)
    
    try:
        recognizer = HandGestureRecognizer(model_path=model_path, num_classes=num_classes)
        return jsonify({
            'status': 'success',
            'message': 'Model initialized successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to initialize model: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict gesture from image"""
    global recognizer
    
    # Check if model is initialized
    if recognizer is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized. Call /initialize first.'
        }), 400
    
    # Get image from request
    if 'image' not in request.json:
        return jsonify({
            'status': 'error',
            'message': 'No image provided'
        }), 400
    
    try:
        # Decode base64 image
        image_data = request.json['image']
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        
        # Convert base64 to numpy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Make prediction
        _, result = recognizer.predict_in_real_time(frame)
        
        if result:
            return jsonify({
                'status': 'success',
                'gesture': result['gesture'],
                'confidence': result['confidence']
            })
        else:
            return jsonify({
                'status': 'success',
                'gesture': None,
                'confidence': 0.0,
                'message': 'No gesture detected'
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing image: {str(e)}'
        }), 500

@app.route('/available-gestures', methods=['GET'])
def available_gestures():
    """Get list of available gestures"""
    global recognizer
    
    # Check if model is initialized
    if recognizer is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized. Call /initialize first.'
        }), 400
    
    return jsonify({
        'status': 'success',
        'gestures': recognizer.labels
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train model with provided data path"""
    global recognizer
    
    # Check if model is initialized
    if recognizer is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized. Call /initialize first.'
        }), 400
    
    data = request.json
    data_path = data.get('data_path')
    epochs = data.get('epochs', 50)
    batch_size = data.get('batch_size', 32)
    
    if not data_path:
        return jsonify({
            'status': 'error',
            'message': 'No data path provided'
        }), 400
    
    try:
        # Train the model
        recognizer.train(data_path, epochs=epochs, batch_size=batch_size)
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'model_path': recognizer.model_path
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error training model: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)